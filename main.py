# --- STEP 1: INSTALL TOOLS ---
!pip install numerapi lightgbm xgboost pandas polars pyarrow psutil

import numerapi
import pandas as pd
import polars as pl
import lightgbm as lgb
import xgboost as xgb
import json
import os
import time
import psutil
import torch
from google.colab import userdata

# --- STEP 1.5: HARDWARE DIAGNOSTICS ---
print("\n--- SYSTEM DIAGNOSTICS ---")
# Check RAM
ram_gb = psutil.virtual_memory().total / 1e9
print(f"TOTAL RAM DETECTED: {ram_gb:.2f} GB")

if ram_gb < 25:
    print("!!! WARNING: High-RAM not detected. Go to Runtime > Change Runtime Type > Runtime Shape > High-RAM !!!")
else:
    print(">>> SYSTEM GREEN: High-RAM Operational.")

# Check GPU
if torch.cuda.is_available():
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    print("!!! WARNING: GPU Not Detected. Enable GPU in Runtime Settings. !!!")
print("-" * 30)

# --- STEP 2: CONNECT TO NUMERAI ---
public_id = userdata.get('NUMERAI_PUBLIC_ID')
secret_key = userdata.get('NUMERAI_SECRET_KEY')
napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

# --- STEP 2.5: THE WATCHDOG ---
print("\nChecking Round Status...")
while True:
    if napi.check_round_open():
        print(">>> ROUND IS OPEN. ENGAGING.")
        break
    else:
        print("... Round CLOSED. Holding pattern (60s) ...")
        time.sleep(60)

# --- STEP 3: DATA INGESTION (MEDIUM SET) ---
print("\nDownloading metadata and files...")
napi.download_dataset("v5.1/features.json", "features.json")
napi.download_dataset("v5.1/train.parquet", "train.parquet")
napi.download_dataset("v5.1/live.parquet", "live.parquet")

print("Identifying 'Medium' feature set...")
with open("features.json", "r") as f:
    feature_metadata = json.load(f)

# STRATEGIC SHIFT: MEDIUM FEATURE SET
# This moves us from 42 features -> ~780 features
features = feature_metadata["feature_sets"]["medium"]
print(f"Loading {len(features)} features...")

print("Ingesting Training Data (Polars Accelerated)...")
# Using Polars to handle the larger dataset efficiently
# We read with Polars, then convert to Pandas for the models
q = (
    pl.scan_parquet("train.parquet")
    .select(["era", "target"] + features)
)
training_data = q.collect().to_pandas()

print(f"Data Loaded. Shape: {training_data.shape}")

print("Ingesting Live Data...")
live_data = pd.read_parquet(
    "live.parquet",
    columns=["id"] + features
)

# --- STEP 4: VALIDATION SEQUENCE (HEAVY TWIN ENGINE) ---
print("\n--- INITIATING VALIDATION SEQUENCE (MEDIUM SET) ---")

# 1. Prepare Split
training_data["era"] = training_data["era"].astype(int)
eras = training_data["era"].unique()
eras.sort()
cutoff = int(len(eras) * 0.8)
split_era = eras[cutoff]

train_split = training_data[training_data["era"] <= split_era]
val_split = training_data[training_data["era"] > split_era]

# 2. Train Engine 1: LightGBM
print(f"Training LightGBM (Medium Set)...")
model_lgbm = lgb.LGBMRegressor(
    n_estimators=2500, # Increased for more features
    learning_rate=0.01,
    max_depth=6,       # Slightly deeper for complex interactions
    num_leaves=64,     # Increased capacity
    colsample_bytree=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    device='gpu'       # GPU Acceleration
)
model_lgbm.fit(train_split[features], train_split["target"])

# 3. Train Engine 2: XGBoost
print(f"Training XGBoost (Medium Set)...")
model_xgb = xgb.XGBRegressor(
    n_estimators=2500,
    learning_rate=0.01,
    max_depth=6,
    colsample_bytree=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    device='cuda'      # GPU Acceleration
)
model_xgb.fit(train_split[features], train_split["target"])

# 4. Generate Ensemble Predictions
print("Calculating Ensemble Performance...")
pred_lgbm = model_lgbm.predict(val_split[features])
pred_xgb = model_xgb.predict(val_split[features])

# 50/50 Ensemble
ensemble_preds = (0.5 * pred_lgbm) + (0.5 * pred_xgb)
val_split.loc[:, "prediction"] = ensemble_preds

# 5. Score the Ensemble
era_scores = val_split.groupby("era").apply(
    lambda x: x["prediction"].corr(x["target"], method="spearman")
)

val_mean = era_scores.mean()
val_std = era_scores.std()
val_sharpe = val_mean / val_std

print("-" * 30)
print(f"MEDIUM ENS MEAN CORR: {val_mean:.4f}")
print(f"MEDIUM ENS STD DEV:   {val_std:.4f}")
print(f"MEDIUM ENS SHARPE:    {val_sharpe:.4f}")
print("-" * 30)

# --- STEP 5: PRODUCTION DEPLOYMENT ---
print("\n--- RETRAINING FULL MODELS ---")
model_lgbm.fit(training_data[features], training_data["target"])
model_xgb.fit(training_data[features], training_data["target"])

print("Generating Live Predictions...")
live_lgbm = model_lgbm.predict(live_data[features])
live_xgb = model_xgb.predict(live_data[features])

live_ensemble = (0.5 * live_lgbm) + (0.5 * live_xgb)

if "id" in live_data.columns:
    ids = live_data["id"]
else:
    ids = live_data.index

submission = pd.Series(live_ensemble, index=ids).to_frame("prediction")
submission.to_csv("submission.csv")

print("Uploading Submission...")
model_id = list(napi.get_models().values())[0]
napi.upload_predictions("submission.csv", model_id=model_id)

print(f"MISSION COMPLETE: 'Medium' Set Deployed. (Val Sharpe: {val_sharpe:.2f})")
