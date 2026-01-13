"""
PROJECT: KINETIC ZERO (v2.3)
STRATEGY ENGINE: TWIN-ENGINE MARKET NEUTRAL
AUTHOR: Dr. Brian Penrod, DBA
STATUS: PRODUCTION (COLAB PRO+)

DESCRIPTION:
    - Architecture: Heterogeneous Ensemble (LightGBM + XGBoost).
    - Validation: Chronological Era-Wise Split (Stationarity Defense).
    - Risk Mgmt: Phase 4 Feature Neutralization (Orthogonalization).
"""

# --- STEP 1: INSTALL & SETUP ---
# (Uncomment the line below if running in a fresh Colab environment)
# !pip install numerapi lightgbm xgboost pandas polars pyarrow psutil scipy

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
import numpy as np
from google.colab import userdata

# --- CONFIGURATION: KINETIC ZERO PROTOCOLS ---
NEUTRALIZATION_RATIO = 0.50  # 50% Feature Neutralization (Phase 4 Standard)
RISK_MODE = "PRODUCTION"     # Reporting Tag

# --- STEP 1.5: HARDWARE DIAGNOSTICS ---
print("\n--- SYSTEM DIAGNOSTICS: KINETIC ZERO ---")
ram_gb = psutil.virtual_memory().total / 1e9
print(f"TOTAL RAM: {ram_gb:.2f} GB")

if ram_gb < 25:
    print("!!! WARNING: High-RAM not detected. Upgrade Runtime. !!!")
else:
    print(">>> SYSTEM GREEN: High-RAM Operational.")

if torch.cuda.is_available():
    print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
else:
    print("!!! WARNING: GPU Not Detected. Enable GPU. !!!")
print("-" * 30)

# --- MATHEMATICAL MODULE (PHASE 4: NEUTRALIZATION) ---
def neutralize(df, columns, target="prediction", proportion=1.0):
    """
    PHASE 4: KINETIC NEUTRALIZATION
    Objective: Remove linear correlation between 'prediction' and 'risk factors'.
    Math: OLS Residualization -> Pred_neutral = Pred - (Beta * Features)
    """
    print(f"... Neutralizing exposure to {len(columns)} features ({proportion*100}%)...")
    scores = df[target]
    exposures = df[columns].values
    
    # The 'Alpha' Calculation: Pred - (Beta * Factors)
    scores_neutral = scores - proportion * exposures.dot(
        np.linalg.pinv(exposures).dot(scores)
    )
    return scores_neutral

# --- STEP 2: CONNECT & WATCHDOG ---
public_id = userdata.get('NUMERAI_PUBLIC_ID')
secret_key = userdata.get('NUMERAI_SECRET_KEY')
napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)

print("\nChecking Round Status...")
while True:
    if napi.check_round_open():
        print(">>> ROUND IS OPEN. ENGAGING KINETIC ZERO.")
        break
    else:
        print("... Round CLOSED. Holding pattern (60s) ...")
        time.sleep(60)

# --- STEP 3: DATA INGESTION (MEDIUM SET) ---
print("\n--- INGESTION PHASE ---")
napi.download_dataset("v5.1/features.json", "features.json")
napi.download_dataset("v5.1/train.parquet", "train.parquet")
napi.download_dataset("v5.1/live.parquet", "live.parquet")

with open("features.json", "r") as f:
    feature_metadata = json.load(f)

features = feature_metadata["feature_sets"]["medium"]
print(f"Targeting {len(features)} Features (Medium Set)...")

print("Accelerated Loading (Polars)...")
q = pl.scan_parquet("train.parquet").select(["era", "target"] + features)
training_data = q.collect().to_pandas()
print(f"Training Data Shape: {training_data.shape}")

print("Loading Live Data...")
live_data = pd.read_parquet("live.parquet", columns=["id"] + features)

# --- STEP 4: VALIDATION (TWIN ENGINE + NEUTRALIZATION) ---
print("\n--- VALIDATION SEQUENCE (v2.3) ---")

# 1. Chronological Split (The Firewall)
training_data["era"] = training_data["era"].astype(int)
eras = training_data["era"].unique()
eras.sort()
cutoff = int(len(eras) * 0.8)
split_era = eras[cutoff]

train_split = training_data[training_data["era"] <= split_era]
val_split = training_data[training_data["era"] > split_era]

# 2. Train Engine 1: LightGBM
print(f"Training Engine 1: LightGBM (The Sniper)...")
model_lgbm = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    num_leaves=64,
    colsample_bytree=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
    device='gpu'
)
model_lgbm.fit(train_split[features], train_split["target"])

# 3. Train Engine 2: XGBoost
print(f"Training Engine 2: XGBoost (The Spotter)...")
model_xgb = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=6,
    colsample_bytree=0.1,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    device='cuda'
)
model_xgb.fit(train_split[features], train_split["target"])

# 4. Ensemble & Neutralize
print("Generating Signals...")
pred_lgbm = model_lgbm.predict(val_split[features])
pred_xgb = model_xgb.predict(val_split[features])
val_split.loc[:, "prediction"] = (0.5 * pred_lgbm) + (0.5 * pred_xgb)

# APPLY PHASE 4: NEUTRALIZATION
val_split["prediction_neutral"] = neutralize(
    val_split, 
    columns=features, 
    target="prediction", 
    proportion=NEUTRALIZATION_RATIO
)

# 5. Score Comparison
def get_sharpe(df, col):
    return df.groupby("era").apply(lambda x: x[col].corr(x["target"])).mean() / \
           df.groupby("era").apply(lambda x: x[col].corr(x["target"])).std()

raw_sharpe = get_sharpe(val_split, "prediction")
neut_sharpe = get_sharpe(val_split, "prediction_neutral")

print("-" * 30)
print(f"RAW SHARPE:        {raw_sharpe:.4f}")
print(f"KINETIC SHARPE:    {neut_sharpe:.4f} (Target: >1.5)")
print("-" * 30)

# --- STEP 5: PRODUCTION DEPLOYMENT ---
print("\n--- DEPLOYING TO PRODUCTION ---")
# Retrain on Full Data
model_lgbm.fit(training_data[features], training_data["target"])
model_xgb.fit(training_data[features], training_data["target"])

# Live Predictions
live_lgbm = model_lgbm.predict(live_data[features])
live_xgb = model_xgb.predict(live_data[features])
live_ensemble = (0.5 * live_lgbm) + (0.5 * live_xgb)

# Live Neutralization
# Note: We need a dataframe context for neutralization
live_df = pd.DataFrame(live_ensemble, columns=["prediction"])
live_df = pd.concat([live_df, live_data[features].reset_index(drop=True)], axis=1)

final_signal = neutralize(
    live_df, 
    columns=features, 
    target="prediction", 
    proportion=NEUTRALIZATION_RATIO
)

# Formatting
if "id" in live_data.columns:
    ids = live_data["id"]
else:
    ids = live_data.index

submission = pd.Series(final_signal, index=ids).to_frame("prediction")
submission.to_csv("submission.csv")

# Upload
model_id = list(napi.get_models().values())[0]
napi.upload_predictions("submission.csv", model_id=model_id)

print(f"MISSION COMPLETE: KINETIC ZERO Deployed. (Neutralization: {NEUTRALIZATION_RATIO*100}%)")
