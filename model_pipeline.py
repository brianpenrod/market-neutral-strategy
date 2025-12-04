# --- STEP 1: INSTALL TOOLS ---
!pip install numerapi lightgbm pandas

import numerapi
import pandas as pd
import lightgbm as lgb
import json

# --- STEP 2: CONNECT TO NUMERAI ---
# REPLACE THESE WITH YOUR KEYS!
napi = numerapi.NumerAPI ("YOUR_PUBLIC_ID_GOES_HERE", "YOUR_SECRET_KEY_GOES_HERE")

# --- STEP 3: GET DATA (SMART DOWNLOAD) ---
print("Downloading metadata and files...")
napi.download_dataset("v5.1/features.json", "features.json")
napi.download_dataset("v5.1/train.parquet", "train.parquet")
napi.download_dataset("v5.1/live.parquet", "live.parquet")

# --- STEP 4: LOAD ONLY "SMALL" FEATURE SET ---
# This is the secret weapon to save RAM.
print("Identifying 'Small' feature set...")
with open("features.json", "r") as f:
    feature_metadata = json.load(f)

# We grab only the essential features (~40 instead of 2000)
small_features = feature_metadata["feature_sets"]["small"]
print(f"Loading only {len(small_features)} features to save RAM.")

# Load training data but ONLY the specific columns we need
training_data = pd.read_parquet(
    "train.parquet",
    columns=["target"] + small_features
)

# Load live data (same small columns + the ID column so we can submit)
live_data = pd.read_parquet(
    "live.parquet",
    columns=["id"] + small_features
)

# --- STEP 5: TRAIN LIGHTWEIGHT MODEL ---
print("Training lightweight model...")
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=32,
    colsample_bytree=0.1
)

model.fit(
    training_data[small_features],
    training_data["target"]
)

# --- STEP 6: PREDICT & SUBMIT (FIXED) ---
print("Generating predictions...")
live_predictions = model.predict(live_data[small_features])

# CRITICAL FIX: The "id" is hiding in the index, not a column
# We check where it is to be safe
if "id" in live_data.columns:
    ids = live_data["id"]
else:
    ids = live_data.index

# Format the submission
submission = pd.Series(live_predictions, index=ids).to_frame("prediction")
submission.to_csv("submission.csv")

# Send it
print("Uploading submission...")
# Auto-detect your model ID
model_id = list(napi.get_models().values())[0]
napi.upload_predictions("submission.csv", model_id=model_id)

print("MISSION COMPLETE: Submission uploaded!")
