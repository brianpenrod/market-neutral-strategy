# ==============================================================================
# PROJECT: NUMERAI MARKET NEUTRAL STRATEGY (ZERO BETA)
# AUTHOR: Dr. Brian Penrod, DBA
# DESCRIPTION: Automated pipeline for ranking 5,000 equities using LightGBM
# ==============================================================================

import numerapi
import pandas as pd
import lightgbm as lgb
import json
import os

# --- CONFIGURATION ---
# SECURITY PROTOCOL: We use environment variables to protect API keys.
# This prevents your secrets from being visible if you share this code.
# When running locally, you can set these in your terminal or replace 
# the 'os.getenv' part with your string keys (BUT DO NOT UPLOAD KEYS TO GITHUB).
PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID", "PASTE_YOUR_PUBLIC_ID_HERE_FOR_LOCAL_RUN")
SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY", "PASTE_YOUR_SECRET_KEY_HERE_FOR_LOCAL_RUN")

def run_pipeline():
    print("--- SITREP: INITIATING PIPELINE ---")
    
    # 1. INITIALIZE API
    # Connects to the Numerai Tournament infrastructure
    napi = numerapi.NumerAPI(PUBLIC_ID, SECRET_KEY)
    
    # 2. DOWNLOAD METADATA & DATA
    # [cite_start]We use the v5.1 "Small" feature set to optimize for RAM constraints [cite: 98, 108]
    print("Downloading metadata...")
    napi.download_dataset("v5.1/features.json", "features.json")
    
    print("Identifying 'Small' feature set...")
    with open("features.json", "r") as f:
        feature_metadata = json.load(f)
    small_features = feature_metadata["feature_sets"]["small"]
    print(f"Strategic Decision: Using {len(small_features)} features for regularization.")

    # 3. LOAD TRAINING DATA
    # Loading only specific columns to save memory (Logistics)
    print("Loading Training Data (Parquet)...")
    training_data = pd.read_parquet(
        "v5.1/train.parquet",
        columns=["target"] + small_features
    )
    
    # 4. TRAIN MODEL (LightGBM)
    # [cite_start]Hyperparameters tuned for generalization to prevent overfitting [cite: 108, 109]
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(
        n_estimators=2000,      # High number of trees
        learning_rate=0.01,     # Slow learning rate for robustness
        max_depth=5,            # Shallow depth to avoid memorizing noise
        num_leaves=32,
        colsample_bytree=0.1,   # Feature fractioning
        random_state=42
    )
    
    model.fit(
        training_data[small_features],
        training_data["target"]
    )
    print("Model Training Complete.")

    # 5. INFERENCE (LIVE PREDICTION)
    # [cite_start]Downloading the new live data for the current era [cite: 26]
    print("Downloading Live Data...")
    napi.download_dataset("v5.1/live.parquet", "live.parquet")
    live_data = pd.read_parquet(
        "live.parquet",
        columns=["id"] + small_features # Ensure we get IDs for submission
    )
    
    print("Generating Predictions...")
    live_predictions = model.predict(live_data[small_features])
    
    # 6. FORMAT SUBMISSION
    # [cite_start]Aligning predictions with their corresponding Stock IDs [cite: 29]
    if "id" in live_data.columns:
        ids = live_data["id"]
    else:
        ids = live_data.index
        
    submission = pd.Series(live_predictions, index=ids).to_frame("prediction")
    submission.to_csv("submission.csv")
    print("Submission file generated: submission.csv")

    # 7. UPLOAD (READY TO FIRE)
    # Uncomment the lines below when you are ready to automate the full loop
    # current_model_id = list(napi.get_models().values())[0]
    # napi.upload_predictions("submission.csv", model_id=current_model_id)
    # print("Submission uploaded to Leaderboard.")

if __name__ == "__main__":
    run_pipeline()

# Strategy: Load only the "Small" feature set to optimize memory
with open("features.json", "r") as f:
    feature_metadata = json.load(f)
small_features = feature_metadata["feature_sets"]["small"]
