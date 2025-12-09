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
# BEST PRACTICE: Load keys from environment variables to prevent leakage
# When running locally, you can set these in your terminal or IDE
PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID", "YOUR_PUBLIC_ID_HERE_IF_RUNNING_LOCALLY")
SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY", "YOUR_SECRET_KEY_HERE_IF_RUNNING_LOCALLY")

def run_pipeline():
    print("--- SITREP: INITIATING PIPELINE ---")
    
    # 1. INITIALIZE API
    napi = numerapi.NumerAPI(PUBLIC_ID, SECRET_KEY)
    
    # 2. DOWNLOAD METADATA & DATA
    # We use the v5.1 "Small" feature set to optimize for RAM constraints
    print("Downloading metadata...")
    napi.download_dataset("v5.1/features.json", "features.json")
    
    print("Identifying 'Small' feature set...")
    with open("features.json", "r") as f:
        feature_metadata = json.load(f)
    small_features = feature_metadata["feature_sets"]["small"]
    print(f"Strategic Decision: Using {len(small_features)} features for regularization.")

    # 3. LOAD TRAINING DATA
    print("Loading Training Data (Parquet)...")
    training_data = pd.read_parquet(
        "v5.1/train.parquet",
        columns=["target"] + small_features
    )
    
    # 4. TRAIN MODEL (LightGBM)
    # Hyperparameters tuned for generalization (preventing overfitting)
    print("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.1,
        random_state=42
    )
    
    model.fit(
        training_data[small_features],
        training_data["target"]
    )
    print("Model Training Complete.")

    # 5. INFERENCE (LIVE PREDICTION)
    print("Downloading Live Data...")
    napi.download_dataset("v5.1/live.parquet", "live.parquet")
    live_data = pd.read_parquet(
        "live.parquet",
        columns=["id"] + small_features # Ensure we get IDs for submission
    )
    
    print("Generating Predictions...")
    live_predictions = model.predict(live_data[small_features])
    
    # 6. FORMAT SUBMISSION
    # Handle index vs column edge case for IDs
    if "id" in live_data.columns:
        ids = live_data["id"]
    else:
        ids = live_data.index
        
    submission = pd.Series(live_predictions, index=ids).to_frame("prediction")
    submission.to_csv("submission.csv")
    print("Submission file generated: submission.csv")

    # 7. UPLOAD (OPTIONAL - Uncomment to enable auto-submit)
    # model_id = list(napi.get_models().values())[0]
    # napi.upload_predictions("submission.csv", model_id=model_id)
    # print("Submission uploaded to Leaderboard.")

if __name__ == "__main__":
    run_pipeline()


