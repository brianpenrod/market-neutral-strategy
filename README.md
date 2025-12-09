# 🛡️ Project: Numerai Market Neutral Strategy (Zero Beta)
### Automated Quantitative Trading Pipeline | LightGBM | Python

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-LightGBM-green)
![Strategy](https://img.shields.io/badge/Strategy-Market%20Neutral-orange)
![Status](https://img.shields.io/badge/Status-Live%20Trading-red)

---

## 📋 Executive Summary
**Objective:** To engineer a machine learning model capable of predicting the *relative* performance of 5,000 global equities, independent of market direction.

**The Problem:** Traditional investing is exposed to "Beta" (Market Risk). If the S&P 500 drops 20%, most portfolios drop 20%.
**The Solution:** A **Market Neutral (Zero Beta)** strategy. By predicting stock *rankings* (0 to 1) rather than raw prices, this system targets pure Alpha. It generates returns based on the model's intelligence, not the market's mood.

---

## 🎖️ Commander's Intent (The Philosophy)
As a retired **US Army Special Operations CSM**, I approach financial modeling with the same discipline used in mission planning:
1.  **Risk Mitigation First:** We do not bet on the direction of the wind (Market direction). We bet on the performance of the unit (Stock fundamentals).
2.  **Simplicity is Speed:** We use a lightweight feature set to ensure the model is robust and generalizable, preventing "overfitting" (seeing patterns that aren't there).
3.  **Skin in the Game:** This model participates in the Numerai Tournament, where cryptocurrency (NMR) is staked on its accuracy. Poor performance results in a "Burn" (loss of capital). Accountability is absolute.

---

## ⚙️ Technical Architecture

### 1. The Engine: Light Gradient Boosting Machine (LightGBM)
I selected `LightGBM` for its efficiency with tabular financial data. It uses tree-based learning to identify non-linear relationships between obfuscated market features and future returns.

### 2. The Logistics: "Small" Feature Set (v5.1)
Instead of using the full 2,000+ feature set, this pipeline utilizes the **"Small" feature set (~40 features)**.
* **Tactical Reason:** Allows for training on standard hardware (RAM efficiency).
* **Strategic Reason:** Acts as a form of **Regularization**. By limiting the input data to the most "mission-essential" signals, we force the model to learn broad, robust trends rather than memorizing noise.

### 3. The Validation: Era-Wise Split
Financial data is "Non-Stationary" (the rules change over time). To prevent **Look-Ahead Bias**, we strictly validate the model using Era-Wise Time Series splits, ensuring we never use future data to predict the past.

---

## 💻 Code Structure (The Pipeline)

**Step 1: Ingestion & Feature Selection**
Automated API calls via `numerapi` to retrieve the latest encrypted market data.
```python
import numerapi
import pandas as pd
import json

# Initialize API (Keys are stored securely in environment variables)
napi = numerapi.NumerAPI()

# Smart Download: Only fetching essential metadata to save bandwidth
napi.download_dataset("v5.1/features.json", "features.json")

# Strategy: Load only the "Small" feature set to optimize memory
with open("features.json", "r") as f:
    feature_metadata = json.load(f)
small_features = feature_metadata["feature_sets"]["small"]
