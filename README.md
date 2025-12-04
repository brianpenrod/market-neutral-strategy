# market-neutral-strategy
Algorithmic trading model predicting stock rankings for the Numerai Hedge Fund tournament.
# Numerai Tournament: Market Neutral Strategy

### Project Overview
This project implements a machine learning pipeline for the **Numerai Hedge Fund Tournament**, a crowdsourced quantitative trading competition. The objective is to predict the rank-ordered performance of nearly 5,000 global equities based on obfuscated financial features.

This model contributes to a meta-model that powers a **Market Neutral Long/Short Equity Strategy**, designed to generate alpha regardless of broad market direction (zero beta).

### Technical Approach
* **Model Architecture:** Light Gradient Boosting Machine (LightGBM) Regressor.
* **Feature Engineering:** Utilized Numerai's "Small" feature set (v5.1) for memory-efficient processing on cloud instances.
* **Target:** 20-day returns relative to the market (Rank).
* **Validation:** Time-series split based on "Eras" to prevent look-ahead bias (data leakage).

### Workflow
1.  **Data Ingestion:** Automated API retrieval of encrypted financial data via `numerapi`.
2.  **Training:** Supervised learning on historical eras using decision tree ensembles.
3.  **Inference:** Generating daily rank predictions (0 to 1) for live market data.
4.  **Deployment:** Automated submission to the Numerai leaderboard.

### Future Development
* **Ensembling:** Integrating XGBoost and CatBoost models to reduce variance.
* **Risk Management:** Implementing feature neutralization to limit exposure to single factors (e.g., momentum, volatility).
* **Infrastructure:** Migrating from Cloud SaaS to local high-performance compute (Ryzen 9 / RTX Architecture).

---
*Author: Brian Penrod - DBA (Finance) & Quantitative Analyst*
