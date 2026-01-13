# Numerai Market Neutral Strategy (v2.2)
**Production-Grade Quantitative Architecture for Global Equity Ranking**

![Status](https://img.shields.io/badge/Status-Production-success)
![Platform](https://img.shields.io/badge/Platform-Numerai-white)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👨‍💼 Executive Summary
**Author:** Dr. Brian Penrod, DBA  
**Objective:** Generate rank-ordered predictive signals for the Numerai Hedge Fund Tournament using a "Twin-Engine" Ensemble architecture.  
**Key Differentiator:** Strict adherence to **Chronological Regime Separation** (No Look-Ahead Bias) and **Feature Neutrality**.

---

## 🏗️ System Architecture (The "Twin Engines")
*For a detailed architectural deep dive, see [SYSTEM_MAP.md](SYSTEM_MAP.md).*

```mermaid
graph TD
    %% --- STYLES ---
    classDef raw fill:#2d2d2d,stroke:#555,stroke-width:2px,color:#fff;
    classDef process fill:#0D47A1,stroke:#000,stroke-width:2px,color:#fff;
    classDef model fill:#1B5E20,stroke:#00cc96,stroke-width:2px,color:#fff;
    classDef output fill:#4a148c,stroke:#aa00ff,stroke-width:2px,color:#fff;

    Raw[Numerai Parquet]:::raw -->|Polars Ingestion| Features[Medium Set ~780 Features]:::process
    Features -->|Time-Series Split| Train[Training Set]:::raw
    
    subgraph ENGINES ["THE TWIN ENGINES"]
        Train --> LGBM[Engine 1: LightGBM]:::model
        Train --> XGB[Engine 2: XGBoost]:::model
    end

    LGBM -->|0.5| Ensemble((Weighted Signal)):::output
    XGB -->|0.5| Ensemble
    Ensemble -->|Rank & Upload| API[Numerai API]:::output
```
Key Documentation
📄 Strategy White Paper: A detailed "Research Note" style explanation of the math, validation logic, and variance reduction theory.

🗺️ System Map: Full architectural diagram and strategic doctrine definitions.

⚙️ Core Capabilities
1. The "Chronological Firewall"
Standard K-Fold cross-validation is rejected to prevent look-ahead bias. This model utilizes an Era-wise TimeSeriesSplit, training on the first 80% of eras and validating strictly on the subsequent 20%.

2. High-Performance Ingestion (Polars)
Utilizes Rust-based Polars for lazy-loading of the ~50GB+ dataset, enabling rapid iteration on Google Colab Pro+ (A100 GPU) environments.

3. Regime Adaptation
The ensemble combines Gradient-based One-Side Sampling (LightGBM) with Histogram-based Splitting (XGBoost) to capture both deep non-linear interactions and broad structural signals.
🚀 Quick Start
Prerequisites
pip install numerapi lightgbm xgboost pandas polars pyarrow
Execution (Production Mode)
# Authenticate & Engage Watchdog
import numerapi
napi = numerapi.NumerAPI(public_id="YOUR_ID", secret_key="YOUR_KEY")

if napi.check_round_open():
    print(">>> ENGAGING TWIN ENGINES...")
    # ... (Load Pipeline)
© 2026 Dr. Brian Penrod. All Rights Reserved.
