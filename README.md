# PROJECT: KINETIC ZERO (v2.3)
**Production-Grade Quantitative Architecture for Global Equity Ranking**

![Status](https://img.shields.io/badge/Status-Production-success)
![Platform](https://img.shields.io/badge/Platform-Numerai-white)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👨‍💼 Executive Summary
**Author:** Dr. Brian Penrod, DBA  
**Codename:** **KINETIC ZERO**  
**Objective:** Generate rank-ordered predictive signals for the Numerai Tournament using a **Twin-Engine** ensemble (LightGBM + XGBoost) reinforced by **OLS orthogonalization / neutralization**.  
**Key Differentiator:** Strict adherence to **chronological regime separation** (no look-ahead bias) and **Phase 4 feature neutralization**.

---

## 🏗️ System Architecture (Kinetic Zero Protocol)
*For a detailed architectural deep dive, see [SYSTEM_MAP.md](SYSTEM_MAP.md).*

```mermaid
graph TD
    %% --- STYLES ---
    classDef raw fill:#2d2d2d,stroke:#555,stroke-width:2px,color:#fff;
    classDef process fill:#0D47A1,stroke:#000,stroke-width:2px,color:#fff;
    classDef model fill:#1B5E20,stroke:#00cc96,stroke-width:2px,color:#fff;
    classDef logic fill:#b71c1c,stroke:#ef553b,stroke-width:2px,color:#fff;
    classDef output fill:#4a148c,stroke:#aa00ff,stroke-width:2px,color:#fff;

    Raw[Numerai Parquet]:::raw -->|Polars Ingestion| Features[Medium Set ~780 Features]:::process
    Features -->|Era-aware Split| Train[Training Set]:::raw

    subgraph ENGINES ["THE TWIN ENGINES"]
        Train --> LGBM[Engine 1: LightGBM]:::model
        Train --> XGB[Engine 2: XGBoost]:::model
    end

    LGBM -->|0.5| Ensemble((Weighted Signal)):::output
    XGB -->|0.5| Ensemble

    Ensemble -->|OLS Orthogonalization| Neutral[Kinetic Neutralization]:::logic
    Neutral -->|Rank & Upload| API[Numerai API]:::output
```
📚 Key Documentation

📄 Strategy White Paper: [whitepaper.md](https://github.com/brianpenrod/market-neutral-strategy/blob/09b9abc051107a5fa2ed859236dafbd4316b8902/whitepaper.md)

🗺️ System Map: [SYSTEM_MAP.md](SYSTEM_MAP.md)

⚙️ Core Capabilities
1) The “Chronological Firewall”

Standard K-Fold cross-validation is rejected to prevent look-ahead bias. This model uses an era-wise chronological split, training on the first ~80% of eras and validating strictly on the subsequent ~20%.

2) High-Performance Ingestion (Polars)

Utilizes Polars for fast Parquet ingestion and transformation, enabling rapid iteration in local or hosted environments (e.g., Colab).

3) Regime Adaptation (Twin-Engine Ensemble)

The ensemble combines LightGBM and XGBoost to capture both deep non-linear interactions and broader structural signals.

4) Kinetic Neutralization (Phase 4)

Post-processing applies OLS orthogonalization / neutralization to reduce linear correlations between predictions and the selected feature set, isolating more idiosyncratic signal from common risk factors.

🚀 Quick Start
```markdown
```bash
pip install -U numerapi lightgbm xgboost pandas polars pyarrow scipy
```
Execution (Production Mode)
```import os
from numerapi import NumerAPI

napi = NumerAPI(
    public_id=os.getenv("NAPI_PUBLIC_ID", "YOUR_ID"),
    secret_key=os.getenv("NAPI_SECRET_KEY", "YOUR_KEY"),
)

if napi.check_round_open():
    print(">>> ENGAGING KINETIC ZERO...")
    # TODO: load -> train -> ensemble -> neutralize -> rank -> upload
else:
    print(">>> Round is closed. Standing by.")
```
© 2026 Dr. Brian Penrod. All Rights Reserved.
