# PROJECT: KINETIC ZERO (v2.3)
**Production-Grade Quantitative Architecture for Global Equity Ranking**

![Status](https://img.shields.io/badge/Status-Production-success)
![Platform](https://img.shields.io/badge/Platform-Numerai-white)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 👨‍💼 Executive Summary
**Author:** Dr. Brian Penrod, DBA  
**Codename:** KINETIC ZERO  
**Objective:** Generate rank-ordered predictive signals for the Numerai Hedge Fund Tournament using a "Twin-Engine" Ensemble architecture reinforced by OLS Orthogonalization.  
**Key Differentiator:** Strict adherence to **Chronological Regime Separation** (No Look-Ahead Bias) and **Phase 4 Feature Neutralization**.

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
    Features -->|Time-Series Split| Train[Training Set]:::raw
    
    subgraph ENGINES ["THE TWIN ENGINES"]
        Train --> LGBM[Engine 1: LightGBM]:::model
        Train --> XGB[Engine 2: XGBoost]:::model
    end

    LGBM -->|0.5| Ensemble((Weighted Signal)):::output
    XGB -->|0.5| Ensemble
    
    Ensemble -->|OLS Orthogonalization| Neutral[Kinetic Neutralization]:::logic
    Neutral -->|Rank & Upload| API[Numerai API]:::output
