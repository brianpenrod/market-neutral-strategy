# Market Neutral Strategy: Causal Factor-Aware Alpha Engine
### Codename: "Operation Overwatch"

> **Executive Summary:** A high-performance quantitative trading engine designed for the Numerai Hedge Fund Tournament. This system leverages **Causal Discovery (LiNGAM)** to identify idiosyncratic alpha and utilizes a proprietary **Risk Molding** protocol to orthogonalize signals against latent market factors (Beta), ensuring pure, uncorrelated returns.

---

## 🏗 System Architecture: The Operational Map

The architecture follows a strict Event-Driven Pipeline designed for **v5.2 "Faith2"** data integrity. It separates the **Alpha Generation** (Signal) from the **Risk Management** (Exposure), allowing for dynamic portfolio construction from a single intelligence source.

```mermaid
graph TD
  %% ===================== STYLES =====================
  classDef raw fill:#2d2d2d,stroke:#555,stroke-width:2px,color:#fff;
  classDef process fill:#0D47A1,stroke:#000,stroke-width:2px,color:#fff;
  classDef model fill:#1B5E20,stroke:#00cc96,stroke-width:2px,color:#fff;
  classDef logic fill:#b71c1c,stroke:#ef553b,stroke-width:2px,color:#fff;
  classDef output fill:#4a148c,stroke:#aa00ff,stroke-width:2px,color:#fff;

  %% ===================== CONTROL PLANE =====================
  subgraph CONTROL["CONTROL PLANE"]
    env["Env vars: PUBLIC_ID + SECRET_KEY"]:::raw --> gate{"RISK_MODE"}:::logic
    target["Target: target (v5.2 Standard)"]:::raw --> gate
    gate --> dry["DRYRUN: No Upload"]:::logic
    gate --> prod["PRODUCTION: Live Upload"]:::logic
  end

  %% ===================== LOGISTICS =====================
  subgraph LOGISTICS["PHASE 1: INTELLIGENCE LOGISTICS (v5.2)"]
    api["NumerAPI"]:::process --> patch{"Live Intel Patch"}:::logic
    patch -->|Force Delete| fresh["Download Fresh live.parquet"]:::process
    patch -->|Check Exists| train["Load train.parquet"]:::process
    
    fresh --> castL["Cast Int8 -> Float32"]:::process
    train --> castT["Cast Int8 -> Float32"]:::process
    
    castL --> uniCheck{"Universe Safety Check (>5500 rows?)"}:::logic
    uniCheck -->|Pass| readyL["Live Matrix (Float32)"]:::raw
    castT --> readyT["Train Matrix (Float32)"]:::raw
  end

  %% ===================== CAUSAL DISCOVERY =====================
  subgraph CAUSAL["PHASE 2: CAUSAL DISCOVERY (LiNGAM)"]
    readyT --> lingam["DirectLiNGAM: Learn Structure"]:::model
    lingam --> pairs["Identify Causal Pairs (A -> B)"]:::output
    
    pairs --> augT["Augment Train (Interaction Terms)"]:::process
    pairs --> augL["Augment Live (Interaction Terms)"]:::process
    
    readyT --> augT
    readyL --> augL
  end

  %% ===================== THE CORE FORGE =====================
  subgraph FORGE["PHASE 3: ALPHA GENERATION (Train Once)"]
    augT --> lgbm["Engine A: LightGBM (Gradient Boosting)"]:::model
    augT --> xgb["Engine B: XGBoost (Histogram)"]:::model
    
    lgbm --> preds["Generating Raw Alpha..."]:::process
    xgb --> preds
    augL --> preds
    
    preds --> rawSignal["Raw Signal (Ensemble Average)"]:::output
  end

  %% ===================== RISK MOLDING =====================
  subgraph RISK["PHASE 4: RISK MOLDING (Factor Neutralization)"]
    readyT --> pca["PCA: Extract 50 Latent Factors (Beta)"]:::model
    pca --> riskVec["Project Live Data -> Risk Vectors"]:::process
    readyL --> riskVec
    
    rawSignal --> mold{"ORTHOGONALIZATION MATRIX"}:::logic
    riskVec --> mold
    
    mold -->|0% Neut| n00["KZ_CORE_N00 (Pure Alpha)"]:::output
    mold -->|50% Neut| n50["KZ_BAL_N50 (Hybrid)"]:::output
    mold -->|75% Neut| n75["KZ_DEF_N75 (Bunker)"]:::output
  end

  %% ===================== DEPLOYMENT =====================
  subgraph OPS["PHASE 5: DEPLOYMENT"]
    n00 & n50 & n75 --> rank["Rank -> Uniform Dist"]:::process
    rank --> csv["Write CSVs"]:::output
    
    prod --> resolve["Resolve Model IDs"]:::logic
    resolve --> upload["API Upload"]:::process
    csv --> upload
  end
