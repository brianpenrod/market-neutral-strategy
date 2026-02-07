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
🧠 Methodology: The "Sniper" Doctrine
Traditional quantitative models often suffer from "over-neutralization"—stripping out valid signals in an attempt to reduce volatility (the "Shotgun" approach). This system employs a "Sniper" doctrine, distinguishing between Market Beta (systemic risk) and Idiosyncratic Alpha (true skill).

1. Causal Discovery (Signal)
Rather than relying solely on linear correlations, this engine utilizes LiNGAM (Linear Non-Gaussian Acyclic Model) to detect directional causality between features.

The Problem: Correlation does not imply causation. A stock might move because the market moved, not because of the feature.

The Solution: The engine constructs a Directed Acyclic Graph (DAG) of the feature set to find non-linear interaction pairs (e.g., Feature_A → causes → Feature_B). These are engineered into interaction terms to capture the "hidden physics" of the market.

2. Ensemble Stability
To minimize variance and prevent overfitting, the raw signal is generated via a 50/50 weighted ensemble of:

LightGBM: Optimized for leaf-wise growth and speed.

XGBoost: Utilizing histogram-based learning for robust handling of the v5.2 quantized data.

🛡️ Risk Molding: Active Exposure Management
"Risk Molding" is the proprietary process of sculpting the return distribution to fit specific volatility profiles. Instead of training three separate models, the system trains one high-conviction Alpha model and then mathematically projects it onto different risk planes.

The system extracts the top 50 latent market factors (representing Sector, Momentum, Volatility, and Value) via PCA and orthogonalizes the alpha signal against them using Ridge Regression residuals.

Profile	Ticker	Neutralization	Role in Portfolio
Aggressive (Core)	KZ_CORE_N00	0%	Pure Alpha. Captures maximum upside during trending markets. High volatility, high potential Sharpe. Captures the "Look Above/Below" liquidity sweeps.
Balanced (Hybrid)	KZ_BAL_N50	50%	Sharpe Optimizer. Removes the "loudest" market noise while retaining directional signal.
Defensive (Bunker)	KZ_DEF_N75	75%	Market Neutral. Strictly orthogonal to market moves. Generates returns solely from stock-specific selection (True Contribution).
Validation: Live production analysis confirms a 0.69 correlation between the Aggressive and Defensive profiles, proving the system successfully decouples Alpha from Beta.

⚡ Technical Specifications
Language: Python 3.12

Data Structure: Polars (Rust-based DataFrame) for high-velocity ingest of int8 quantized Parquet files.

Compute: GPU-Accelerated training (CUDA) for rapid retraining cycles.

Robustness: Implements a "Live Intel Patch" that forces fresh data acquisition per epoch to prevent look-ahead bias or stale universe errors.

📈 Performance Objectives
Universe Size: ~7,000 Global Equities.

Consensus Conviction: In recent validation sets, the system identified a "Strong Buy" consensus (>0.8 probability) on 11.5% of the universe, demonstrating high selectivity.

Distribution: Perfect uniform distribution of ranked predictions (Mean: 0.50, Std: 0.29), ensuring maximum information entropy in the submission.

Author Profile
Brian Penrod, DBA Retired US Army Special Forces CSM | Doctor of Business Administration (Finance)

"I combine military strategic planning with advanced quantitative finance to build systems that prioritize risk management, data integrity, and tactical execution."
