
### Architectural Overview

The pipeline follows a six-phase architecture that separates signal generation from risk management. Raw v5.2 data flows through a Purged Walk-Forward CV split into three parallel alpha engines — a DAE-initialized neural ensemble, a multi-target gradient boosting ensemble, and a binary tail-event classifier. Their outputs converge at the Ensemble Blender, which constructs two distinct signal profiles (Default and Binary). The Risk Engine then orthogonalizes each signal against 50 PCA-derived market factors at varying neutralization ratios, producing three deployment-ready model slots with differentiated return profiles.

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
    target["Primary: target (v5.2 Standard)"]:::raw --> gate
    mt["Multi-Target: ender / cyrusd / teager2b / victor"]:::raw --> gate
    gate --> dry["DRYRUN: No Upload"]:::logic
    gate --> prod["PRODUCTION: Live Upload"]:::logic
  end

  %% ===================== LOGISTICS =====================
  subgraph LOGISTICS["PHASE 1: INTELLIGENCE LOGISTICS (v5.2 Sunshine)"]
    api["NumerAPI"]:::process --> patch{"Live Intel Patch"}:::logic
    patch -->|Force Delete| fresh["Download Fresh live.parquet"]:::process
    patch -->|Saturday Sync| train["Download train.parquet + features.json"]:::process

    train --> fset["Feature Selection: Medium Set (780 dims)"]:::process
    train --> tload["Load All Target Columns"]:::process

    fresh --> castL["Cast Int8 → Float32 / Fill Nulls"]:::process
    fset --> castT["Cast Int8 → Float32 / Fill Nulls"]:::process
    tload --> castT

    castL --> uniCheck{"Universe Safety Check (>5500 rows?)"}:::logic
    uniCheck -->|Pass| readyL["Live Matrix (Float32)"]:::raw
    castT --> readyT["Train Matrix (Float32)"]:::raw
  end

  %% ===================== PURGED CV =====================
  subgraph CV["PHASE 2: PURGED WALK-FORWARD CV"]
    readyT --> purge["5-Fold Purged Walk-Forward Split"]:::process
    purge --> gap["Purge Gap: 4 Eras"]:::logic
    gap --> trainSet["Train Partition"]:::raw
    gap --> valSet["Validation Partition"]:::raw
  end

  %% ===================== NEURAL TACTICIAN =====================
  subgraph NEURAL["PHASE 3A: NEURAL TACTICIAN (DAE-MLP)"]
    trainSet --> dae["Denoising Autoencoder (0.1 Swap Noise)"]:::model
    dae --> bottleneck["64-Dim Latent Bottleneck"]:::process
    bottleneck --> encoder["2-Layer Encoder (1024 → 64) / SiLU + BN"]:::process

    encoder --> seed1["ResMLP Seed 1"]:::model
    encoder --> seed2["ResMLP Seed 2"]:::model
    encoder --> seed3["ResMLP Seed 3"]:::model
    encoder --> seed4["ResMLP Seed 4"]:::model
    encoder --> seed5["ResMLP Seed 5"]:::model

    seed1 & seed2 & seed3 & seed4 & seed5 --> nnAvg["5-Seed Average (Rank Normalized)"]:::output
  end

  %% ===================== MULTI-TARGET TREES =====================
  subgraph TREES["PHASE 3B: MULTI-TARGET TREE ENSEMBLE"]
    trainSet --> lgbm["LightGBM (10K trees, lr=0.005, GPU)"]:::model
    trainSet --> xgb["XGBoost (10K trees, lr=0.005, CUDA)"]:::model

    lgbm --> lgbT1["ender_20 (40%)"]:::process
    lgbm --> lgbT2["cyrusd_20 (25%)"]:::process
    lgbm --> lgbT3["teager2b_20 (20%)"]:::process
    lgbm --> lgbT4["victor_20 (15%)"]:::process

    xgb --> xgbT1["ender_20 (40%)"]:::process
    xgb --> xgbT2["cyrusd_20 (25%)"]:::process
    xgb --> xgbT3["teager2b_20 (20%)"]:::process
    xgb --> xgbT4["victor_20 (15%)"]:::process

    lgbT1 & lgbT2 & lgbT3 & lgbT4 --> lgbBlend["LGB Weighted Rank Avg"]:::output
    xgbT1 & xgbT2 & xgbT3 & xgbT4 --> xgbBlend["XGB Weighted Rank Avg"]:::output
  end

  %% ===================== BINARY ALPHA =====================
  subgraph BINARY["PHASE 3C: BINARY ALPHA (Raw Features)"]
    trainSet --> ridge["RidgeClassifier (α=10.0)"]:::model
    ridge --> threshold{"> 0.5 Threshold Detection"}:::logic
    threshold --> binSignal["Binary Decision Function (Rank Normalized)"]:::output
  end

  %% ===================== ENSEMBLE BLENDER =====================
  subgraph BLEND["PHASE 4: ENSEMBLE BLENDER"]
    nnAvg --> defBlend{"DEFAULT BLEND"}:::logic
    lgbBlend --> defBlend
    xgbBlend --> defBlend
    binSignal -->|5% Tilt| defBlend

    nnAvg --> binBlend{"BINARY BLEND"}:::logic
    lgbBlend --> binBlend
    xgbBlend --> binBlend
    binSignal -->|40% Weight| binBlend

    defBlend --> rawDefault["Raw Default Signal"]:::output
    binBlend --> rawBinary["Raw Binary Signal"]:::output
  end

  %% ===================== RISK MOLDING =====================
  subgraph RISK["PHASE 5: RISK ENGINE (Factor Neutralization)"]
    readyT --> pca["PCA: Extract 50 Latent Factors (Beta)"]:::model
    pca --> riskVec["Project Live Data → Risk Vectors"]:::process
    readyL --> riskVec

    rawDefault --> mold{"ORTHOGONALIZATION MATRIX (Ridge Residuals)"}:::logic
    rawBinary --> mold
    riskVec --> mold

    mold -->|"0% Neut / Default"| n00["KZ_CORE_N00 (Raw Return Alpha)"]:::output
    mold -->|"90% Neut / Default"| n90["KZ_DEF_N90 (High Sharpe / Low Drawdown)"]:::output
    mold -->|"20% Neut / Binary"| n20["KZ_BINARY_N20 (Tail-Event Capture)"]:::output
  end

  %% ===================== DEPLOYMENT =====================
  subgraph OPS["PHASE 6: DEPLOYMENT"]
    n00 & n90 & n20 --> rank["Rank → Uniform Distribution"]:::process
    rank --> csv["Write Submission CSVs"]:::output

    prod --> resolve["Resolve Model Slot IDs"]:::logic
    resolve --> upload["NumerAPI Upload"]:::process
    csv --> upload
  end

  %% ===================== PERSISTENCE =====================
  subgraph CACHE["WEIGHT CACHE (Google Drive)"]
    dae -.->|Save| weights["Weights_v92/"]:::raw
    lgbm -.->|Save| weights
    xgb -.->|Save| weights
    ridge -.->|Save| weights
    pca -.->|Save| weights
    weights -.->|"Load (Tue-Fri)"| quickInfer["Daily Inference (~2 min)"]:::process
  end
```
Author Profile
Brian Penrod, DBA Retired US Army Special Forces CSM | Doctor of Business Administration (Finance)

"I combine military strategic planning with advanced quantitative finance to build systems that prioritize risk management, data integrity, and tactical execution."
