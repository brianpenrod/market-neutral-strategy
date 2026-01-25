# Kinetic Zero — System Map (Multi-Model Commander)

```mermaid
graph TD
  classDef raw fill:#2d2d2d,stroke:#555,stroke-width:2px,color:#fff;
  classDef process fill:#0D47A1,stroke:#000,stroke-width:2px,color:#fff;
  classDef model fill:#1B5E20,stroke:#00cc96,stroke-width:2px,color:#fff;
  classDef logic fill:#b71c1c,stroke:#ef553b,stroke-width:2px,color:#fff;
  classDef output fill:#4a148c,stroke:#aa00ff,stroke-width:2px,color:#fff;

  subgraph CONTROL["CONTROL PLANE: CONFIG + SAFETY"]
    cfg[config.yaml<br/>runtime + model_specs]:::raw --> spec[MODEL_SPECS<br/>3 slots]:::process
    env[Env Vars<br/>RISK_MODE + NUMERAI_KEYS]:::raw --> gate{RISK_MODE?}:::logic
    gate -->|DRYRUN| dry[DRYRUN: No wait, No upload]:::logic
    gate -->|PRODUCTION| prod[PROD: Wait for round-open + upload]:::logic
  end

 subgraph INGESTION["PHASE 1: INGESTION (NumerAPI + Polars)"]
  api[NumerAPI]:::process --> dl[Download v5.1<br/>features.json, train.parquet, live.parquet]:::process
  dl --> polars[Polars scan_parquet<br/>(lazy → collect)]:::process
  polars --> train[(train_df)]:::raw
  polars --> live[(live_df)]:::raw
  train --> clean[Clean + Fill NaN/Inf]:::process
  live --> clean2[Clean + Fill NaN/Inf]:::process
  clean --> era[Parse era to int]:::process
end


  subgraph VALIDATION["PHASE 2: CHRONOLOGICAL VALIDATION (NO LOOKAHEAD)"]
    era --> split{Era Cutoff<br/>(~80/20)}:::logic
    split --> tr[(train_split)]:::raw
    split --> va[(val_split)]:::raw
    note1[Chronological Firewall<br/>No shuffling]:::logic
    split -.-> note1
  end

  subgraph MULTI["PHASE 3: MULTI-SLOT EXECUTION LOOP (per ModelSpec)"]
    spec --> loop[For each ModelSpec:<br/>KZ_CORE_N00 / KZ_BAL_N50 / KZ_DEF_N75_DM]:::process

    loop --> lgbm[Engine A: LightGBM]:::model
    loop --> xgb[Engine B: XGBoost]:::model
    tr --> lgbm
    tr --> xgb

    lgbm --> pA[Pred_A (val)]:::process
    xgb --> pB[Pred_B (val)]:::process
    pA --> ensV((Weighted Ensemble)):::process
    pB --> ensV
    ensV --> rankV[Per-era Rank]:::process
    va --> rankV
    rankV --> corr[CORR proxy<br/>per-era corr -> sharpe_like]:::logic

    ensV --> retrain[Retrain on FULL train_df]:::process
    retrain --> lgbmF[LightGBM full]:::model
    retrain --> xgbF[XGBoost full]:::model
    clean --> lgbmF
    clean --> xgbF

    lgbmF --> pAL[Pred_A (live)]:::process
    xgbF --> pBL[Pred_B (live)]:::process
    pAL --> ensL((Weighted Ensemble)):::process
    pBL --> ensL

    ensL --> neutL{Submit neutralized?}:::logic
    neutL -->|Yes| neutLive[Neutralize to features<br/>(ridge residualization)]:::logic
    neutL -->|No| rawLive[Raw live signal]:::process

    meta[(meta_model.parquet<br/>(optional))]:::raw --> demeta{De-meta enabled<br/>& meta available?}:::logic
    neutLive --> demeta
    rawLive --> demeta
    demeta -->|Yes| orth[Rank + Gaussianize<br/>Orthogonalize to meta]:::logic
    demeta -->|No| noorth[Skip de-meta]:::process

    orth --> rankL[Final rank(pct)]:::process
    noorth --> rankL

    rankL --> checks[Safety checks:<br/>NaN fill + jitter if flat]:::logic
    checks --> csv[Write submission_<model>.csv]:::output
  end

  subgraph OPS["PHASE 4: UPLOAD + PORTFOLIO OVERSIGHT"]
    prod --> slots[Resolve model slots<br/>get_models() + case-insensitive match]:::logic
    csv --> gate2{Upload allowed?}:::logic
    gate2 -->|DRYRUN| skip[Skip upload]:::logic
    gate2 -->|PRODUCTION| up[upload_predictions()]:::output
    slots --> up
    up --> status[Numerai UI: awaiting -> submitted]:::output

    csv --> corrM[Cross-model correlation<br/>(live preds)]:::logic
    corrM --> warn{Any pair > 0.985?}:::logic
    warn -->|Yes| dup[Flag duplication<br/>recommend 1 lever change]:::logic
    warn -->|No| ok[Diversification OK]:::process
  end

  dry --> INGESTION
  prod --> INGESTION
  corr --> OPS
