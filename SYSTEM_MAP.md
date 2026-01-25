# Kinetic Zero — System Map (Multi-Model Numerai Commander)

This diagram shows the end-to-end control plane, data ingestion, per-model training loop, risk gates (DRYRUN vs PROD), and upload safeguards for the three-slot deployment.

```mermaid
graph TD
  %% ===================== STYLES =====================
  classDef raw fill:#2d2d2d,stroke:#555,stroke-width:2px,color:#fff;
  classDef process fill:#0D47A1,stroke:#000,stroke-width:2px,color:#fff;
  classDef model fill:#1B5E20,stroke:#00cc96,stroke-width:2px,color:#fff;
  classDef logic fill:#b71c1c,stroke:#ef553b,stroke-width:2px,color:#fff;
  classDef output fill:#4a148c,stroke:#aa00ff,stroke-width:2px,color:#fff;

  %% ===================== CONTROL PLANE =====================
  subgraph CONTROL["CONTROL PLANE (CONFIG + SAFETY)"]
    cfg["config.yaml: runtime + model specs"]:::raw --> specs["MODEL SPECS (3 slots)"]:::process
    env["Env vars: NUMERAI_PUBLIC_ID + NUMERAI_SECRET_KEY"]:::raw --> gate{"RISK_MODE"}:::logic
    gate --> dry["DRYRUN: no wait, no upload"]:::logic
    gate --> prod["PRODUCTION: wait + upload"]:::logic
  end

  %% ===================== INGESTION =====================
  subgraph INGESTION["PHASE 1: INGESTION (NumerAPI + Polars)"]
    api["NumerAPI"]:::process --> dl["Download: features.json, train.parquet, live.parquet"]:::process
    dl --> metaOpt["(Optional) meta_model.parquet"]:::raw
    dl --> polars["Polars scan_parquet -> collect"]:::process
    polars --> train["train_df"]:::raw
    polars --> live["live_df"]:::raw
    train --> cleanT["Clean train (fill NaN/Inf)"]:::process
    live --> cleanL["Clean live (fill NaN/Inf)"]:::process
    cleanT --> era["Parse era -> int"]:::process
  end

  %% ===================== VALIDATION =====================
  subgraph VALIDATION["PHASE 2: VALIDATION (NO LOOKAHEAD)"]
    era --> split{"Chronological split (80/20 by era)"}:::logic
    split --> tr["train_split"]:::raw
    split --> va["val_split"]:::raw
    split -.-> firewall["Chronological firewall"]:::logic
  end

  %% ===================== PER-MODEL LOOP =====================
  subgraph MULTI["PHASE 3: PER-MODEL LOOP (CORE / BAL / DEF)"]
    specs --> loop["For each ModelSpec"]:::process

    tr --> lgbm["Engine A: LightGBM"]:::model
    tr --> xgb["Engine B: XGBoost"]:::model

    lgbm --> valEns["Ensemble preds (val)"]:::process
    xgb --> valEns
    valEns --> rankVal["Per-era rank (val)"]:::process
    rankVal --> corrProxy["Corr proxy (val)"]:::logic

    loop --> retrain["Retrain on full train_df"]:::process
    retrain --> lgbmF["LightGBM full"]:::model
    retrain --> xgbF["XGBoost full"]:::model

    lgbmF --> liveEns["Ensemble preds (live)"]:::process
    xgbF --> liveEns

    liveEns --> neut{"Neutralize ratio > 0 ?"}:::logic
    neut --> neutLive["Neutralize to features (ridge)"]:::logic
    neut --> rawLive["Raw live signal"]:::process

    metaOpt --> demeta{"De-meta enabled + available?"}:::logic
    neutLive --> demeta
    rawLive --> demeta
    demeta --> orth["Orthogonalize to meta"]:::logic
    demeta --> noOrth["Skip de-meta"]:::process

    orth --> final["Final rank pct + safety checks (NaN/flat/jitter)"]:::logic
    noOrth --> final
    final --> csv["Write submission CSV per model"]:::output
  end

  %% ===================== UPLOAD + OPS =====================
  subgraph OPS["PHASE 4: UPLOAD + OPS CHECKS"]
    prod --> slots["Resolve slots via get_models (case-insensitive)"]:::logic
    csv --> allow{"Upload allowed?"}:::logic
    allow --> skip["DRYRUN: skip upload"]:::logic
    allow --> up["upload_predictions"]:::output
    slots --> up

    csv --> corrM["Cross-model correlation (live preds)"]:::logic
    corrM --> dup{"Any pair > 0.985?"}:::logic
    dup --> fix["Flag duplicates; change ONE lever next round"]:::logic
    dup --> ok["Diversification OK"]:::process
  end

  %% ===================== FLOW CONTROL =====================
  dry --> api
  prod --> api
```
## Legend
- **Raw**: datasets, config, environment variables  
- **Process**: ingestion, transforms, orchestration steps  
- **Model**: LightGBM / XGBoost training + inference components  
- **Logic**: safety gates, splits, neutralization / de-meta decisions  
- **Output**: CSV artifacts + Numerai API uploads
