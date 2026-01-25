# PROJECT: KINETIC ZERO (v2.3)
**Production-Grade Quantitative Deployment for Numerai (Multi-Model Ops + Research Loop)**

![Status](https://img.shields.io/badge/Status-Production-success)
![Platform](https://img.shields.io/badge/Platform-Numerai-white)
![Language](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Executive Summary
**Author:** Dr. Brian Penrod, DBA  
**Codename:** **KINETIC ZERO**  
**Mission:** Generate rank-ordered predictive signals for the Numerai Tournament via a controlled, reproducible pipeline.  
**Architecture:** Multi-slot deployment using a **Twin-Engine ensemble** (LightGBM + XGBoost) with optional **feature neutralization** and optional **de-meta orthogonalization**.  
**Key Discipline:** **Chronological regime separation** (no look-ahead bias) + strict **ops safety gates** (DRYRUN by default).

---

## System Architecture
For the detailed operational diagram, see **[SYSTEM_MAP.md](SYSTEM_MAP.md)**.

### Executive Map (one-screen view)
```mermaid
%%{init: {"theme":"base","themeVariables":{
  "primaryColor":"#E8F0FE",
  "primaryTextColor":"#0B1020",
  "primaryBorderColor":"#1E3A8A",
  "lineColor":"#475569",
  "secondaryColor":"#ECFDF5",
  "tertiaryColor":"#FFF7ED",
  "fontFamily":"ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto"
}}}%%
flowchart TD

  %% --- NODE STYLES ---
  classDef config fill:#E0F2FE,stroke:#0369A1,stroke-width:2px,color:#0B1020;
  classDef secrets fill:#FEF3C7,stroke:#B45309,stroke-width:2px,color:#0B1020;
  classDef script fill:#EDE9FE,stroke:#6D28D9,stroke-width:2px,color:#0B1020;
  classDef gate fill:#FFE4E6,stroke:#BE123C,stroke-width:2px,color:#0B1020;
  classDef action fill:#DCFCE7,stroke:#15803D,stroke-width:2px,color:#0B1020;
  classDef output fill:#FCE7F3,stroke:#BE185D,stroke-width:2px,color:#0B1020;

  %% --- EXECUTIVE MAP ---
  cfg["config.yaml<br/>(runtime + model specs)"]:::config
  env["Env vars<br/>(NUMERAI_PUBLIC_ID / SECRET_KEY)"]:::secrets
  cmd["numerai_multi_model_commander.py"]:::script
  mode{"RISK_MODE"}:::gate

  dry["DRYRUN<br/>Download + Train + Predict<br/>Write CSVs"]:::action
  prod["PRODUCTION<br/>Wait round-open + Upload<br/>per model slot"]:::action

  out["submissions/<br/>submission_&lt;MODEL&gt;.csv"]:::output

  cfg --> cmd
  env --> cmd
  cmd --> mode
  mode -->|DRYRUN| dry
  mode -->|PRODUCTION| prod
  dry --> out
  prod --> out
```
© 2026 Dr. Brian Penrod. All Rights Reserved.
