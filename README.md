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
graph TD
  A["config.yaml (runtime + model specs)"] --> B["numerai_multi_model_commander.py"]
  C["Env vars (NUMERAI_PUBLIC_ID / SECRET_KEY)"] --> B
  B --> D{"RISK_MODE"}
  D -->|DRYRUN| E["Download + Train + Predict + Write CSVs"]
  D -->|PRODUCTION| F["Wait round-open + Upload per model slot"]
  E --> G["submissions/submission_<MODEL>.csv"]
  F --> G
  G --> H["Cross-model correlation + ops checks"]
```
© 2026 Dr. Brian Penrod. All Rights Reserved.
