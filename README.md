# Offshore Fleet MLOps — Performance Scorecard & Predictive Maintenance

An end-to-end **MLOps portfolio project** for offshore marine / oil-support fleet
operations, built the way it would run in production: **Data Engineering → Machine
Learning → MLOps**, one milestone at a time.

> **Portfolio / proof-of-concept.** Built on **synthetic operational data** over a
> realistic offshore support fleet and the **public AI4I 2020** dataset. It uses **no
> confidential or proprietary data** and is not affiliated with any company — it's
> inspired by the offshore marine-services domain.

## What it does

Two independent tracks that mirror the two things fleet operations actually need:

1. **Performance scorecard (grade the past)** — a transparent, rules-based system that
   grades every vessel/crew. It produces **two grades**:
   - a **crew score** (on-time completion, safety, downtime — what a crew controls), and
   - a **vessel commercial score** (utilization-heavy, with an idle-override rule),
   - plus a **confidence flag** so grades based on too few missions aren't over-trusted.
2. **Predictive maintenance (predict the future)** — a model that predicts equipment
   failure from sensor readings.

## Results

- **Failure model: LightGBM** on imbalanced data (~3.4% failures) — **PR-AUC 0.879**,
  **failure recall 0.82** (catches 56 of 68 failures), precision 0.85 (held-out test set;
  accuracy is deliberately *not* the headline metric for such imbalanced data).
- **Model selection, not assumption:** screened ~25 models with LazyPredict, then ran an
  imbalance-aware shootout (RandomForest vs XGBoost vs LightGBM vs GradientBoosting) ranked
  on PR-AUC — **LightGBM won**, beating the obvious RandomForest choice.
- **Feature engineering paid off:** an engineered physics feature (mechanical power) and a
  temperature-difference feature both rank among the model's top signals.
- **Tracked with MLflow:** every run logs params/metrics/figures, and the best model is
  registered in the MLflow Model Registry.

## Architecture (batch pipeline)

```
CSV / source data
      │  ingestion (+ data-quality checks)
      ▼
PostgreSQL  (raw → features → scorecard)         ← Data Engineering
      │  train / evaluate (leakage-safe pipeline)
      ▼
ML model (predict failure)  +  MLflow tracking   ← Machine Learning / MLOps
      │  serve
      ▼
Dashboard + automated report  •  monitoring       ← MLOps
```

## Tech stack

Python · PostgreSQL (Docker) · SQLAlchemy · pandas · scikit-learn · LightGBM · XGBoost ·
MLflow · Streamlit · Evidently · Docker · GitHub Actions

## Data sources

| Data | Role | Notes |
|------|------|-------|
| Synthetic fleet dataset | scorecard + DE pipeline | Synthetic operational metrics over a public offshore-fleet roster. No private data. |
| [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | ML model | UCI ML Repository, **CC BY 4.0**. |

## Project structure

```
config/         non-secret settings (weights, thresholds) -> config.yaml
data/raw|interim|processed   data layers (git-ignored)
notebooks/      exploratory analysis (EDA, with graphs)
src/
  db.py         single Postgres connection helper (reads .env)
  ingestion/    load raw data + data-quality checks   (M2)
  features/     cleaning + feature engineering          (M3)
  scorecard/    rules-based crew + vessel scorecards     (M4)
  ml/           model benchmark + shootout, train/evaluate (M5-M6)
  serving/      dashboard + API                           (M8)
  monitoring/   data-quality & drift checks              (M9)
pipelines/      orchestration (scheduled runs)           (M7)
docker/         docker-compose for local Postgres
```

## Milestones

- [x] **M0** Project setup & structure
- [x] **M1** EDA (distributions, outliers, correlations, feature-vs-target)
- [x] **M2** Ingestion into Postgres + data-quality checks
- [x] **M3** Cleaning + feature engineering
- [x] **M4** Rules-based scorecard (crew + vessel grades)
- [x] **M5** Failure-prediction model (LightGBM, selected via benchmark + shootout)
- [x] **M6** MLflow experiment tracking + model registry
- [ ] **M7** Orchestration (scheduled pipeline)
- [ ] **M8** Serving: dashboard + automated report
- [ ] **M9** Monitoring: data-quality + drift
- [ ] **M10** Docker packaging + CI (GitHub Actions)

## Getting started

```bash
# 1. start a local Postgres (needs Docker)
docker compose -f docker/docker-compose.yml up -d

# 2. python environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 3. secrets — copy the template (defaults already match docker-compose)
copy .env.example .env            # cp on macOS/Linux

# 4. run the pipeline, in order
python -m src.ingestion.download_ai4i     # fetch the public dataset
python -m src.ingestion.load_raw          # load raw -> Postgres + checks
python -m src.features.build_features      # clean + engineer features
python -m src.scorecard.build_scorecard    # build the scorecards
python -m src.ml.train_model               # train + evaluate the model
```

## License / attribution

- AI4I 2020 Predictive Maintenance Dataset — © its authors, **CC BY 4.0** (UCI ML Repository).
- Synthetic operational data and all code in this repository are original work for portfolio purposes.
