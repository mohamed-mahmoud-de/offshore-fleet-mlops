"""
Champion-challenger model promotion (M10 extension).

Trains a fresh model (the *challenger*) and promotes it to *champion* ONLY if it
beats the current champion on PR-AUC — so retraining can never silently make the
served model worse. The champion is:
    models/failure_model.joblib     (the model the dashboard/serving loads)
    models/champion_metrics.json    (its recorded metrics)

Typical loop:  monitoring flags drift  ->  run this  ->  promote only if better.

Run from the project root:
    python -m src.ml.promote_model
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CHAMPION = MODELS / "failure_model.joblib"
CHAMPION_METRICS = MODELS / "champion_metrics.json"
TARGET = "machine_failure"


def train_challenger():
    """Train a fresh model and evaluate it on a held-out test split."""
    df = pd.read_sql("SELECT * FROM ai4i_features", get_engine())
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    pre = ColumnTransformer(
        [("type", OneHotEncoder(handle_unknown="ignore"), ["type"])],
        remainder="passthrough")
    model = Pipeline([
        ("preprocess", pre),
        ("clf", LGBMClassifier(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1, verbose=-1)),
    ])
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    metrics = {
        "pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "recall": round(float(recall_score(y_te, pred, pos_label=1)), 4),
    }
    return model, metrics


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    challenger, chal = train_challenger()
    champ = json.loads(CHAMPION_METRICS.read_text()) if CHAMPION_METRICS.exists() else None

    print(f"challenger:  PR-AUC {chal['pr_auc']}   recall {chal['recall']}")
    if champ is None:
        promote, reason = True, "no champion on record -> promote challenger"
    else:
        print(f"champion:    PR-AUC {champ['pr_auc']}   recall {champ.get('recall')}")
        promote = chal["pr_auc"] > champ["pr_auc"]      # strictly better -> avoid churn on noise
        reason = "challenger is better -> PROMOTE" if promote else "champion retained (challenger not better)"

    if promote:
        joblib.dump(challenger, CHAMPION)
        CHAMPION_METRICS.write_text(json.dumps(chal, indent=2))

    print(f"decision: {reason}")


if __name__ == "__main__":
    main()
