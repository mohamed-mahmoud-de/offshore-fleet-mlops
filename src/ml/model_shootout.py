"""
Focused model shootout — RandomForest vs the gradient-boosting heavyweights.

Unlike the LazyPredict screen, every model here gets proper IMBALANCE HANDLING
(class weights / scale_pos_weight) and is judged on the metrics we actually care
about on rare-event data: **PR-AUC** (primary) and **failure recall**, plus
ROC-AUC / precision / F1 / accuracy for context.

Run from the project root:
    python -m src.ml.model_shootout
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.db import get_engine

warnings.filterwarnings("ignore")
SEED = 42


def main() -> None:
    df = pd.read_sql("SELECT * FROM ai4i_features", get_engine())
    y = df["machine_failure"]
    X = pd.get_dummies(df.drop(columns=["machine_failure"]), columns=["type"]).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # ratio of negatives to positives — used to up-weight the rare class in XGBoost
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),  # via sample_weight
        "XGBoost": XGBClassifier(
            n_estimators=300, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(
            n_estimators=300, class_weight="balanced", random_state=SEED,
            n_jobs=-1, verbose=-1),
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000),
    }

    rows = []
    for name, model in models.items():
        if name == "GradientBoosting":  # no class_weight param -> weight the samples
            model.fit(X_train, y_train,
                      sample_weight=compute_sample_weight("balanced", y_train))
        else:
            model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append(dict(
            model=name,
            pr_auc=average_precision_score(y_test, proba),      # PRIMARY
            recall=recall_score(y_test, pred, pos_label=1),
            precision=precision_score(y_test, pred, pos_label=1),
            f1=f1_score(y_test, pred, pos_label=1),
            roc_auc=roc_auc_score(y_test, proba),
            accuracy=accuracy_score(y_test, pred),
        ))

    board = (pd.DataFrame(rows)
             .set_index("model")
             .sort_values("pr_auc", ascending=False)
             .round(3))
    print("\n=== shootout — ranked by PR-AUC (higher = better) ===\n")
    print(board.to_string())
    print(f"\nwinner by PR-AUC: {board.index[0]}  (PR-AUC {board.iloc[0]['pr_auc']}, "
          f"recall {board.iloc[0]['recall']})")


if __name__ == "__main__":
    main()
