"""
Milestone 5 — train + evaluate the equipment-failure predictor.

Pipeline of good practice (each step is here to avoid a specific mistake):
  1. Read ai4i_features from Postgres.
  2. **Stratified** train/test split — keeps the same ~3.4% failure rate in both parts.
  3. A scikit-learn **Pipeline** so all preprocessing is fit on TRAIN only, then applied
     to TEST -> no data leakage.
  4. **RandomForest(class_weight='balanced')** — tree model (robust to outliers, no scaling
     needed) that up-weights the rare failures instead of ignoring them.
  5. Judge with **precision / recall / PR-AUC**, and compare against a naive baseline that
     always predicts "no failure" — to prove why accuracy is the wrong metric here.

Run from the project root:
    python -m src.ml.train_model
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless: save figures, don't open windows
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

from src.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
TARGET = "machine_failure"


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_sql("SELECT * FROM ai4i_features", get_engine())
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    categorical = ["type"]
    numeric = [c for c in X.columns if c not in categorical]

    # 2) stratified split (preserve the rare-failure ratio in both halves)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"train: {len(X_train)} rows ({y_train.mean()*100:.2f}% failures)")
    print(f"test : {len(X_test)} rows ({y_test.mean()*100:.2f}% failures)")

    # 3) leakage-safe pipeline: one-hot the category, pass numerics through
    preprocess = ColumnTransformer(
        [("type", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough")

    # 4) tree model that up-weights the rare class
    model = Pipeline([
        ("preprocess", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 5a) naive baseline: always predict the majority class ("no failure")
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 60)
    print("NAIVE BASELINE (always predicts 'no failure')")
    print("=" * 60)
    print(f"accuracy: {(dummy_pred == y_test).mean()*100:.2f}%   "
          f"<-- looks great, but it catches 0 of the failures")

    print("\n" + "=" * 60)
    print("OUR MODEL (RandomForest, balanced)")
    print("=" * 60)
    print(f"accuracy: {(y_pred == y_test).mean()*100:.2f}%")
    print("\nclassification report (class 1 = failure is what matters):")
    print(classification_report(y_test, y_pred, digits=3,
                                target_names=["no failure", "failure"]))
    print(f"PR-AUC (average precision): {average_precision_score(y_test, y_proba):.3f}")
    print(f"ROC-AUC                   : {roc_auc_score(y_test, y_proba):.3f}")
    print("\nconfusion matrix [rows=actual, cols=predicted]:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 pred_no   pred_fail")
    print(f"  actual_no      {cm[0,0]:>7}   {cm[0,1]:>9}")
    print(f"  actual_fail    {cm[1,0]:>7}   {cm[1,1]:>9}   <- caught {cm[1,1]} of {cm[1].sum()} real failures")

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax[0], colorbar=False,
        display_labels=["no failure", "failure"])
    ax[0].set_title("Confusion matrix — our model")
    ap = average_precision_score(y_test, y_proba)
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1])
    ax[1].set_title(f"Precision-Recall curve (PR-AUC = {ap:.3f})")
    plt.tight_layout()
    plt.savefig(FIG / "ml_evaluation.png", bbox_inches="tight")

    # ---- feature importances ---------------------------------------------
    feat_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = pd.Series(
        model.named_steps["rf"].feature_importances_, index=feat_names
    ).sort_values(ascending=False)
    print("\ntop feature importances:")
    print(importances.head(8).round(3).to_string())
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    importances.head(10)[::-1].plot.barh(ax=ax2, color="#4c72b0")
    ax2.set_title("What the model relies on (feature importance)")
    plt.tight_layout()
    plt.savefig(FIG / "ml_feature_importance.png", bbox_inches="tight")

    joblib.dump(model, MODELS / "failure_model.joblib")
    print(f"\nsaved model -> {MODELS / 'failure_model.joblib'}")
    print("Milestone 5 complete.")


if __name__ == "__main__":
    main()
