"""
Milestone 5 + 6 — train/evaluate the failure predictor, tracked with MLflow.

Model choice: **LightGBM**, selected after a LazyPredict screen (src/ml/benchmark_models.py)
and a proper imbalance-aware shootout (src/ml/model_shootout.py) where it beat
RandomForest, XGBoost, and the rest on PR-AUC.

M5 (modeling):
  - stratified train/test split (keeps the ~3.4% failure rate in both halves)
  - leakage-safe Pipeline (preprocessing fit on TRAIN only)
  - LightGBM(class_weight='balanced') for the rare class
  - judged with precision / recall / PR-AUC (never accuracy alone)

M6 (MLOps):
  - every run is logged to MLflow: parameters, metrics, figures, and the model
  - the model is registered in the MLflow Model Registry as a new version.

Run from the project root:
    python -m src.ml.train_model
View the runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://localhost:5000
"""
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
FIG = ROOT / "reports" / "figures"
TARGET = "machine_failure"

# --- hyper-parameters (logged to MLflow so runs are comparable) -------------
PARAMS = dict(model="LightGBM", n_estimators=300, class_weight="balanced",
              test_size=0.2, random_state=42)


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("failure-prediction")

    df = pd.read_sql("SELECT * FROM ai4i_features", get_engine())
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    categorical = ["type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PARAMS["test_size"], stratify=y,
        random_state=PARAMS["random_state"])

    preprocess = ColumnTransformer(
        [("type", OneHotEncoder(handle_unknown="ignore"), categorical)],
        remainder="passthrough")
    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", LGBMClassifier(
            n_estimators=PARAMS["n_estimators"], class_weight=PARAMS["class_weight"],
            random_state=PARAMS["random_state"], n_jobs=-1, verbose=-1)),
    ])

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        dummy_acc = accuracy_score(y_test, dummy.predict(X_test))

        metrics = dict(
            test_accuracy=accuracy_score(y_test, y_pred),
            failure_precision=precision_score(y_test, y_pred, pos_label=1),
            failure_recall=recall_score(y_test, y_pred, pos_label=1),
            failure_f1=f1_score(y_test, y_pred, pos_label=1),
            pr_auc=average_precision_score(y_test, y_proba),
            roc_auc=roc_auc_score(y_test, y_proba),
            baseline_accuracy=dummy_acc,
        )
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_metrics(metrics)
        mlflow.log_metric("failures_caught", int(cm[1, 1]))
        mlflow.log_metric("failures_total", int(cm[1].sum()))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax[0], colorbar=False,
            display_labels=["no failure", "failure"])
        ax[0].set_title("Confusion matrix — LightGBM")
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax[1])
        ax[1].set_title(f"Precision-Recall (PR-AUC = {metrics['pr_auc']:.3f})")
        plt.tight_layout(); plt.savefig(FIG / "ml_evaluation.png", bbox_inches="tight")

        importances = pd.Series(
            model.named_steps["clf"].feature_importances_,
            index=model.named_steps["preprocess"].get_feature_names_out()
        ).sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        importances.head(10)[::-1].plot.barh(ax=ax2, color="#4c72b0")
        ax2.set_title("Feature importance — LightGBM")
        plt.tight_layout(); plt.savefig(FIG / "ml_feature_importance.png", bbox_inches="tight")

        mlflow.log_artifact(FIG / "ml_evaluation.png")
        mlflow.log_artifact(FIG / "ml_feature_importance.png")

        mlflow.sklearn.log_model(model, name="model",
                                 serialization_format="cloudpickle",  # skops rejects LightGBM types
                                 registered_model_name="failure-predictor")
        joblib.dump(model, MODELS / "failure_model.joblib")

        print("=" * 60)
        print(f"MLflow run: {run.info.run_id}  (model: {PARAMS['model']})")
        print("=" * 60)
        print(f"baseline (always 'no failure') accuracy: {dummy_acc*100:.2f}%  (catches 0 failures)")
        print(f"model accuracy:  {metrics['test_accuracy']*100:.2f}%")
        print(f"failure recall:  {metrics['failure_recall']:.3f}   (caught {cm[1,1]} of {cm[1].sum()})")
        print(f"failure precision:{metrics['failure_precision']:.3f}")
        print(f"PR-AUC:          {metrics['pr_auc']:.3f}")
        print(f"ROC-AUC:         {metrics['roc_auc']:.3f}")
        print("\nlogged to MLflow + registered model 'failure-predictor'.")
        print("view:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
