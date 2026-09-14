"""
Model-selection benchmark with LazyPredict.

Trains ~30 classifiers with DEFAULT settings on ai4i_features and ranks them, so we
don't just assume RandomForest is best — we check.

IMPORTANT: LazyPredict uses default hyper-parameters and NO imbalance handling, so
plain Accuracy is misleading here (3.4% failures). We rank by **Balanced Accuracy**
and also read **F1** and **ROC AUC**. This is a screen, not the final word — the
winners still need proper tuning + class weights (that's M5/M6).

Run from the project root:
    python -m src.ml.benchmark_models
"""
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from src.db import get_engine

warnings.filterwarnings("ignore")


def main() -> None:
    from lazypredict.Supervised import LazyClassifier  # imported late (heavy)

    df = pd.read_sql("SELECT * FROM ai4i_features", get_engine())
    y = df["machine_failure"]
    # one-hot the only categorical; make everything numeric for a clean benchmark
    X = pd.get_dummies(df.drop(columns=["machine_failure"]), columns=["type"]).astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False)
    models, _ = clf.fit(X_train, X_test, y_train, y_test)

    leaderboard = models.sort_values("Balanced Accuracy", ascending=False)
    pd.set_option("display.width", 130)
    pd.set_option("display.max_rows", 40)
    print("\n=== leaderboard (ranked by Balanced Accuracy) ===\n")
    print(leaderboard.to_string())


if __name__ == "__main__":
    main()
