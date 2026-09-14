"""
Smoke tests for CI — fast, no database or model files required.

They catch the two things that actually break a pipeline in practice: a module
that no longer imports, and a core pure-function that silently changed behaviour.
"""
import os

os.environ["MLFLOW_ENABLED"] = "0"  # keep imports light (skip the mlflow import path)

import numpy as np
import pandas as pd


def test_modules_import():
    """Every pipeline module imports cleanly (catches syntax / import errors)."""
    import src.db  # noqa: F401
    import src.ingestion.load_raw  # noqa: F401
    import src.features.build_features  # noqa: F401
    import src.scorecard.build_scorecard  # noqa: F401
    import src.ml.train_model  # noqa: F401
    import src.ml.promote_model  # noqa: F401
    import src.monitoring.check_drift  # noqa: F401


def test_scorecard_flag_thresholds():
    from src.scorecard.build_scorecard import CFG, _flag
    th = CFG["flag_thresholds"]
    assert _flag(90, th) == "green"
    assert _flag(60, th) == "amber"
    assert _flag(40, th) == "red"


def test_ks_drift_detects_shift():
    from src.monitoring.check_drift import NUM_FEATURES, ks_drift
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({c: rng.normal(0, 1, 500) for c in NUM_FEATURES})
    same = pd.DataFrame({c: rng.normal(0, 1, 500) for c in NUM_FEATURES})
    shifted = pd.DataFrame({c: rng.normal(5, 1, 500) for c in NUM_FEATURES})

    share_same, _ = ks_drift(ref, same)
    share_shift, _ = ks_drift(ref, shifted)
    assert share_same < 0.3        # same distribution -> little/no drift
    assert share_shift >= 0.5      # clearly shifted -> flagged
