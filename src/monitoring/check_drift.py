"""
Milestone 9 — monitoring: data-drift detection.

Answers the question a live model must keep answering: "has the incoming data
shifted enough that the model should be retrained?" — so we retrain *when needed*
instead of blindly every day.

  - Transparent signal: a per-feature Kolmogorov-Smirnov test -> share of drifted
    features -> a clear retrain / no-retrain verdict (this is the automatable part).
  - Rich visual: an Evidently HTML report saved under reports/monitoring/.

To demonstrate both outcomes we run two scenarios against the training reference:
  1. a fresh random sample            -> no real drift  -> "no retrain needed"
  2. a shifted batch (simulated aging) -> drift          -> "retrain recommended"

Run from the project root:
    python -m src.monitoring.check_drift
"""
import math
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "monitoring"
NUM_FEATURES = ["air_temp_k", "process_temp_k", "rotational_speed_rpm",
                "torque_nm", "tool_wear_min", "temp_diff_k", "power_w"]
DRIFT_SHARE_THRESHOLD = 0.3   # >= 30% of features drifted -> recommend retraining


def load_reference() -> pd.DataFrame:
    """The training data is our drift reference (a real setup snapshots it at train time)."""
    df = pd.read_sql("SELECT * FROM ai4i_features", get_engine())
    return df.drop(columns=["machine_failure"])


def ks_drift(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05):
    """Share of numeric features whose distribution shifted (KS test p < alpha)."""
    drifted = [c for c in NUM_FEATURES
               if ks_2samp(reference[c], current[c]).pvalue < alpha]
    return len(drifted) / len(NUM_FEATURES), drifted


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> None:
    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    snapshot.save_html(str(path))


def run_scenario(name: str, reference, current, out_html: Path) -> bool:
    share, drifted = ks_drift(reference, current)
    evidently_report(reference, current, out_html)
    retrain = share >= DRIFT_SHARE_THRESHOLD
    verdict = "DRIFT DETECTED -> retrain recommended" if retrain else "no significant drift -> no retrain needed"
    print(f"\n[{name}]")
    print(f"  drifted features: {len(drifted)}/{len(NUM_FEATURES)} ({share*100:.0f}%)  {drifted}")
    print(f"  verdict: {verdict}")
    print(f"  report:  {out_html}")
    return retrain


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = load_reference()

    # scenario 1 — a fresh batch drawn from the same distribution (no real drift)
    fresh = reference.sample(frac=0.5, random_state=1).reset_index(drop=True)
    run_scenario("scenario 1: fresh batch", reference, fresh, OUT / "drift_none.html")

    # scenario 2 — a shifted batch: simulate equipment aging / new operating regime
    shifted = reference.sample(frac=0.5, random_state=2).reset_index(drop=True).copy()
    shifted["tool_wear_min"] = shifted["tool_wear_min"] + 60
    shifted["torque_nm"] = shifted["torque_nm"] * 1.15
    shifted["rotational_speed_rpm"] = shifted["rotational_speed_rpm"] + 300
    shifted["power_w"] = (shifted["torque_nm"] * shifted["rotational_speed_rpm"]
                          * 2 * math.pi / 60).round(1)   # keep the engineered feature consistent
    run_scenario("scenario 2: shifted batch", reference, shifted, OUT / "drift_detected.html")

    print("\nMonitoring complete. Open the HTML reports in reports/monitoring/ for the full breakdown.")


if __name__ == "__main__":
    main()
