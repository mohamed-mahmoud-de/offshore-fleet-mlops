"""
Milestone 4 — TWO rules-based scorecards (no ML), plus context.

We produce two different grades because they answer two different questions:

  CREW score   -> "did the crew do their job?"   (on-time, safety, downtime only —
                   the things a crew actually controls; utilization is excluded)
  VESSEL score -> "is the vessel earning its keep?" (utilization-heavy, commercial),
                   with an override: a near-idle vessel is forced to red.

Context we add so a human can judge a grade fairly:
  missions        -> how many jobs the grade is based on
  low_confidence  -> True when that's only a few (5 perfect jobs != 50 perfect jobs;
                     it could even be a data-entry glitch worth checking).

Run from the project root:
    python -m src.scorecard.build_scorecard
"""
from pathlib import Path

import pandas as pd
import yaml

from src.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
with open(ROOT / "config" / "config.yaml") as f:
    CFG = yaml.safe_load(f)["scorecard"]


def _flag(score: float, th: dict) -> str:
    if score >= th["green"]:
        return "green"
    if score >= th["amber"]:
        return "amber"
    return "red"


def build() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM fleet_features", engine)
    ref, th = CFG["references"], CFG["flag_thresholds"]

    # --- 1) the 4 sub-scores (0-100, higher = better) ----------------------
    df["utilization_score"] = df["utilization_pct"].clip(0, 100)
    df["on_time_score"] = df["on_time_completion_pct"].clip(0, 100)
    df["safety_score"] = (100 - df["hse_incidents_12m"] * (100 / ref["incident_zero_at"])).clip(0, 100)
    df["downtime_score"] = (100 - df["unplanned_downtime_hours_12m"] * (100 / ref["downtime_zero_at_hours"])).clip(0, 100)

    # --- 2) VESSEL commercial score (+ idle override) ----------------------
    vw = CFG["vessel_weights"]
    df["vessel_score"] = (
        vw["utilization"] * df["utilization_score"]
        + vw["on_time_completion"] * df["on_time_score"]
        + vw["safety"] * df["safety_score"]
        + vw["downtime"] * df["downtime_score"]
    ).round(1)
    df["vessel_flag"] = df["vessel_score"].apply(lambda s: _flag(s, th))
    # a near-idle vessel is a commercial problem regardless of the average:
    df.loc[df["utilization_pct"] < CFG["vessel_idle_override_pct"], "vessel_flag"] = "red"

    # --- 3) CREW performance score (crew-controllable only) ----------------
    cw = CFG["crew_weights"]
    df["crew_score"] = (
        cw["on_time_completion"] * df["on_time_score"]
        + cw["safety"] * df["safety_score"]
        + cw["downtime"] * df["downtime_score"]
    ).round(1)
    df["crew_flag"] = df["crew_score"].apply(lambda s: _flag(s, th))

    # crew "top reason" = weakest crew component
    shortfall = pd.DataFrame({
        "Late job completion": cw["on_time_completion"] * (100 - df["on_time_score"]),
        "Safety incidents": cw["safety"] * (100 - df["safety_score"]),
        "High downtime": cw["downtime"] * (100 - df["downtime_score"]),
    })
    df["crew_top_reason"] = shortfall.idxmax(axis=1)
    df.loc[df["crew_flag"] == "green", "crew_top_reason"] = "Performing well"

    # --- 4) context: sample size + confidence ------------------------------
    df["missions"] = df["jobs_completed_12m"]
    df["low_confidence"] = df["jobs_completed_12m"] < CFG["low_activity_jobs"]

    out = df[[
        "vessel_id", "vessel_name", "vessel_type", "crew_id", "contract_status",
        "utilization_pct", "missions", "low_confidence",
        "crew_score", "crew_flag", "crew_top_reason",
        "vessel_score", "vessel_flag",
    ]].sort_values("crew_score").reset_index(drop=True)

    out.to_sql("vessel_scorecard", engine, if_exists="replace", index=False)
    return out


def main() -> None:
    out = build()
    print(f"scored {len(out)} vessels -> table vessel_scorecard\n")
    print("CREW flags:  ", out["crew_flag"].value_counts().to_dict())
    print("VESSEL flags:", out["vessel_flag"].value_counts().to_dict())
    print("low-confidence grades (few missions):", int(out["low_confidence"].sum()))

    print("\n--- Maridive 703 (the idle one we discussed) ---")
    cols = ["utilization_pct", "missions", "low_confidence",
            "crew_score", "crew_flag", "vessel_score", "vessel_flag"]
    print(out.loc[out.vessel_name == "Maridive 703", cols].to_string(index=False))

    print("\n--- lowest CREW scores (genuine crew concerns) ---")
    print(out.head(6)[["vessel_name", "missions", "low_confidence",
                       "crew_score", "crew_flag", "crew_top_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
