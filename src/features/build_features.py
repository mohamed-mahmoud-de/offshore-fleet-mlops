"""
Milestone 3 — cleaning + feature engineering.

Read the RAW tables from Postgres -> apply structural fixes + engineer new
features -> write model-ready tables (fleet_features, ai4i_features) back.

Only DETERMINISTIC, row-wise transforms happen here — each row is computed from
its own values, so nothing "learns" from the whole dataset and there is no leakage.
Learned transforms (scaling, category encoding, imbalance handling) wait for the
M5 pipeline, where they are fit on the TRAINING split only.

Run from the project root:
    python -m src.features.build_features
"""
import numpy as np
import pandas as pd

from src.db import get_engine


def build_fleet(engine) -> pd.DataFrame:
    """Clean fleet_raw and engineer the operational features the scorecard needs."""
    df = pd.read_sql("SELECT * FROM fleet_raw", engine)

    # --- structural cleaning ---
    # Only 2 vessels lack an owner; a category is safer than dropping the row.
    df["owner"] = df["owner"].fillna("Unknown")
    # NOTE: we deliberately do NOT impute crane/bollard/pax nulls — those are
    # structural (a PSV has no crane) and aren't performance signals, so they're
    # simply left out of the feature table below.

    # --- engineered features (row-wise, domain-informed) ---
    op_days = df["operating_days_12m"].replace(0, np.nan)  # guard divide-by-zero
    # downtime normalised by how much the vessel actually worked (fairer than raw hours)
    df["downtime_per_op_day"] = (df["unplanned_downtime_hours_12m"] / op_days).fillna(0).round(2)
    # safety incidents per 100 operating days
    df["incident_rate_per_100_op_days"] = (df["hse_incidents_12m"] / op_days * 100).fillna(0).round(2)
    # a business feature: roughly how much revenue each vessel generated
    df["revenue_estimate_usd_12m"] = (df["day_rate_usd"] * df["operating_days_12m"]).round(0)
    # simple flag
    df["is_idle"] = (df["contract_status"] == "Idle").astype(int)

    keep = [
        # identity
        "vessel_id", "vessel_name", "vessel_type", "owner", "region",
        "contract_status", "crew_id",
        # scorecard inputs (all non-null)
        "utilization_pct", "on_time_completion_pct",
        "hse_incidents_12m", "unplanned_downtime_hours_12m",
        # context
        "operating_days_12m", "idle_days_12m", "jobs_completed_12m",
        "day_rate_usd", "vessel_age_years",
        # engineered
        "downtime_per_op_day", "incident_rate_per_100_op_days",
        "revenue_estimate_usd_12m", "is_idle",
    ]
    out = df[keep]
    out.to_sql("fleet_features", engine, if_exists="replace", index=False)
    return out


def build_ai4i(engine) -> pd.DataFrame:
    """Clean ai4i_raw: drop leakage/id columns, rename, add physics-based features."""
    df = pd.read_sql("SELECT * FROM ai4i_raw", engine)

    # Drop identifiers (no signal) and the 5 failure-mode flags (they DEFINE the
    # target -> using them as inputs would be textbook leakage).
    df = df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

    # clean, snake_case names
    df = df.rename(columns={
        "Type": "type",
        "Air temperature [K]": "air_temp_k",
        "Process temperature [K]": "process_temp_k",
        "Rotational speed [rpm]": "rotational_speed_rpm",
        "Torque [Nm]": "torque_nm",
        "Tool wear [min]": "tool_wear_min",
        "Machine failure": "machine_failure",
    })

    # Domain-informed engineered features (pure physics, not learned):
    #  - heat dissipation relates to the temp gap (the HDF failure mode)
    df["temp_diff_k"] = (df["process_temp_k"] - df["air_temp_k"]).round(2)
    #  - mechanical power P = torque * angular velocity (the PWF/OSF modes)
    df["power_w"] = (df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60).round(1)

    df.to_sql("ai4i_features", engine, if_exists="replace", index=False)
    return df


def report(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    print(f"rows: {len(df)}   cols: {df.shape[1]}")
    print("columns:", list(df.columns))
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    print("remaining nulls:", dict(nulls) if len(nulls) else "none")


def main() -> None:
    engine = get_engine()
    report(build_fleet(engine), "fleet_features")
    report(build_ai4i(engine), "ai4i_features")
    print("\nMilestone 3 complete: fleet_features + ai4i_features written to Postgres.")


if __name__ == "__main__":
    main()
