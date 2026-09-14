"""
Milestone 2 — load the raw datasets into Postgres and run data-quality checks.

Principle of the RAW layer: mirror the source faithfully. We load the CSVs
AS-IS (no cleaning, no fixing) into raw tables, then CHECK them (detect problems,
don't fix — fixes happen in later stages). Re-running is safe: it replaces the
tables, so the result is always identical (idempotent).

Run from the project root:
    python -m src.ingestion.load_raw
"""
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.db import get_engine

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

# table name in Postgres  ->  source CSV
DATASETS = {
    "fleet_raw": RAW / "maridive_fleet_synthetic.csv",
    "ai4i_raw": RAW / "ai4i2020.csv",
}


def load_table(engine, table: str, csv_path: Path) -> pd.DataFrame:
    """Read a CSV and write it to Postgres, replacing any existing table."""
    df = pd.read_csv(csv_path)
    df.to_sql(table, engine, if_exists="replace", index=False)
    print(f"loaded {table:<10} <- {csv_path.name}  ({len(df)} rows, {df.shape[1]} cols)")
    return df


def quality_report(df: pd.DataFrame, table: str) -> None:
    """Print row/column counts, duplicate rows, and nulls per column."""
    print(f"\n--- quality report: {table} ---")
    print(f"rows: {len(df)}   columns: {df.shape[1]}")
    print(f"duplicate rows: {df.duplicated().sum()}")

    nulls = df.isna().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    if len(nulls):
        print("nulls per column:")
        for col, n in nulls.items():
            print(f"  {col:<28} {n:>4}  ({n / len(df) * 100:4.1f}%)")
    else:
        print("nulls: none")


def validity_checks(dfs: dict[str, pd.DataFrame]) -> None:
    """Targeted business-rule checks — the kind a real pipeline asserts on."""
    fleet, ai4i = dfs["fleet_raw"], dfs["ai4i_raw"]
    checks = [
        ("fleet utilization_pct within 0-100", fleet["utilization_pct"].between(0, 100).all()),
        ("fleet operating_days_12m not negative", (fleet["operating_days_12m"] >= 0).all()),
        ("fleet vessel_id is unique", fleet["vessel_id"].is_unique),
        ("ai4i 'Machine failure' is binary {0,1}", set(ai4i["Machine failure"].unique()) <= {0, 1}),
        ("ai4i has no missing values", int(ai4i.isna().sum().sum()) == 0),
    ]
    print("\n=== validity checks ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


def verify_in_postgres(engine) -> None:
    """Ask Postgres itself what tables exist and how many rows — trust, but verify."""
    print("\n=== tables now in Postgres ===")
    with engine.connect() as conn:
        tables = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )).fetchall()
        for (t,) in tables:
            n = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
            print(f"  {t:<12} {n} rows")


def main() -> None:
    engine = get_engine()
    dfs = {t: load_table(engine, t, p) for t, p in DATASETS.items()}
    for table, df in dfs.items():
        quality_report(df, table)
    validity_checks(dfs)
    verify_in_postgres(engine)
    print("\nMilestone 2 complete: raw data is in Postgres and checked.")


if __name__ == "__main__":
    main()
