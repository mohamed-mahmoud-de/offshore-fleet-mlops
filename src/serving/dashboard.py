"""
Milestone 8 — Streamlit serving dashboard.

Two views:
  1. Fleet performance scorecard (crew + vessel grades) from the vessel_scorecard table.
  2. A live equipment failure-risk predictor using the trained LightGBM model.

Reads from Postgres (via src.db) and loads models/failure_model.joblib.

Run:
    .venv\\Scripts\\python -m streamlit run src/serving/dashboard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

# make `src` importable no matter how Streamlit launches this file
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.db import get_engine  # noqa: E402

MODEL_PATH = ROOT / "models" / "failure_model.joblib"
# must match the training feature order (ai4i_features minus the target)
FEATURE_ORDER = ["type", "air_temp_k", "process_temp_k", "rotational_speed_rpm",
                 "torque_nm", "tool_wear_min", "temp_diff_k", "power_w"]
FLAG = {"green": "🟢", "amber": "🟡", "red": "🔴"}

st.set_page_config(page_title="Offshore Fleet Ops", page_icon="🚢", layout="wide")


@st.cache_data(ttl=300)
def load_scorecard() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM vessel_scorecard", get_engine())


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def scorecard_view() -> None:
    st.subheader("Fleet performance scorecard")
    try:
        df = load_scorecard()
    except Exception as e:
        st.error(f"Could not load the scorecard from Postgres: {e}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vessels", len(df))
    c2.metric("🟢 Crew performing", int((df.crew_flag == "green").sum()))
    c3.metric("🟡 Crew needs attention", int((df.crew_flag == "amber").sum()))
    c4.metric("🔴 Vessels idle (commercial)", int((df.vessel_flag == "red").sum()))

    f1, f2 = st.columns(2)
    crew_sel = f1.multiselect("Crew flag", ["green", "amber", "red"],
                              default=["green", "amber", "red"])
    types = sorted(df.vessel_type.unique())
    type_sel = f2.multiselect("Vessel type", types, default=types)

    view = df[df.crew_flag.isin(crew_sel) & df.vessel_type.isin(type_sel)].copy()
    view = view.sort_values("crew_score")
    view["Crew grade"] = view.crew_flag.map(FLAG) + " " + view.crew_score.round(1).astype(str)
    view["Vessel grade"] = view.vessel_flag.map(FLAG) + " " + view.vessel_score.round(1).astype(str)

    table = view[["vessel_name", "vessel_type", "missions", "low_confidence",
                  "Crew grade", "crew_top_reason", "Vessel grade"]].rename(columns={
        "vessel_name": "Vessel", "vessel_type": "Type", "missions": "Missions",
        "low_confidence": "Low confidence", "crew_top_reason": "Crew — top reason"})
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "**Crew grade** = what the crew controls (on-time, safety, downtime). "
        "**Vessel grade** = commercial (utilization-heavy; idle vessels forced red). "
        "**Low confidence** = grade based on too few missions to fully trust.")


def predictor_view() -> None:
    st.subheader("Equipment failure-risk predictor")
    st.caption("Set the machine sensor readings and click **Predict** — the LightGBM model estimates failure probability.")
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load the model ({MODEL_PATH.name}). Train it first. Details: {e}")
        return

    # A form batches the inputs so nothing recomputes until the button is clicked
    # (avoids the per-field "Press Enter to apply" confusion).
    with st.form("predictor"):
        c1, c2, c3 = st.columns(3)
        ptype = c1.selectbox("Product type", ["L", "M", "H"], index=0)
        air = c1.number_input("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
        proc = c2.number_input("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
        rpm = c2.number_input("Rotational speed (rpm)", 1000, 3000, 1500, 10)
        torque = c3.number_input("Torque (Nm)", 3.0, 80.0, 40.0, 0.5)
        wear = c3.number_input("Tool wear (min)", 0, 260, 100, 1)
        st.form_submit_button("🔎 Predict failure risk", type="primary", use_container_width=True)

    temp_diff = round(proc - air, 2)
    power = round(torque * rpm * 2 * np.pi / 60, 1)
    row = pd.DataFrame([[ptype, air, proc, rpm, torque, wear, temp_diff, power]],
                       columns=FEATURE_ORDER)
    proba = float(model.predict_proba(row)[0, 1])

    st.markdown("### Result")
    m1, m2 = st.columns([1, 2])
    m1.metric("Failure probability", f"{proba * 100:.1f}%")
    m2.progress(min(max(proba, 0.0), 1.0))
    if proba >= 0.5:
        st.error("⚠️ High failure risk — schedule maintenance.")
    elif proba >= 0.2:
        st.warning("Elevated risk — monitor closely.")
    else:
        st.success("Low risk — normal operation.")
    with st.expander("Engineered features the model used"):
        st.write({"temp_diff_k": temp_diff, "power_w": power})


st.title("🚢 Offshore Fleet Operations")
st.caption("Portfolio MLOps demo — synthetic fleet data + public AI4I model.")
tab1, tab2 = st.tabs(["📊 Performance scorecard", "🔧 Failure-risk predictor"])
with tab1:
    scorecard_view()
with tab2:
    predictor_view()
