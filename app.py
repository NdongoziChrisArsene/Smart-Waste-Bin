import json
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ART = Path(__file__).parent / "artifacts"
st.set_page_config(page_title="Smart Waste-Bin", page_icon="🗑️", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(ART / "bin_model.pkl")


@st.cache_data
def load_metrics():
    return json.load(open(ART / "metrics.json"))


def predict_hours(bundle, fill, weight, now, last_collection):
    hours_since = max((now - last_collection).total_seconds() / 3600, bundle["min_hours"])
    row = pd.DataFrame([{
        "fill_pct": fill,
        "weight_kg": weight,
        "hours_since_collection": hours_since,
        "avg_fill_rate": fill / hours_since,
        "hour": now.hour,
        "is_weekend": int(now.weekday() >= 5),
    }])[bundle["features"]]
    return max(float(bundle["model"].predict(row)[0]), 0.0), hours_since


st.title("🗑️ Smart Waste-Bin System")
st.caption("Predicts how long until the bin is full (Multiple Linear Regression)")

if not (ART / "bin_model.pkl").exists():
    st.error("Model not found. Run the training notebook first so it creates the `artifacts/` folder.")
    st.stop()

bundle, metrics = load_model(), load_metrics()
if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["Predict", "Model and data"])

with tab1:
    c1, c2 = st.columns(2)
    fill = c1.slider("Current fill level (%)", 0, 100, 40)
    weight = c2.number_input("Current weight (kg)", 0.0, 100.0, 8.0, step=0.5)

    st.markdown("**Last collection (when the bin was emptied)**")
    d1, d2 = st.columns(2)
    default_last = dt.datetime.now() - dt.timedelta(hours=24)
    last_date = d1.date_input("Date", default_last.date())
    last_time = d2.time_input("Time", default_last.time().replace(second=0, microsecond=0))

    use_now = st.checkbox("Use current time as 'now'", value=True)
    if use_now:
        now = dt.datetime.now()
    else:
        n1, n2 = st.columns(2)
        now = dt.datetime.combine(n1.date_input("Now: date", dt.date.today()),
                                  n2.time_input("Now: time", dt.datetime.now().time().replace(second=0, microsecond=0)))
    last_collection = dt.datetime.combine(last_date, last_time)

    if st.button("Predict time until full", type="primary"):
        if last_collection >= now:
            st.error("The last collection must be before the current time.")
        else:
            threshold = bundle["full_threshold"]
            hours, since = predict_hours(bundle, fill, weight, now, last_collection)
            st.progress(fill / 100, text=f"Fill level: {fill}%")
            if fill >= threshold:
                st.error(f"Bin is already at or above {threshold}%. Collect now.")
                result = "Full: collect now"
            else:
                h, m = int(hours), int(round((hours % 1) * 60))
                st.metric("Estimated time until full", f"{h} h {m} min")
                st.write(f"Expected full around **{(now + dt.timedelta(hours=hours)):%a %d %b, %H:%M}**")
                if hours <= 6:
                    st.warning("Collect soon.")
                else:
                    st.success("No urgent collection needed.")
                result = f"{h} h {m} min"
            st.session_state.history.append({
                "Time": f"{now:%d %b %H:%M}", "Fill %": fill, "Weight kg": weight,
                "Hours since collection": round(since, 1), "Result": result})

    if st.session_state.history:
        st.markdown("**This session**")
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)

with tab2:
    st.write(f"**Data source:** {metrics['data_source']}  |  **Rows used:** {metrics['rows']}")
    st.write(f"**Weight:** {metrics['weight']}")
    st.write(f"Bin counts as full at **{bundle['full_threshold']}%**.")
    st.markdown("**Test-set results (hours)**")
    st.dataframe(pd.DataFrame({"Baseline (fill % only)": metrics["baseline"],
                               "MLR (all features)": metrics["mlr"],
                               "Ridge": metrics["ridge"]}).T)
    st.markdown("**Regression equation coefficients**")
    st.dataframe(pd.Series(metrics["coefficients"], name="coefficient").round(4).to_frame())
    st.caption(f"Intercept: {metrics['intercept']:.3f}")
    for img in ("eda.png", "evaluation.png"):
        if (ART / img).exists():
            st.image(str(ART / img))
