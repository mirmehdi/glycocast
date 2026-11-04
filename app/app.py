"""
app.py - GlycoCast demo page (Streamlit).

    streamlit run app/app.py        (from the project folder)

The page only shows things. All the work is done by pipeline.py (the engine):
real patient -> virtual twin -> simulation -> messy device files -> OUR ETL -> SAVED model -> report.
"""

# ============================================================================
# 0. IMPORTS AND SETTINGS
# ============================================================================
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import pipeline as P                     # our engine (app/pipeline.py)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results"
FIGURES = ROOT / "reports" / "figures"

# one colour = one meaning, everywhere on the page
BLUE, ORANGE, GREY, VIOLET = "#2a78d6", "#eb6834", "#9a9893", "#6250d6"
RED, AMBER = "#e34948", "#eda100"
RANGE_BAND, LOW_BAND = "rgba(27,175,122,0.08)", "rgba(227,73,72,0.10)"

st.set_page_config(page_title="GlycoCast", page_icon="🩸", layout="wide")


@st.cache_resource                       # load the models ONCE, not on every click
def models():
    return P.load_models()


# ============================================================================
# APP BLOCK 1 - INPUTS (sidebar)
# ============================================================================
with st.sidebar:
    st.header("1. Real patient")
    age = st.number_input("Age (years)", min_value=7, max_value=70, value=10)
    weight = st.number_input("Body weight (kg)", min_value=20.0, max_value=150.0, value=30.0, step=0.5)
    st.caption(f"Age group: **{P.age_group(age)}** - the app finds the closest virtual twin.")

    st.header("2. Meal plan of the day")
    c1, c2 = st.columns(2)
    b_h = c1.slider("Breakfast", 5.0, 10.0, 7.5, 0.25, format="%.2f h")
    b_g = c2.slider("Breakfast g", 0, 120, 50, 5)
    l_h = c1.slider("Lunch", 11.0, 14.5, 12.5, 0.25, format="%.2f h")
    l_g = c2.slider("Lunch g", 0, 120, 70, 5)
    d_h = c1.slider("Dinner", 17.0, 21.0, 18.5, 0.25, format="%.2f h")
    d_g = c2.slider("Dinner g", 0, 120, 80, 5)

    st.header("3. Real-life problems")
    forgot = st.checkbox("Forgot to log lunch (eaten, not typed in)")
    new_sensor = st.checkbox("New sensor at 08:00 (warm-up error)")
    compression = st.checkbox("Compression low at 03:00 (lying on the sensor)")
    seed = st.number_input("Random seed (sensor noise, day-0 meals)", 0, 999, 1)

    run = st.button("▶ Simulate and predict", type="primary", width="stretch")

if run:                                  # the ONE call into the engine
    with st.spinner("Twin -> simulation -> device files -> ETL -> model -> report ..."):
        r = P.run_all(age, weight, [(b_h, b_g), (l_h, l_g), (d_h, d_g)],
                      forgot, new_sensor, compression, int(seed), models())
        pdf_path = Path(tempfile.gettempdir()) / f"glycocast_{r['reg']['patient_id']}.pdf"
        P.make_pdf(r, pdf_path)
        st.session_state["r"] = r                      # keep the result when the page re-draws
        st.session_state["pdf"] = pdf_path.read_bytes()
        pdf_path.unlink(missing_ok=True)               # privacy: no report file left on disk


# ============================================================================
# APP BLOCK 2 - THE INTERACTIVE DAY CHART
# ============================================================================
def day_chart(r: dict) -> go.Figure:
    """Glucose (true / raw / cleaned / forecast) + alarms + lows; carbs eaten vs logged; insulin."""
    g = r["g"]
    w = ((g["time_local"] >= r["t0"]) & (g["time_local"] <= r["t1"])).to_numpy()
    x = g.loc[w, "time_local"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.66, 0.17, 0.17],
                        subplot_titles=("Glucose (mg/dL)", "Carbs (g): eaten vs logged", "Insulin bolus (U)"))

    # background: target range, low zone, sensor warm-up
    fig.add_hrect(y0=70, y1=180, fillcolor=RANGE_BAND, line_width=0, row=1, col=1, exclude_empty_subplots=False)
    fig.add_hrect(y0=0, y1=70, fillcolor=LOW_BAND, line_width=0, row=1, col=1, exclude_empty_subplots=False)
    wu = g.loc[w & g["is_warmup"].to_numpy(), "time_local"]
    if len(wu):
        fig.add_vrect(x0=wu.min(), x1=wu.max(), fillcolor="rgba(150,150,150,0.15)", line_width=0,
                      annotation_text="sensor warm-up (removed)", annotation_position="top left",
                      row=1, col=1, exclude_empty_subplots=False)

    # the four glucose lines
    fig.add_trace(go.Scatter(x=x, y=g.loc[w, "bg_true"], name="True glucose (simulator)",
                             line=dict(color=ORANGE, width=1.5, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=g.loc[w, "cgm_raw"], name="Raw sensor (as received)",
                             line=dict(color=GREY, width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=g.loc[w, "cgm_clean"], name="Cleaned CGM",
                             line=dict(color=BLUE, width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x + pd.Timedelta(minutes=30), y=g.loc[w, "f30"],   # drawn at the time it predicts
                             name="30-min forecast (made 30 min earlier)",
                             line=dict(color=VIOLET, width=1.5, dash="dot")), row=1, col=1)

    # what the ETL removed: compression low points
    comp = w & g["is_compression"].to_numpy()
    if comp.any():
        fig.add_trace(go.Scatter(x=g.loc[comp, "time_local"], y=g.loc[comp, "cgm_raw"], mode="markers",
                                 name="Removed: compression low",
                                 marker=dict(symbol="circle-open", size=9, color="#0b0b0b")), row=1, col=1)

    # alarms: red = low within 15 min, amber = low within 30 min
    for urgent, color, label in [(True, RED, "🔴 Alarm: low within 15 min - eat now"),
                                 (False, AMBER, "🟠 Alarm: low likely within 30 min - plan a snack")]:
        idx = [a for a, u in zip(r["alarms"], r["urgent"]) if u == urgent and w[a]]
        if idx:
            fig.add_trace(go.Scatter(x=g["time_local"].iloc[idx], y=g["cgm"].iloc[idx], mode="markers",
                                     name=label, marker=dict(symbol="triangle-down", size=15, color=color,
                                                             line=dict(color="white", width=1.5))), row=1, col=1)

    # where real lows started
    if len(r["onsets"]):
        fig.add_trace(go.Scatter(x=g["time_local"].iloc[r["onsets"]], y=[62] * len(r["onsets"]),
                                 mode="markers+text", text=["low starts"] * len(r["onsets"]),
                                 textposition="bottom center", name="Real low (< 70 for 15 min)",
                                 marker=dict(symbol="x", size=11, color=RED)), row=1, col=1)

    # night risk line at 22:00
    if r["night"]:
        t22 = r["t0"].normalize() + pd.Timedelta(hours=22)
        n = r["night"]
        fig.add_vline(x=t22, line=dict(color=VIOLET, width=1, dash="dash"), row=1, col=1)
        fig.add_annotation(x=t22, y=1.0, yref="y domain", xanchor="left", showarrow=False,
                           text=f"22:00 night risk {n['risk']:.0%} -> {'snack advised' if n['snack'] else 'no snack'}",
                           font=dict(color=VIOLET, size=12), row=1, col=1)

    # meals and insulin (a forgotten meal = grey bar without a blue bar)
    five_min = 5 * 60 * 1000                                       # bar width in milliseconds
    fig.add_trace(go.Bar(x=x, y=g.loc[w, "carbs_eaten"], name="Carbs eaten", marker_color="#c9c7c1",
                         width=five_min * 3), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=g.loc[w, "carbs_g"], name="Carbs logged", marker_color=BLUE,
                         width=five_min * 1.5), row=2, col=1)
    fig.add_trace(go.Bar(x=x, y=g.loc[w, "bolus_u"], name="Insulin bolus (pump)", marker_color=VIOLET,
                         width=five_min * 2), row=3, col=1)

    top = max(300, np.nanmax(g.loc[w, ["bg_true", "cgm_raw"]].to_numpy()) + 20)
    fig.update_yaxes(range=[40, top], row=1, col=1)
    fig.update_xaxes(tickformat="%H:%M\n%a", row=3, col=1)
    fig.update_layout(height=760, barmode="overlay", hovermode="x unified", template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0, font=dict(size=11)),
                      margin=dict(l=10, r=10, t=90, b=10))
    return fig


# ============================================================================
# APP BLOCK 3 - THE PAGE: tiles + chart, report, test results
# ============================================================================
st.title("🩸 GlycoCast- mySugr Interview")
st.caption("30-min glucose forecast · low-glucose alarm · night-low risk — for a virtual twin of a real patient.")
st.warning(P.DISCLAIMER, icon="⚠️")

tab_day, tab_report, tab_results, tab_about = st.tabs(
    ["🧪 Patient day", "📄 Patient report", "📊 Test results (9 patients)", "ℹ️ About"])

if "r" not in st.session_state:
    with tab_day:
        st.info("👈 Enter age and weight, design the day, and press **Simulate and predict**.")
else:
    r = st.session_state["r"]
    reg, twin = r["reg"], r["twin"]
    rep = P.report_numbers(r)

    # ---------------- tab 1: the day ----------------
    with tab_day:
        st.subheader(f"Patient {reg['patient_id']} · {reg['diagnosis']['code']} {reg['diagnosis']['text']}")
        st.caption(f"Real patient {r['inputs']['age']} y, {r['inputs']['weight_kg']} kg → twin "
                   f"{twin['Age']:.0f} y, {twin['BW']:.1f} kg, {twin['TDI']:.1f} U/day · match: **{r['quality']}** · "
                   f"shown {r['t0']:%d %b %H:%M} – {r['t1']:%d %b %H:%M}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Real lows", r["lows"])
        k2.metric("Warned in time", f"{r['caught']} of {r['lows']}")
        k3.metric("Median warning", f"{r['lead_min']:.0f} min" if r["lead_min"] else "–")
        k4.metric("False alarms", r["false_alarms"], f"{r['false_but_treated']} treated by a snack",
                  delta_color="off")
        if r["night"]:
            n = r["night"]
            k5.metric("Night risk 22:00", f"{n['risk']:.0%} · {'snack' if n['snack'] else 'ok'}",
                      f"night low: {'yes' if n['low_happened'] else 'no'}", delta_color="off")
        st.plotly_chart(day_chart(r), width="stretch")
        st.caption(f"Alarm rule (chosen on training patients): 30-min forecast < {r['thr']:.0f} mg/dL; "
                   "no alarm while already low; 30-min snooze.")

    # ---------------- tab 2: the report ----------------
    with tab_report:
        st.download_button("⬇ Download PDF report", st.session_state["pdf"],
                           file_name=f"glycocast_report_{reg['patient_id']}.pdf", mime="application/pdf",
                           type="primary")
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**CGM metrics (shown 28 h)**")
            targets = {"data %": "≥ 70", "time in range 70-180 %": "> 70", "time below 70 %": "< 4",
                       "time below 54 %": "< 1", "time above 180 %": "< 25", "CV %": "≤ 36"}
            st.dataframe(pd.DataFrame({"value": pd.Series(rep["cgm"]).round(1),
                                       "consensus target": pd.Series(targets)}).fillna(""), width="stretch")
            f = rep["forecast"]
            err_model, err_same = f["RMSE model (mg/dL)"], f["RMSE 'stays the same' (mg/dL)"]
            st.markdown(f"**30-min forecast:** error {err_model:.1f} mg/dL vs {err_same:.1f} for "
                        f"'stays the same' · MARD {f['MARD %']:.1f} %")
            st.markdown("**Virtual twin**")
            st.dataframe(r["match"], hide_index=True, width="stretch")
        with right:
            st.pyplot(rep["clarke_fig"], width="stretch")
        st.markdown("**Alarms in the shown period**")
        st.dataframe(r["alarm_table"], hide_index=True, width="stretch")
        with st.expander("Intended use, privacy and traceability"):
            st.markdown(f"**Intended use:** {reg['intended_use']}")
            st.markdown(f"**Data minimization (GDPR):** {reg['data_minimization']}. "
                        f"Temporary device files deleted: {'yes' if r['temp_files_deleted'] else 'no'}.")
            st.json(reg["traceability"])
            st.markdown("**Known limitations**\n" + "\n".join(f"- {x}" for x in reg["limitations"]))

# ---------------- tab 3: test results (always available) ----------------
with tab_results:
    st.markdown("**Validated on 9 patients the model never saw (3 children, 3 adolescents, 3 adults).**")
    a, b = st.columns(2)
    with a:
        st.markdown("30-min forecast: the model ladder (RMSE, mg/dL)")
        st.dataframe(pd.read_csv(RESULTS / "forecast_30min_models.csv", index_col=0), width="stretch")
        st.markdown("Near lows (true value < 70): the weakness and the fix")
        st.dataframe(pd.read_csv(RESULTS / "low_range_check.csv", index_col=0), width="stretch")
    with b:
        st.markdown("Low alarm (per low event)")
        st.dataframe(pd.read_csv(RESULTS / "alarm_test.csv", index_col=0), width="stretch")
        st.markdown("Forgotten meals (all carb logs removed)")
        st.dataframe(pd.read_csv(RESULTS / "no_carbs_test.csv", index_col=0), width="stretch")
        st.markdown("Night-low risk at 22:00")
        st.dataframe(pd.read_csv(RESULTS / "night_risk_test.csv", index_col=0), width="stretch")
    st.image(str(FIGURES / "clarke_horizons.png"), caption="Clarke error grid at 15 / 30 / 60 / 120 min")

# ---------------- tab 4: about ----------------
with tab_about:
    st.markdown("""
**Pipeline:** simglucose (UVA/Padova, 30 virtual type-1 patients) → messy device files (UTC / local / no time
zone, mmol/L, gaps, duplicates, warm-up, compression lows, forgotten logs) → ETL (5-min grid, artifact removal,
causal Kalman filter, insulin and carbs on board) → LightGBM, risk-weighted near lows → patient-wise validation.

**Virtual twin:** the 2 virtual patients closest in age and weight (same age group) are blended into a new one.

**Key findings**
- In decision moments (glucose moving ≥ 30 mg/dL) the error drops from 66 to 39 mg/dL; 30-min MARD 9.8 %.
- Risk-weighted training cut the forecast bias near lows by 30 % (Clarke A+B 98.9 %).
- Low alarm: 97 % of lows warned in time, median 15 min ahead, about 3 false alarms per day.
- Forgetting all meal logs raises the error only ~2 % — the pump's bolus still signals the meal.
- Sensor error is slow drift (autocorrelation 0.96); it corrupts 1 in 3 night-low labels.
""")