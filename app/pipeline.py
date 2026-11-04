"""
pipeline.py - the GlycoCast engine for ONE new patient (used by the Streamlit app).

Block 1: Find a virtual twin for a real patient (from age + weight)
Block 2: Simulate the twin for 3 days (day 0 = history with normal meals, day 1 = YOUR meal plan)
Block 3: Turn the simulation into messy device files (CGM + logbook), controlled by the checkboxes
Block 4: Run OUR ETL (src/etl.py) and OUR model features (src/train.py) on those files
Block 5: Predict with the SAVED models, raise alarms, score the shown day - plus ONE function that runs all
Block 6: Report numbers for this patient + Clarke error grid (lows in red)
Block 7: Regulatory items - pseudonymous ID, ICD-10 code, traceability, intended use, temp files deleted
Block 8: The PDF patient report (reportlab) - incl. a static day chart and the Clarke plot
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import contextlib                        # to silence the ETL's print messages
import hashlib                           # fingerprints: pseudonymous ID, model file hash
import io
import json
import shutil                            # delete the temporary device files after use
import sys
import tempfile                          # a temporary folder for the device files
from datetime import datetime, timedelta
from pathlib import Path

import joblib                            # load the saved, trained models
import matplotlib
matplotlib.use("Agg")                    # draw figures to memory/files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# make our own project code (src/) importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simglucose.actuator.pump import InsulinPump                                    # noqa: E402
from simglucose.controller.base import Action                                       # noqa: E402
from simglucose.controller.basal_bolus_ctrller import CONTROL_QUEST, BBController   # noqa: E402
from simglucose.patient.t1dpatient import PATIENT_PARA_FILE, T1DPatient             # noqa: E402
from simglucose.sensor.cgm import CGMSensor                                         # noqa: E402
from simglucose.simulation.scenario import CustomScenario                           # noqa: E402
from simglucose.simulation.sim_engine import SimObj                                 # noqa: E402
from simulate import RescueEnv          # noqa: E402  our world with the 15-15 hypo rule (src/simulate.py)
from export_devices import make_logbook  # noqa: E402  logbook with human errors (src/export_devices.py)
import etl                               # noqa: E402  our ETL functions (src/etl.py)
import train                             # noqa: E402  our feature + alarm functions (src/train.py)
from metrics import mard, plot_clarke, rmse   # noqa: E402  (src/metrics.py)

START = datetime(2026, 1, 5)            # day 0 starts at midnight (local time)
DAYS = 3                                # day 0 = history, day 1 = the shown day, day 2 = next morning
SHOW_DAY = 1
TZ = "Europe/Vienna"                    # the patient's home time zone
PER_DAY = 288                           # 5-min steps per day
MODEL_FILE = ROOT / "models" / "glycocast_v1.joblib"   # made by src/train.py (Block 5b)
TRAIN_SUMMARY = ROOT / "reports" / "results" / "summary.json"   # when/how the model was trained
PIPELINE_VERSION = "app-1.0"            # change this when the demo pipeline changes


# ============================================================================
# 2. BLOCK 1 - FIND A VIRTUAL TWIN
# ============================================================================
def age_group(age: float) -> str:
    """Age -> the simglucose age group (children 7-12, adolescents 14-19, adults 26-68)."""
    if age < 13:
        return "child"
    if age < 20:
        return "adolescent"
    return "adult"


def virtual_patients() -> pd.DataFrame:
    """All 30 virtual patients: ~60 body numbers + therapy settings + age, weight, daily insulin."""
    body = pd.read_csv(PATIENT_PARA_FILE)                  # 30 rows x 62 columns (body physiology)
    therapy = pd.read_csv(CONTROL_QUEST)                   # Age, TDI, CR, CF (therapy settings)
    vp = body.merge(therapy, on="Name")
    vp["group"] = vp["Name"].str.split("#").str[0]         # "adult#003" -> "adult"
    return vp


def find_twin(age: float, weight: float):
    """Find the 2 most similar virtual patients (same age group) and BLEND them into a new one.

    Returns (twin, match):
      twin  = one row of numbers for the new virtual patient (body + therapy)
      match = small table for the report: who was used, how close, how much weight
    """
    vp = virtual_patients()
    group = age_group(age)
    cand = vp[vp["group"] == group].copy()                 # only the same age group (10 patients)

    # distance on a common scale: how many "standard deviations" apart (age and weight count equally)
    cand["distance"] = np.sqrt(((cand["Age"] - age) / cand["Age"].std()) ** 2 +
                               ((cand["BW"] - weight) / cand["BW"].std()) ** 2)
    two = cand.nsmallest(2, "distance")                    # the 2 closest virtual patients

    # blend weights: the closer patient counts more (inverse distance)
    d1, d2 = two["distance"].to_numpy()
    w1 = 1.0 if d1 == 0 else d2 / (d1 + d2)
    weights = np.array([w1, 1 - w1])

    # the new patient = weighted average of ALL numbers of the two donors
    numeric = two.select_dtypes("number").columns.drop("distance")
    twin = (two[numeric].mul(weights, axis=0)).sum().astype(object)   # object: holds numbers AND text
    twin["Name"] = f"twin-{group}"
    twin["group"] = group

    match = pd.DataFrame({
        "donor": two["Name"].to_numpy(),
        "age": two["Age"].to_numpy(), "weight_kg": two["BW"].round(1).to_numpy(),
        "daily_insulin_u": two["TDI"].round(1).to_numpy(),
        "distance": two["distance"].round(2).to_numpy(), "blend_weight": weights.round(2),
    })
    return twin, match


def match_quality(age: float, weight: float, twin: pd.Series) -> str:
    """Is the twin close to the real patient? (simple, explainable rule)"""
    close_age = abs(twin["Age"] - age) <= 5                 # within 5 years
    close_weight = abs(twin["BW"] - weight) / weight <= 0.10   # within 10 % body weight
    return "good" if (close_age and close_weight) else "approximate - no close virtual patient"


# ============================================================================
# 3. BLOCK 2 - SIMULATE THE TWIN
# ============================================================================
class TwinController(BBController):
    """The same manual basal-bolus therapy as in training, but with the TWIN's own settings.
    (simglucose's controller looks settings up by patient name - our twin has no row there.)"""

    def __init__(self, twin: pd.Series, target: float = 140):
        super().__init__(target)
        self.cr, self.cf = float(twin["CR"]), float(twin["CF"])        # carb ratio, correction factor
        self.basal = float(twin["u2ss"]) * float(twin["BW"]) / 6000    # background insulin (U/min)

    def _bb_policy(self, name, meal, glucose, sample_time):
        bolus = 0.0
        if meal > 0:                                                   # a meal is happening now
            carbs = meal * sample_time                                 # g/min x min = grams
            correction = (glucose > 150) * (glucose - self.target) / self.cf
            bolus = (carbs / self.cr + correction) / sample_time       # units -> U/min
        return Action(basal=self.basal, bolus=bolus)


def twin_body(twin: pd.Series) -> pd.Series:
    """The twin's body numbers in EXACTLY the column order of simglucose's file
    (simglucose reads the start state by position: columns 2 to 14)."""
    cols = pd.read_csv(PATIENT_PARA_FILE, nrows=0).columns              # Name, i, x0_ 1, ..., x0_13, ...
    values = [twin["Name"]] + [float(twin[c]) for c in cols[1:]]
    return pd.Series(values, index=cols, dtype=object)


def meal_plan(user_meals: list, seed: int) -> list:
    """All meals for the 3 days as (hour from start, grams).
    Day 0: normal random meals (history). Day 1: YOUR plan. Day 2: your breakfast only."""
    rng = np.random.default_rng(seed)
    day0 = [(rng.uniform(7, 8.5), rng.integers(40, 61)),               # breakfast
            (rng.uniform(12, 13.5), rng.integers(55, 81)),             # lunch
            (rng.uniform(18, 19.5), rng.integers(60, 91))]             # dinner
    day1 = [(24 + h, g) for h, g in user_meals if g > 0]
    day2 = [(48 + user_meals[0][0], user_meals[0][1])] if user_meals[0][1] > 0 else []
    return [(float(h), float(g)) for h, g in day0 + day1 + day2]


def simulate_twin(twin: pd.Series, user_meals: list, seed: int) -> pd.DataFrame:
    """Run the twin for 3 days. Returns one row per 5 min (same columns as data/sim/clean.parquet)."""
    body = twin_body(twin)
    patient = T1DPatient(body, init_state=body.iloc[2:15].astype(float).to_numpy())
    env = RescueEnv(patient,
                    CGMSensor.withName("GuardianRT", seed=seed),       # 5-min CGM with noise
                    InsulinPump.withName("Insulet"),
                    CustomScenario(start_time=START, scenario=meal_plan(user_meals, seed)))
    sim = SimObj(env, TwinController(twin), timedelta(days=DAYS), animate=False)
    sim.simulate()

    df = sim.results().reset_index().iloc[:-1].copy()                  # last row has no action yet
    df["rescue_g"] = env.rescue_hist
    return pd.DataFrame({"time": df["Time"], "bg_true": df["BG"], "cgm": df["CGM"],
                         "carbs_g": df["CHO"] * 5, "insulin_u": df["insulin"] * 5,
                         "rescue_g": df["rescue_g"]})


# ============================================================================
# 4. BLOCK 3 - MESSY DEVICE FILES (like a real app receives them)
# ============================================================================
def messy_cgm(sim: pd.DataFrame, pid: str, new_sensor: bool, compression: bool,
              rng: np.random.Generator) -> pd.DataFrame:
    """The CGM export: the same artifacts as in Step 2b, but switched on/off by the user."""
    n = len(sim)
    cgm = sim["cgm"].to_numpy(float).copy()
    reported = rng.random(n) > 0.02                         # always: ~2 % of values lost (Bluetooth)
    sensor_no = np.ones(n, dtype=int)

    if new_sensor:                                          # sensor change at 08:00 on the shown day
        change = SHOW_DAY * PER_DAY + 8 * 12                # 8 h x 12 steps per hour
        reported[change - 6:change] = False                 # 30 min without data while changing
        sensor_no[change:] = 2                              # from now on: sensor no. 2
        k = np.arange(24)                                   # first 2 h of the new sensor:
        cgm[change + k] += 30 * np.exp(-k / 8) + rng.normal(0, 12, 24)   # reads too high + noisy

    if compression:                                         # lying on the sensor at 03:00 that night
        i0 = (SHOW_DAY + 1) * PER_DAY + 3 * 12
        shape = np.concatenate([np.linspace(0, 1, 4)[1:],   # fast drop (15 min)
                                np.ones(5),                 # flat (25 min)
                                np.linspace(1, 0, 4)[:-1]]) # fast recovery (15 min)
        cgm[i0:i0 + len(shape)] -= 55 * shape               # 55 mg/dL fake drop

    cgm = np.clip(np.round(cgm), 40, 400)                   # device limits, whole numbers
    t_local = sim["time"][reported] + pd.to_timedelta(rng.uniform(-60, 60, reported.sum()), unit="s")
    out = pd.DataFrame({
        "patient_id": pid,
        "sensor_id": [f"SG-{pid}-{s}" for s in sensor_no[reported]],
        "timestamp_utc": t_local.dt.tz_localize(TZ).dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "glucose_mg_dl": cgm[reported].astype(int),
    })
    dup = out.sample(frac=0.01, random_state=int(rng.integers(1_000_000)))   # always: ~1 % duplicates
    return pd.concat([out, dup]).sort_values("timestamp_utc")


def write_device_files(sim: pd.DataFrame, pid: str, age_group_: str, forgot_lunch: bool,
                       new_sensor: bool, compression: bool, seed: int) -> Path:
    """Write cgm.csv, logbook.csv and patients.csv into a NEW temporary folder. Returns the folder."""
    rng = np.random.default_rng(seed)
    sim = sim.assign(patient_id=pid)

    logged = sim.copy()                                     # what the person TYPES into the app
    if forgot_lunch:
        day = (logged["time"] - START).dt.days
        hour = logged["time"].dt.hour + logged["time"].dt.minute / 60
        lunch = (day == SHOW_DAY) & hour.between(11, 15)    # lunch of the shown day
        logged.loc[lunch, "carbs_g"] = 0.0                  # eaten, but not typed in (bolus stays!)

    folder = Path(tempfile.mkdtemp(prefix="glycocast_"))
    messy_cgm(sim, pid, new_sensor, compression, rng).to_csv(folder / "cgm.csv", index=False)
    make_logbook(logged, rng).to_csv(folder / "logbook.csv", index=False)
    pd.DataFrame([{"patient_id": pid, "age_group": age_group_}]).to_csv(folder / "patients.csv", index=False)
    return folder


# ============================================================================
# 5. BLOCK 4 - ETL + FEATURES (exactly the project code, nothing re-written)
# ============================================================================
def run_etl(folder: Path) -> pd.DataFrame:
    """Device files -> clean 5-min table with model features. Same steps, same order as src/etl.py."""
    etl.RAW_DIR = folder                                    # tell the ETL where the files are
    with contextlib.redirect_stdout(io.StringIO()):         # hide its print() lines
        cgm = etl.load_cgm()                                # Block 1: UTC times, duplicates removed
        logbook = etl.load_logbook()                        #          local time + offset -> UTC
        patients = etl.load_patients()

    g = etl.to_grid(cgm)                                    # Block 2: 5-min grid, gaps visible
    g = etl.remove_warmup(g)                                #          first 2 h of each sensor
    g = etl.remove_compression(g)                           #          fake night lows
    g = etl.fill_short_gaps(g)                              #          gaps <= 15 min (target only)
    g = etl.add_kalman(g, logbook)                          # Block 3: causal Kalman (speed)
    g = etl.add_events(g, logbook)                          # Block 4: bolus / carbs / hypo on the grid
    g = etl.add_features(g, patients)                       #          insulin & carbs on board, time of day
    g = etl.add_targets(g)                                  #          future values (only for scoring)

    g, _ = train.make_features(g)                           # model inputs: lags, changes, missing count
    return g


# ============================================================================
# 6. BLOCK 5 - PREDICT, ALARM, SCORE
# ============================================================================
def load_models() -> dict:
    """The models trained in src/train.py. They were trained on 21 patients - NOT on this twin."""
    return joblib.load(MODEL_FILE)


def predict(g: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Add the 30-min and 15-min forecasts (the models predict the CHANGE -> add 'now' back)."""
    X = g[models["features"]]                               # the 20 inputs, same order as in training
    has_now = g["cgm"].notna()                              # no glucose now -> no forecast
    g["f30"] = np.where(has_now, g["cgm"] + models["forecast_30"].predict(X), np.nan)
    g["f15"] = np.where(has_now, g["cgm"] + models["forecast_15"].predict(X), np.nan)
    return g


def shown_window(g: pd.DataFrame):
    """The part we show and score: 06:00 on the shown day -> 10:00 next morning (28 h)."""
    t0 = g["time_local"].iloc[0].normalize() + pd.Timedelta(days=SHOW_DAY, hours=6)
    t1 = t0 + pd.Timedelta(hours=28)
    return t0, t1, ((g["time_local"] >= t0) & (g["time_local"] <= t1)).to_numpy()


def alarms_and_score(g: pd.DataFrame, models: dict) -> dict:
    """Raise alarms like the app would (same rules as in training) and score the shown window."""
    thr = models["alarm_threshold_mgdl"]
    alarms = train.alarm_times(g["f30"].to_numpy(), g["cgm"].to_numpy(), thr, higher_is_risk=False)
    urgent = g["f15"].to_numpy()[alarms] < train.LOW        # red: low expected within 15 min
    onsets = train.low_onsets(g["cgm_clean"].to_numpy())    # real lows (< 70 for >= 15 min)

    t0, t1, win = shown_window(g)
    lows = [o for o in onsets if win[o]]                    # only the shown window counts
    shown_alarms = [a for a in alarms if win[a]]
    W = train.ALARM_WINDOW                                  # 6 steps = 30 min
    caught = [o for o in lows if np.any((alarms >= o - W) & (alarms < o))]
    leads = [5 * (o - alarms[(alarms >= o - W) & (alarms < o)].min()) for o in caught]
    false = [a for a in shown_alarms if not np.any((onsets > a) & (onsets <= a + W))]

    # one line per alarm in the window - for the report: what happened in the 30 min after it?
    rows = []
    for a, red in zip(alarms, urgent):
        if not win[a]:
            continue
        after = g.iloc[a + 1:a + W + 1]                     # the next 30 minutes
        if a not in false:
            outcome = "real low followed"
        elif after["hypo_g"].sum() > 0:
            outcome = "hypo snack logged (patient treated it)"   # the feedback loop
        else:
            outcome = "no low (false alarm)"
        rows.append({"time": g["time_local"].iloc[a].strftime("%d %b %H:%M"),
                     "glucose": round(g["cgm"].iloc[a]), "forecast_30": round(g["f30"].iloc[a]),
                     "urgency": "red: eat now" if red else "orange: plan a snack",
                     "lowest_next_30min": round(after["cgm_clean"].min()), "outcome": outcome})
    alarm_table = pd.DataFrame(rows)
    treated = int((alarm_table["outcome"].str.startswith("hypo")).sum()) if len(rows) else 0

    return {"alarms": alarms, "urgent": urgent, "onsets": np.array(lows, dtype=int),
            "t0": t0, "t1": t1, "thr": thr, "alarm_table": alarm_table,
            "lows": len(lows), "caught": len(caught), "false_alarms": len(false),
            "false_but_treated": treated,
            "lead_min": float(np.median(leads)) if leads else None}


def night_risk(g: pd.DataFrame, models: dict):
    """Risk at 22:00 on the shown day (same night features as in training)."""
    nights = train.build_nights(g)
    shown_date = (g["time_local"].iloc[0].normalize() + pd.Timedelta(days=SHOW_DAY)).date()
    row = nights[nights["night"] == shown_date]
    if row.empty:
        return None
    X = row[models["night_features"]].astype(float)
    risk = float(models["night_model"].predict_proba(X)[:, 1][0])
    return {"risk": risk, "snack": risk >= models["night_threshold"],
            "low_happened": bool(row["night_low"].iloc[0])}


def run_all(age, weight, meals, forgot_lunch, new_sensor, compression, seed, models) -> dict:
    """THE one function the app calls: real patient -> twin -> simulation -> files -> ETL -> models."""
    twin, match = find_twin(age, weight)                                   # Block 1
    sim = simulate_twin(twin, meals, seed)                                 # Block 2
    folder = write_device_files(sim, "NEW01", twin["group"], forgot_lunch,
                                new_sensor, compression, seed)             # Block 3
    g = predict(run_etl(folder), models)                                   # Block 4 + 5
    shutil.rmtree(folder, ignore_errors=True)                              # Block 7: privacy by design

    # add the TRUE glucose (only possible in simulation) for the chart and the report
    sim["time_utc"] = sim["time"].dt.tz_localize(TZ).dt.tz_convert("UTC")
    g = g.merge(sim[["time_utc", "bg_true", "carbs_g"]].rename(columns={"carbs_g": "carbs_eaten"}),
                on="time_utc", how="left")

    out = {"g": g, "twin": twin, "match": match,
           "quality": match_quality(age, weight, twin), "night": night_risk(g, models),
           "reg": regulatory_info(age, weight, seed, forgot_lunch, new_sensor, compression),
           "inputs": {"age": age, "weight_kg": weight, "meals": meals, "forgot_lunch": forgot_lunch,
                      "new_sensor": new_sensor, "compression": compression, "seed": seed},
           "temp_files_deleted": not folder.exists()}
    out.update(alarms_and_score(g, models))
    return out


# ============================================================================
# 7. BLOCK 6 - REPORT NUMBERS + CLARKE GRID FOR THIS PATIENT
# ============================================================================
def cgm_metrics(g: pd.DataFrame, win: np.ndarray) -> dict:
    """Standard CGM metrics of the shown 28 h (same definitions as the ETL's CGM report)."""
    x = g.loc[win, "cgm"].dropna()                          # only real sensor values
    mean = x.mean()
    return {"data %": 100 * len(x) / win.sum(),
            "mean glucose (mg/dL)": mean,
            "time in range 70-180 %": 100 * x.between(70, 180).mean(),
            "time below 70 %": 100 * (x < 70).mean(),
            "time below 54 %": 100 * (x < 54).mean(),
            "time above 180 %": 100 * (x > 180).mean(),
            "CV %": 100 * x.std() / mean,
            "GMI %": 3.31 + 0.02392 * mean}


def forecast_accuracy(g: pd.DataFrame, win: np.ndarray) -> dict:
    """How good was the 30-min forecast for THIS patient? (only really measured targets)"""
    ok = win & g["f30"].notna().to_numpy() & g["y_30"].notna().to_numpy() & g["y_30_measured"].to_numpy()
    y, p = g.loc[ok, "y_30"], g.loc[ok, "f30"]
    return {"points": int(ok.sum()),
            "RMSE model (mg/dL)": rmse(y, p),
            "RMSE 'stays the same' (mg/dL)": rmse(y, g.loc[ok, "cgm"]),
            "MARD %": mard(y, p)}                           # Clarke zones -> shown as a PLOT


def clarke_figure(g: pd.DataFrame, win: np.ndarray):
    """Clarke grid of this patient: true value in 30 min (x) vs forecast (y). LOW points in red."""
    ok = win & g["f30"].notna().to_numpy() & g["y_30"].notna().to_numpy() & g["y_30_measured"].to_numpy()
    y, p = g.loc[ok, "y_30"].to_numpy(), g.loc[ok, "f30"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    plot_clarke(ax, y, p, "This patient - 30 min ahead")           # zone borders + zone % in the title
    low = y < train.LOW                                              # true value below 70
    ax.scatter(y[~low], p[~low], s=14, color="#2a78d6", alpha=0.7, linewidths=0,   # few points ->
               zorder=3, label=f"other points ({(~low).sum()})")                   # bigger, darker dots
    ax.scatter(y[low], p[low], s=30, color="#e34948", edgecolors="white", linewidths=0.6,
               zorder=4, label=f"true value < 70 mg/dL ({low.sum()} points)")
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0, -0.12), ncol=2)
    fig.tight_layout()
    return fig


def report_numbers(r: dict) -> dict:
    """Everything the report needs, from the result of run_all."""
    _, _, win = shown_window(r["g"])
    return {"cgm": cgm_metrics(r["g"], win),
            "forecast": forecast_accuracy(r["g"], win),
            "clarke_fig": clarke_figure(r["g"], win)}


# ============================================================================
# 8. BLOCK 7 - REGULATORY ITEMS (GDPR, ICD-10, MDR traceability)
# ============================================================================
DIAGNOSIS = {"system": "ICD-10 (WHO)", "code": "E10", "text": "Type 1 diabetes mellitus"}

INTENDED_USE = ("Research demonstration. Forecasts glucose 30 minutes ahead and warns of a likely low "
                "(below 70 mg/dL) for people with type 1 diabetes using CGM. Advisory only: the user "
                "decides; no automatic insulin changes.")

DISCLAIMER = ("NOT A MEDICAL DEVICE. Simulated patient (UVA/Padova simulator, simglucose), no real "
              "personal data. Not CE-marked or FDA-cleared. Do not use for treatment decisions.")

LIMITATIONS = [
    "Simulated physiology: no exercise, stress, illness or changing insulin sensitivity.",
    "Twin = blend of 2 of 30 virtual patients, matched on age and weight only (not fitted to real CGM data).",
    "Forecast is biased high near lows (+13 mg/dL on test patients); the alarm threshold compensates.",
    "Alarm: about 3 false alarms per day in testing; near-lows and treated lows count as false.",
    "Night-risk model is weak: sensor-based labels are wrong on about 1 in 3 nights.",
]


def pseudonym(age: float, weight: float, seed: int) -> str:
    """Pseudonymous ID from the inputs (no name, not reversible without the inputs).
    In production: a KEYED hash (HMAC) with a secret key stored separately (GDPR Art. 4(5))."""
    digest = hashlib.sha256(f"{age}|{weight}|{seed}".encode()).hexdigest()
    return "TWIN-" + digest[:6].upper()


def file_fingerprint(path: Path) -> str:
    """Short SHA-256 of a file: proves WHICH model file produced the result."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def regulatory_info(age, weight, seed, forgot_lunch, new_sensor, compression) -> dict:
    """Everything the report needs for privacy, traceability and safe use."""
    training = json.loads(TRAIN_SUMMARY.read_text()) if TRAIN_SUMMARY.exists() else {}
    return {
        "patient_id": pseudonym(age, weight, seed),
        "diagnosis": DIAGNOSIS,
        "data_minimization": "only age, body weight and age group are used - no name, birth date or address",
        "traceability": {
            "pipeline_version": PIPELINE_VERSION,
            "model_file": MODEL_FILE.name,
            "model_sha256": file_fingerprint(MODEL_FILE),
            "model_trained_utc": training.get("created_utc", "unknown"),
            "etl_version": training.get("data", {}).get("etl_version", "unknown"),
        },
        "intended_use": INTENDED_USE,
        "disclaimer": DISCLAIMER,
        "limitations": LIMITATIONS,
    }


# ============================================================================
# 9. BLOCK 8 - THE PDF PATIENT REPORT
# ============================================================================
from datetime import timezone as _tz                                      # noqa: E402
from reportlab.lib import colors                                           # noqa: E402
from reportlab.lib.pagesizes import A4                                     # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet       # noqa: E402
from reportlab.lib.units import cm                                         # noqa: E402
from reportlab.platypus import (CondPageBreak, Image, ListFlowable, ListItem, PageBreak, Paragraph,  # noqa: E402
                                SimpleDocTemplate, Spacer, Table, TableStyle)

BLUE, ORANGE, GREY, VIOLET, RED, AMBER = "#2a78d6", "#eb6834", "#9a9893", "#6250d6", "#e34948", "#eda100"


def day_figure(r: dict):
    """Static chart of the shown 28 h (for the PDF): glucose + alarms, carbs eaten vs logged, insulin."""
    g = r["g"]
    t0, t1, win = shown_window(g)
    w = g[win]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True,
                                     gridspec_kw={"height_ratios": [3.2, 1, 1]})
    a1.axhspan(70, 180, color="#1baf7a", alpha=0.08, lw=0)                  # target range
    a1.axhspan(40, 70, color=RED, alpha=0.08, lw=0)                         # low zone
    wu = w[w["is_warmup"]]
    if len(wu):
        a1.axvspan(wu["time_local"].min(), wu["time_local"].max(), color="grey", alpha=0.15, lw=0,
                   label="sensor warm-up (removed)")
    a1.plot(w["time_local"], w["bg_true"], color=ORANGE, lw=1.2, ls="--", label="true glucose (simulator)")
    a1.plot(w["time_local"], w["cgm_raw"], color=GREY, lw=0.8, label="raw sensor")
    a1.plot(w["time_local"], w["cgm_clean"], color=BLUE, lw=1.8, label="cleaned CGM")
    a1.plot(w["time_local"] + pd.Timedelta(minutes=30), w["f30"], color=VIOLET, lw=1.0, ls=":",
            label="30-min forecast (drawn at the time it predicts)")
    comp = w[w["is_compression"]]
    if len(comp):
        a1.scatter(comp["time_local"], comp["cgm_raw"], s=18, facecolors="none", edgecolors="black",
                   lw=0.8, label="removed: compression low", zorder=4)
    for urgent, color, lab in [(True, RED, "alarm: low within 15 min"), (False, AMBER, "alarm: low within 30 min")]:
        idx = [a for a, u in zip(r["alarms"], r["urgent"]) if u == urgent and win[a]]
        if idx:
            a1.scatter(g["time_local"].iloc[idx], g["cgm"].iloc[idx], marker="v", s=60, color=color,
                       edgecolors="white", lw=0.8, zorder=5, label=lab)
    if len(r["onsets"]):
        a1.scatter(g["time_local"].iloc[r["onsets"]], [60] * len(r["onsets"]), marker="x", s=40,
                   color=RED, zorder=5, label="real low starts")
    t22 = t0.normalize() + pd.Timedelta(hours=22)
    a1.axvline(t22, color=VIOLET, lw=0.8, ls="--")
    top = max(300, np.nanmax(w[["bg_true", "cgm_raw"]].to_numpy()) + 20)
    a1.set_ylim(40, top)
    if r["night"]:
        a1.text(t22, top - 5, f" 22:00 night risk {r['night']['risk']:.0%}", color=VIOLET, fontsize=8, va="top")
    a1.set_ylabel("glucose (mg/dL)")
    a1.legend(fontsize=7, ncol=3, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.32))

    bar_w = pd.Timedelta(minutes=12)
    a2.bar(w["time_local"], w["carbs_eaten"], width=bar_w, color="#c9c7c1", label="carbs eaten")
    a2.bar(w["time_local"], w["carbs_g"], width=bar_w / 2, color=BLUE, label="carbs logged")
    a2.set_ylabel("carbs (g)")
    a2.legend(fontsize=7, frameon=False, loc="upper right")
    a3.bar(w["time_local"], w["bolus_u"], width=bar_w, color=VIOLET)
    a3.set_ylabel("bolus (U)")
    a3.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M", tz=TZ))
    for ax in (a1, a2, a3):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def fig_image(fig, width_cm: float) -> Image:
    """matplotlib figure -> picture for the PDF (keeps the aspect ratio)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    w_in, h_in = fig.get_size_inches()
    img = Image(buf, width=width_cm * cm, height=width_cm * cm * h_in / w_in)
    return img


CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=9.5)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Helvetica-Bold")


def table(rows: list, col_widths: list, header=True) -> Table:
    """A simple, clean table for the PDF. Every cell is a Paragraph -> long text wraps."""
    cells = [[Paragraph(str(v), HEAD if (header and i == 0) else CELL) for v in row]
             for i, row in enumerate(rows)]
    t = Table(cells, colWidths=[w * cm for w in col_widths], hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0cfca")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f5")))
    t.setStyle(TableStyle(style))
    return t


def make_pdf(r: dict, path: Path) -> Path:
    """Write the patient-day report as a PDF and return its path."""
    rep, reg, twin, inp = report_numbers(r), r["reg"], r["twin"], r["inputs"]
    ss = getSampleStyleSheet()
    h1, h2, body = ss["Title"], ss["Heading2"], ss["BodyText"]
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10)
    warn = ParagraphStyle("warn", parent=body, fontSize=8.5, textColor=colors.HexColor("#b3261e"),
                          backColor=colors.HexColor("#fdecea"), borderPadding=5, leading=11)
    fmt = lambda v, d=1: "-" if v is None else f"{v:.{d}f}"
    story = []

    # --- header ---
    story += [Paragraph("GlycoCast - patient day report", h1),
              Paragraph(f"<b>Patient {reg['patient_id']}</b> &nbsp;·&nbsp; {reg['diagnosis']['code']} "
                        f"{reg['diagnosis']['text']} ({reg['diagnosis']['system']}) &nbsp;·&nbsp; "
                        f"shown period {r['t0']:%d %b %H:%M} - {r['t1']:%d %b %H:%M} &nbsp;·&nbsp; "
                        f"created {datetime.now(_tz.utc):%Y-%m-%d %H:%M} UTC", small),
              Spacer(1, 6), Paragraph(reg["disclaimer"], warn), Spacer(1, 8)]

    # --- 1. summary ---
    n = r["night"] or {}
    story += [Paragraph("1. Summary", h2), table([
        ["Real lows", "Warned in time", "Median warning", "False alarms", "Night risk at 22:00"],
        [str(r["lows"]), f"{r['caught']} of {r['lows']}",
         f"{fmt(r['lead_min'], 0)} min" if r["lead_min"] else "-",
         f"{r['false_alarms']} (of which {r['false_but_treated']} treated by a hypo snack)",
         f"{n.get('risk', 0):.0%} -> {'snack advised' if n.get('snack') else 'no snack'}"
         f" (night low: {'yes' if n.get('low_happened') else 'no'})"]],
        [2.3, 2.6, 2.6, 4.6, 5.4])]

    # --- 2. patient and twin ---
    m = r["match"]
    story += [Paragraph("2. Patient and virtual twin", h2),
              Paragraph(f"Real patient: <b>{inp['age']} years, {inp['weight_kg']} kg</b> -> age group "
                        f"<b>{twin['group']}</b>. Twin = blend of the 2 most similar virtual patients. "
                        f"Match quality: <b>{r['quality']}</b>.", body),
              table([["", "age (y)", "weight (kg)", "daily insulin (U)", "carb ratio (g/U)",
                      "correction (mg/dL/U)", "blend weight"]] +
                    [[d.donor, d.age, d.weight_kg, d.daily_insulin_u, "", "", d.blend_weight] for d in m.itertuples()] +
                    [["twin", fmt(twin["Age"], 0), fmt(twin["BW"]), fmt(twin["TDI"]), fmt(twin["CR"]),
                      fmt(twin["CF"]), "1.00"]],
                    [2.8, 1.5, 2.0, 2.6, 2.6, 3.0, 2.0])]
    meals = ", ".join(f"{int(h)}:{int(round(h % 1 * 60)):02d} {g} g" for h, g in inp["meals"])
    problems = [k for k, on in [("forgot to log lunch", inp["forgot_lunch"]),
                                ("new sensor at 08:00", inp["new_sensor"]),
                                ("compression low at 03:00", inp["compression"])] if on]
    story += [Spacer(1, 4), Paragraph(f"Day design: meals {meals}; real-life problems: "
                                      f"{', '.join(problems) or 'none'}; sensor noise seed {inp['seed']}.", small)]

    # --- 3. the day ---
    story += [Paragraph("3. The day", h2), fig_image(day_figure(r), 17.5), PageBreak()]

    # --- 4. CGM metrics ---
    c = rep["cgm"]
    targets = {"data %": ">= 70", "time in range 70-180 %": "> 70", "time below 70 %": "< 4",
               "time below 54 %": "< 1", "time above 180 %": "< 25", "CV %": "<= 36"}
    story += [Paragraph("4. CGM metrics (international consensus definitions)", h2),
              table([["metric", "value", "consensus target"]] +
                    [[k, fmt(v), targets.get(k, "")] for k, v in c.items()], [6, 2.5, 3.5])]

    # --- 5. forecast accuracy + Clarke plot ---
    f = rep["forecast"]
    err_model, err_same = fmt(f["RMSE model (mg/dL)"]), fmt(f["RMSE 'stays the same' (mg/dL)"])
    story += [Paragraph("5. 30-minute forecast for this patient", h2),
              Paragraph(f"Error of the model <b>{err_model} mg/dL</b> (RMSE) vs "
                        f"{err_same} mg/dL for 'glucose stays the same'; "
                        f"MARD {fmt(f['MARD %'])} % on {f['points']} points. "
                        "Clarke error grid below - red points: the true value was below 70 mg/dL.", body),
              fig_image(rep["clarke_fig"], 9.5)]

    # --- 6. alarms ---
    story += [Paragraph("6. Alarms in the shown period", h2),
              Paragraph(f"Rule (chosen on training patients): alarm if the 30-min forecast is below "
                        f"{r['thr']:.0f} mg/dL; no alarm while already low; 30-min snooze.", small)]
    at = r["alarm_table"]
    if len(at):
        heads = ["time", "glucose (mg/dL)", "forecast in 30 min", "urgency",
                 "lowest glucose in next 30 min", "what happened"]
        story.append(table([heads] + at.astype(str).values.tolist(), [2.3, 1.8, 2.0, 3.2, 2.6, 5.5]))
    else:
        story.append(Paragraph("No alarms.", body))

    # --- 7. regulatory and traceability ---
    tr = reg["traceability"]
    story += [CondPageBreak(8 * cm),                      # don't leave the heading alone at a page bottom
              Paragraph("7. Intended use, privacy and traceability", h2),
              Paragraph(f"<b>Intended use:</b> {reg['intended_use']}", body),
              Paragraph(f"<b>Data minimization (GDPR):</b> {reg['data_minimization']}. Patient shown only "
                        f"by a pseudonymous ID. Temporary device files deleted after processing: "
                        f"{'yes' if r['temp_files_deleted'] else 'no'}.", body),
              table([["pipeline", "model file", "model SHA-256", "model trained (UTC)", "ETL version"],
                     [tr["pipeline_version"], tr["model_file"], tr["model_sha256"],
                      tr["model_trained_utc"], tr["etl_version"]]], [2.2, 3.6, 3.0, 4.6, 2.2]),
              Paragraph("<b>Known limitations</b>", body),
              ListFlowable([ListItem(Paragraph(x, small)) for x in reg["limitations"]], bulletType="bullet")]

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                      topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                      title=f"GlycoCast report {reg['patient_id']}").build(story)
    return path


# ============================================================================
# 10. QUICK TEST: python -W ignore app/pipeline.py
# ============================================================================
if __name__ == "__main__":
    import time

    # Block 1
    for age, weight in [(35, 80), (10, 30), (16, 55), (45, 140)]:
        twin, match = find_twin(age, weight)
        print(f"\nReal patient: {age} y, {weight} kg  ->  group: {twin['group']}")
        print(match.to_string(index=False))
        print(f"twin: {twin['Age']:.0f} y, {twin['BW']:.1f} kg, {twin['TDI']:.1f} U/day, "
              f"carb ratio {twin['CR']:.1f} g/U, correction {twin['CF']:.1f} mg/dL/U"
              f"  | match: {match_quality(age, weight, twin)}")

    # Block 2
    print("\n=== Block 2: simulate the twin of a 10-year-old, 30 kg ===")
    twin, _ = find_twin(10, 30)
    my_meals = [(7.5, 50), (12.5, 70), (18.5, 80)]                    # (hour, grams) for day 1
    t = time.time()
    sim = simulate_twin(twin, my_meals, seed=1)
    print(f"simulated {len(sim)} rows (3 days x 288) in {time.time() - t:.0f} s")
    sim["day"] = (sim["time"] - START).dt.days
    print(sim.groupby("day").agg(mean_bg=("bg_true", "mean"), min_bg=("bg_true", "min"),
                                 max_bg=("bg_true", "max"), carbs_g=("carbs_g", "sum"),
                                 insulin_u=("insulin_u", "sum"),
                                 hypo_snacks=("rescue_g", lambda x: (x > 0).sum())).round(0))

    # Block 3
    print("\n=== Block 3: messy device files (all three problems ON) ===")
    folder = write_device_files(sim, "NEW01", twin["group"], forgot_lunch=True,
                                new_sensor=True, compression=True, seed=1)
    print("folder:", folder)
    cgm = pd.read_csv(folder / "cgm.csv")
    lb = pd.read_csv(folder / "logbook.csv")
    print(f"cgm.csv: {len(cgm)} rows (simulation: {len(sim)}), sensors: {cgm['sensor_id'].unique().tolist()}")
    print(cgm.head(3).to_string(index=False))
    print(f"\nlogbook.csv: {len(lb)} rows")
    day1 = lb[lb["timestamp_local"].str.startswith("2026-01-06")]
    print("shown day (6 Jan) in the logbook - is lunch missing?")
    print(day1.to_string(index=False))

    # Block 4
    print("\n=== Block 4: our ETL + features on the new files ===")
    g = run_etl(folder)
    print(f"clean table: {len(g)} rows x {g.shape[1]} columns")
    print(f"  warm-up removed : {g['is_warmup'].sum()} steps (sensor 1 at start + sensor 2 at 08:00)")
    comp = g[g["is_compression"]]
    print(f"  compression     : {len(comp)} steps found, at "
          f"{comp['time_local'].min():%d %b %H:%M} - {comp['time_local'].max():%H:%M}")
    print(f"  still missing   : {g['cgm'].isna().sum()} steps")
    print("\nLunch of the shown day: carbs NOT logged, but the bolus is there:")
    lunch = g[(g["time_local"].dt.day == 6) & g["time_local"].dt.hour.between(12, 13)]
    cols = ["time_local", "cgm", "carbs_g", "bolus_u", "cob_g", "iob_u", "roc_kf"]
    print(lunch[cols].iloc[5:12].round(1).to_string(index=False))

    # Block 5
    print("\n=== Block 5: the whole chain with ONE call (run_all) ===")
    models = load_models()
    t = time.time()
    r = run_all(10, 30, my_meals, forgot_lunch=True, new_sensor=True, compression=True,
                seed=1, models=models)
    print(f"done in {time.time() - t:.0f} s")
    g = r["g"]
    print(f"shown window: {r['t0']:%d %b %H:%M} -> {r['t1']:%d %b %H:%M}")
    print(f"real lows: {r['lows']} | warned in time: {r['caught']} | median warning: {r['lead_min']} min"
          f" | false alarms: {r['false_alarms']}")
    print(f"   (of the false alarms, {r['false_but_treated']} were followed by a logged hypo snack)")
    print("\nalarm table (for the report):")
    print(r["alarm_table"].to_string(index=False))
    n = r["night"]
    print(f"night risk at 22:00: {n['risk']:.0%} -> {'snack advised' if n['snack'] else 'no snack'}"
          f" | night low happened: {n['low_happened']}")

    # Block 6
    print("\n=== Block 6: report numbers for this patient ===")
    rep = report_numbers(r)
    print("CGM metrics (shown 28 h):")
    print(pd.Series(rep["cgm"]).round(1).to_string())
    print("\n30-min forecast for this patient:")
    print(pd.Series(rep["forecast"]).round(1).to_string())
    out = Path(tempfile.gettempdir()) / "glycocast_twin_clarke.png"
    rep["clarke_fig"].savefig(out, dpi=130)
    print("\nClarke grid saved ->", out)

    # Block 7
    print("\n=== Block 7: regulatory items ===")
    reg = r["reg"]
    print("patient ID :", reg["patient_id"], "(pseudonymous)")
    print("diagnosis  :", reg["diagnosis"]["code"], "-", reg["diagnosis"]["text"], f"({reg['diagnosis']['system']})")
    print("data used  :", reg["data_minimization"])
    print("traceability:", json.dumps(reg["traceability"], indent=2))
    print("temp device files deleted after ETL:", r["temp_files_deleted"])
    print("disclaimer :", reg["disclaimer"])

    # Block 8
    print("\n=== Block 8: PDF report ===")
    pdf = make_pdf(r, Path("reports") / f"patient_report_{reg['patient_id']}.pdf")
    print("PDF saved ->", pdf)