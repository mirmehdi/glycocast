"""
etl.py - Step 3: turn the messy device files into one clean training table.

Block 1: Extract and standardize
    - load the 4 raw files
    - bring all times to ONE standard: UTC
    - bring all glucose values to ONE unit: mg/dL
    - remove exact duplicate rows

Block 2: Clean the CGM signal
    - put every value on a fixed 5-min grid (missing slots become visible)
    - remove sensor warm-up (first 2 h of each sensor)
    - detect and remove compression lows (fake night lows)
    - fill only short gaps (<= 15 min)

Block 3: Causal Kalman filter (model inputs)
    - 2 states: glucose and speed; uses only the past
    - follows faster for 90 min after a logged meal or bolus

Block 4: Events, features and targets
    - put boluses, carbs and hypo treatments on the 5-min grid
    - insulin on board / carbs on board (exponential decay)
    - time of day, age group, missing mask
    - targets: glucose in 15/30/60/120 min, "low within 30 min"

Block 5: Load and check
    - save ONE versioned training table (Parquet) + a metadata file
    - CGM report per patient (time in range, CV, GMI)
    - QC plot: raw vs cleaned vs true glucose for one patient

Run:    python src/etl.py
"""

# ----------------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------------
import hashlib                         # file fingerprints (traceability)
import json
import subprocess                      # read the current git commit
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # draw to a file, no window needed
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.signal import lfilter      # fast "running" calculation for insulin/carbs on board

pd.set_option("display.width", 120)

# ----------------------------------------------------------------------------
# 2. Settings
# ----------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
HOME_TZ = "Europe/Vienna"          # the patients' home time zone (known from the user profile)
MGDL_PER_MMOL = 18.0               # 1 mmol/L = 18 mg/dL
TRUTH_DIR = Path("data/truth")     # answer key - ONLY used to check our cleaning, never as input

STEP = "5min"                      # CGM grid
WARMUP_STEPS = 24                  # 2 h = 24 x 5 min
MAX_FILL_STEPS = 3                 # fill gaps up to 15 min = 3 steps

# compression low rule
NIGHT_HOURS = range(0, 6)          # 00:00-05:59 local time
COMP_DROP = 30                     # fall of at least 30 mg/dL ...
COMP_DROP_STEPS = 3                # ... within 15 min
COMP_RECOVER_STEPS = 12            # and back near the start level within 60 min
COMP_RECOVER_TOL = 10              # "near" = within 10 mg/dL

# Kalman filter
DT = 5.0                           # minutes per step
KF_R = 10.0 ** 2                   # sensor noise: about +-10 mg/dL  -> variance 100
KF_ACC = 0.1                       # normal "speed change" of glucose (mg/dL per min^2), chosen on training patients
KF_ACC_EVENT = 0.3                 # 3x larger after a meal or bolus (body changes faster)
KF_EVENT_STEPS = 18                # boost lasts 90 min = 18 steps
KF_MAX_PREDICT = 6                 # predict through gaps up to 30 min, then "unknown"

# Features and targets
TAU_INSULIN_MIN = 60               # insulin on board fades with time constant 60 min (~4 h action)
TAU_CARBS_MIN = 40                 # carbs on board fade with 40 min (~2.5 h absorption)
TAU_HYPO_MIN = 15                  # fast hypo carbs act within ~45 min
HORIZONS_MIN = [15, 30, 60, 120]   # forecast horizons
LOW_MGDL = 70                      # hypoglycemia threshold
ALARM_HORIZON_MIN = 30             # alarm: "low within the next 30 min"

# Load
ETL_VERSION = "v1"                 # change this when the cleaning logic changes
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
KEEP_COLUMNS = [
    "patient_id", "age_group", "time_utc", "time_local", "sensor_id",
    "cgm_raw", "cgm", "cgm_clean", "cgm_missing", "is_warmup", "is_compression",
    "cgm_kf", "roc_kf", "kf_sd",
    "bolus_u", "carbs_g", "hypo_g", "basal_u_h", "iob_u", "cob_g", "hob_g",
    "tod_sin", "tod_cos",
    "y_15", "y_30", "y_60", "y_120",
    "y_15_measured", "y_30_measured", "y_60_measured", "y_120_measured", "low_30",
]


# ----------------------------------------------------------------------------
# 3. Extract: one function per file
# ----------------------------------------------------------------------------
def load_cgm() -> pd.DataFrame:
    """CGM file: times are already UTC (they end with 'Z'). Remove duplicates."""
    df = pd.read_csv(RAW_DIR / "cgm.csv")
    n_before = len(df)
    df = df.drop_duplicates()                                    # exact copies (sync retries)
    print(f"cgm        : {n_before} rows -> {len(df)} after removing {n_before - len(df)} duplicates")

    df["time_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)   # text -> time (UTC)
    df = df.rename(columns={"glucose_mg_dl": "glucose"})
    return df[["patient_id", "sensor_id", "time_utc", "glucose"]]


def load_logbook() -> pd.DataFrame:
    """Logbook: local time WITH offset (+0100). Convert to UTC."""
    df = pd.read_csv(RAW_DIR / "logbook.csv")
    df = df.drop_duplicates()

    # the offset tells pandas exactly how far from UTC the time is -> no guessing
    df["time_utc"] = pd.to_datetime(df["timestamp_local"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
    print(f"logbook    : {len(df)} rows, events: {df['event'].value_counts().to_dict()}")
    return df[["patient_id", "time_utc", "event", "value", "unit", "source"]]


def load_fingerstick() -> pd.DataFrame:
    """Meter: German date format, NO time zone, mmol/L. Convert to UTC and mg/dL."""
    df = pd.read_csv(RAW_DIR / "fingerstick.csv")
    df = df.drop_duplicates()

    # 1) text -> time, using the German format day.month.year hour:minute
    t_local = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M")

    # 2) the meter has no time zone -> we ASSUME the home time zone, then convert to UTC.
    #    ambiguous / nonexistent: safety for the two DST switch nights per year
    df["time_utc"] = (t_local.dt.tz_localize(HOME_TZ, ambiguous="NaT", nonexistent="shift_forward")
                             .dt.tz_convert("UTC"))

    # 3) units: convert only rows that are in mmol/L (other meters may already send mg/dL)
    is_mmol = df["unit"] == "mmol/L"
    df["glucose"] = df["value"].where(~is_mmol, df["value"] * MGDL_PER_MMOL)
    print(f"fingerstick: {len(df)} rows, {is_mmol.sum()} converted from mmol/L to mg/dL")
    return df[["patient_id", "time_utc", "glucose"]]


def load_patients() -> pd.DataFrame:
    """Patient table: no times, no units -> just load."""
    df = pd.read_csv(RAW_DIR / "patients.csv")
    print(f"patients   : {len(df)} rows, groups: {df['age_group'].value_counts().to_dict()}")
    return df


# ----------------------------------------------------------------------------
# 4. Transform (Block 2): clean the CGM signal
# ----------------------------------------------------------------------------
def to_grid(cgm: pd.DataFrame) -> pd.DataFrame:
    """Snap every value to the nearest 5-min slot and build a complete grid per patient."""
    df = cgm.copy()
    df["time_utc"] = df["time_utc"].dt.round(STEP)          # 07:00:37 -> 07:00:00

    # two values in the same slot -> average them (keep the sensor id)
    df = (df.groupby(["patient_id", "time_utc"], as_index=False)
            .agg(glucose=("glucose", "mean"), sensor_id=("sensor_id", "first")))

    # complete grid: one row for EVERY 5 min, also where no value arrived (-> NaN)
    grids = []
    for pid, p in df.groupby("patient_id"):
        full = pd.date_range(p["time_utc"].min(), p["time_utc"].max(), freq=STEP)
        g = p.set_index("time_utc").reindex(full)            # missing slots get NaN
        g.index.name = "time_utc"
        g["patient_id"] = pid
        g["sensor_id"] = g["sensor_id"].ffill()               # a gap belongs to the sensor before it
        grids.append(g.reset_index())
    out = pd.concat(grids, ignore_index=True)

    out["time_local"] = out["time_utc"].dt.tz_convert(HOME_TZ)   # needed for "night", "time of day"
    out = out.rename(columns={"glucose": "cgm_raw"})
    return out[["patient_id", "time_utc", "time_local", "sensor_id", "cgm_raw"]]


def remove_warmup(g: pd.DataFrame) -> pd.DataFrame:
    """Set the first 2 h of every sensor to NaN (the sensor is not yet accurate)."""
    step_in_sensor = g.groupby(["patient_id", "sensor_id"]).cumcount()   # 0, 1, 2, ... per sensor
    g["is_warmup"] = step_in_sensor < WARMUP_STEPS
    g["cgm"] = g["cgm_raw"].where(~g["is_warmup"])                       # working copy
    return g


def find_compression(x: np.ndarray, night: np.ndarray) -> np.ndarray:
    """Return True where a compression low is found in ONE patient's signal.

    Rule: at night, glucose falls >= 30 mg/dL within 15 min,
          and comes back to within 10 mg/dL of the start within 60 min.
    """
    n = len(x)
    flag = np.zeros(n, dtype=bool)
    for i in range(COMP_DROP_STEPS, n):
        if not night[i]:
            continue
        start, now = x[i - COMP_DROP_STEPS], x[i]
        if np.isnan(start) or np.isnan(now) or start - now < COMP_DROP:
            continue                                         # no fast fall here
        # fast fall found -> does it come back fast?
        for j in range(i + 1, min(i + 1 + COMP_RECOVER_STEPS, n)):
            if not np.isnan(x[j]) and x[j] >= start - COMP_RECOVER_TOL:
                flag[i - COMP_DROP_STEPS + 1:j] = True       # everything between start and recovery
                break
    return flag


def remove_compression(g: pd.DataFrame) -> pd.DataFrame:
    """Detect compression lows per patient and set them to NaN."""
    g["is_compression"] = False
    for pid, idx in g.groupby("patient_id").groups.items():
        x = g.loc[idx, "cgm"].to_numpy(dtype=float)
        night = g.loc[idx, "time_local"].dt.hour.isin(NIGHT_HOURS).to_numpy()
        g.loc[idx, "is_compression"] = find_compression(x, night)
    g["cgm"] = g["cgm"].where(~g["is_compression"])
    return g


def fill_short_gaps(g: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps of <= 15 min by a straight line. Longer gaps stay NaN."""
    def fill_one(s: pd.Series) -> pd.Series:
        missing = s.isna()
        run_id = (missing != missing.shift()).cumsum()                 # number each run of NaN / non-NaN
        run_len = missing.groupby(run_id).transform("sum")              # length of each NaN run
        short = missing & (run_len <= MAX_FILL_STEPS)
        line = s.interpolate(limit_area="inside")                       # straight line between neighbours
        return s.where(~short, line)

    g["cgm_clean"] = g.groupby("patient_id")["cgm"].transform(fill_one)
    return g


def check_against_truth(g: pd.DataFrame) -> None:
    """Compare our cleaning with the answer key from export_devices.py."""
    truth = pd.read_parquet(TRUTH_DIR / "cgm_flags.parquet")
    truth["time_utc"] = truth["time"].dt.tz_localize(HOME_TZ).dt.tz_convert("UTC")
    m = g.merge(truth, on=["patient_id", "time_utc"], how="inner")

    tp = (m["is_compression"] & m["compression"]).sum()
    found, real = m["is_compression"].sum(), (m["compression"] & m["reported"]).sum()
    print(f"  compression: found {found} steps, real {real} -> "
          f"precision {tp / max(found, 1):.0%}, recall {tp / max(real, 1):.0%}")
    print(f"  warm-up    : removed {m['is_warmup'].sum()} steps, real {m['warmup'].sum()}")


# ----------------------------------------------------------------------------
# 5. Transform (Block 3): causal Kalman filter
# ----------------------------------------------------------------------------
F = np.array([[1.0, DT],            # new glucose = glucose + 5 min x speed
              [0.0, 1.0]])          # new speed   = speed
H = np.array([[1.0, 0.0]])          # the sensor sees glucose only, not speed


def process_noise(acc: float) -> np.ndarray:
    """Q: how much glucose and speed can change in one step, from a random 'acceleration'."""
    return acc ** 2 * np.array([[DT ** 4 / 4, DT ** 3 / 2],
                                [DT ** 3 / 2, DT ** 2]])


Q_NORMAL, Q_EVENT = process_noise(KF_ACC), process_noise(KF_ACC_EVENT)


def kalman_one(z: np.ndarray, boost: np.ndarray):
    """Run the filter over ONE patient. z = measurements (NaN = missing).

    Returns glucose estimate, speed estimate, uncertainty (SD) and NIS per step.
    Causal: the value at step t uses only measurements up to t.
    """
    n = len(z)
    est, speed, sd, nis = (np.full(n, np.nan) for _ in range(4))
    x, P = None, None
    since_meas = 0                                     # steps since the last real measurement

    for t in range(n):
        if x is None:                                  # start: wait for the first measurement
            if np.isnan(z[t]):
                continue
            x = np.array([z[t], 0.0])                  # glucose = first value, speed = 0
            P = np.diag([KF_R, 1.0])
        else:
            # ---- PREDICT ----
            Q = Q_EVENT if boost[t] else Q_NORMAL
            x = F @ x
            P = F @ P @ F.T + Q

            # ---- UPDATE (only if a measurement exists) ----
            if np.isnan(z[t]):
                since_meas += 1
            else:
                since_meas = 0
                y = z[t] - (H @ x)[0]                  # innovation: measured - predicted
                S = (H @ P @ H.T)[0, 0] + KF_R         # its expected variance
                K = (P @ H.T)[:, 0] / S                # Kalman gain (2 numbers)
                x = x + K * y
                P = (np.eye(2) - np.outer(K, H[0])) @ P
                nis[t] = y ** 2 / S                    # normalized innovation squared

        if since_meas <= KF_MAX_PREDICT:               # too long without data -> unknown
            est[t], speed[t], sd[t] = x[0], x[1], np.sqrt(P[0, 0])

    return est, speed, sd, nis


def add_kalman(g: pd.DataFrame, logbook: pd.DataFrame) -> pd.DataFrame:
    """Add cgm_kf (glucose), roc_kf (speed, mg/dL per min), kf_sd (uncertainty)."""
    # 1) mark the grid slots where a meal or bolus was LOGGED (known at that time -> causal)
    ev = logbook[logbook["event"].isin(["carbs", "bolus"])].copy()
    ev["time_utc"] = ev["time_utc"].dt.floor(STEP)
    ev = ev[["patient_id", "time_utc"]].drop_duplicates().assign(event=1)
    g = g.merge(ev, on=["patient_id", "time_utc"], how="left")
    g["event"] = g["event"].fillna(0)

    # 2) boost = an event happened in the last 90 min (rolling window looks only backwards)
    g["kf_boost"] = (g.groupby("patient_id")["event"]
                       .transform(lambda s: s.rolling(KF_EVENT_STEPS, min_periods=1).max()) > 0)

    # 3) run the filter per patient on `cgm` (warm-up + compression removed, gaps NOT filled)
    for col in ["cgm_kf", "roc_kf", "kf_sd", "kf_nis"]:
        g[col] = np.nan
    for pid, idx in g.groupby("patient_id").groups.items():
        est, speed, sd, nis = kalman_one(g.loc[idx, "cgm"].to_numpy(float),
                                         g.loc[idx, "kf_boost"].to_numpy())
        g.loc[idx, "cgm_kf"], g.loc[idx, "roc_kf"] = est, speed
        g.loc[idx, "kf_sd"], g.loc[idx, "kf_nis"] = sd, nis
    return g.drop(columns="event")


def check_kalman(g: pd.DataFrame) -> None:
    """Is the filtered signal closer to the true glucose than the raw sensor? (answer key)"""
    sim = pd.read_parquet(Path("data/sim/clean.parquet"))
    sim["time_utc"] = sim["time"].dt.tz_localize(HOME_TZ).dt.tz_convert("UTC")
    m = g.merge(sim[["patient_id", "time_utc", "bg_true"]], on=["patient_id", "time_utc"])
    ok = m["cgm"].notna() & m["cgm_kf"].notna()
    mae_raw = (m.loc[ok, "cgm"] - m.loc[ok, "bg_true"]).abs().mean()
    mae_kf = (m.loc[ok, "cgm_kf"] - m.loc[ok, "bg_true"]).abs().mean()
    print(f"  error vs true glucose: raw CGM {mae_raw:.1f} mg/dL -> Kalman {mae_kf:.1f} mg/dL")
    print(f"  mean NIS: {m['kf_nis'].mean():.2f}  (about 1 = R and Q fit the data)")
    print(f"  estimate available: {m['cgm_kf'].notna().mean():.1%} of all slots")


# ----------------------------------------------------------------------------
# 6. Transform (Block 4): events, features and targets
# ----------------------------------------------------------------------------
def add_events(g: pd.DataFrame, logbook: pd.DataFrame) -> pd.DataFrame:
    """Put logged boluses, carbs and hypo treatments on the 5-min grid (sum per slot)."""
    ev = logbook[logbook["event"].isin(["bolus", "carbs", "hypo_treatment"])].copy()
    ev["time_utc"] = ev["time_utc"].dt.floor(STEP)              # 08:40:19 -> 08:40 slot
    wide = (ev.pivot_table(index=["patient_id", "time_utc"], columns="event",
                           values="value", aggfunc="sum")
              .rename(columns={"bolus": "bolus_u", "carbs": "carbs_g", "hypo_treatment": "hypo_g"})
              .reset_index())
    g = g.merge(wide, on=["patient_id", "time_utc"], how="left")
    g[["bolus_u", "carbs_g", "hypo_g"]] = g[["bolus_u", "carbs_g", "hypo_g"]].fillna(0)

    # basal rate: one entry per patient -> same value on every row of that patient
    basal = logbook[logbook["event"] == "basal_rate"].set_index("patient_id")["value"]
    g["basal_u_h"] = g["patient_id"].map(basal)
    return g


def on_board(x: np.ndarray, tau_min: float) -> np.ndarray:
    """Amount still active: every step the old amount fades by exp(-5/tau), new amount is added.
    on_board[t] = on_board[t-1] * a + x[t]      (uses only the past -> causal)
    """
    a = np.exp(-DT / tau_min)
    return lfilter([1.0], [1.0, -a], x)


def add_features(g: pd.DataFrame, patients: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. Every feature at time t uses only information known at time t."""
    by = g.groupby("patient_id")

    # insulin / carbs / hypo carbs still active in the body
    g["iob_u"] = by["bolus_u"].transform(lambda s: on_board(s.to_numpy(), TAU_INSULIN_MIN))
    g["cob_g"] = by["carbs_g"].transform(lambda s: on_board(s.to_numpy(), TAU_CARBS_MIN))
    g["hob_g"] = by["hypo_g"].transform(lambda s: on_board(s.to_numpy(), TAU_HYPO_MIN))

    # time of day as a circle: 23:55 and 00:05 are close (sin/cos instead of 0-24)
    hour = g["time_local"].dt.hour + g["time_local"].dt.minute / 60
    g["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    g["tod_cos"] = np.cos(2 * np.pi * hour / 24)

    # missing mask: 1 where the sensor value is not available (gap, warm-up, compression)
    g["cgm_missing"] = g["cgm"].isna().astype(int)

    # patient info (known from the user profile)
    g = g.merge(patients[["patient_id", "age_group"]], on="patient_id", how="left")
    return g


def add_targets(g: pd.DataFrame) -> pd.DataFrame:
    """The 'correct answers'. Targets MAY look into the future - that is their job."""
    by = g.groupby("patient_id")
    for h in HORIZONS_MIN:
        k = h // 5
        g[f"y_{h}"] = by["cgm_clean"].shift(-k)                       # cleaned CGM in h minutes
        g[f"y_{h}_measured"] = by["cgm"].shift(-k).notna()            # was it really measured?

    # alarm target: is the cleaned CGM below 70 at ANY step in the next 30 min?
    k = ALARM_HORIZON_MIN // 5
    future_min = by["cgm_clean"].transform(
        lambda s: s[::-1].rolling(k, min_periods=1).min()[::-1].shift(-1))   # min of t+5 ... t+30
    g["low_30"] = (future_min < LOW_MGDL).astype(float)
    g.loc[future_min.isna(), "low_30"] = np.nan                       # no data ahead -> unknown
    return g


# ----------------------------------------------------------------------------
# 7. Load (Block 5): save, report, plot
# ----------------------------------------------------------------------------
def file_hash(path: Path) -> str:
    """Short fingerprint of a file: if one byte changes, the hash changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def save_table(g: pd.DataFrame) -> Path:
    """Save the training table + a metadata file that says HOW it was made."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"train_{ETL_VERSION}.parquet"
    g[KEEP_COLUMNS].to_parquet(out, index=False)

    meta = {
        "etl_version": ETL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "rows": len(g), "patients": int(g["patient_id"].nunique()),
        "input_files": {f.name: file_hash(f) for f in sorted(RAW_DIR.glob("*.csv"))},
        "settings": {"warmup_steps": WARMUP_STEPS, "max_fill_steps": MAX_FILL_STEPS,
                     "comp_drop": COMP_DROP, "kf_r": KF_R, "kf_acc": KF_ACC,
                     "kf_acc_event": KF_ACC_EVENT, "tau_insulin": TAU_INSULIN_MIN,
                     "tau_carbs": TAU_CARBS_MIN, "horizons": HORIZONS_MIN},
    }
    (OUT_DIR / f"train_{ETL_VERSION}.json").write_text(json.dumps(meta, indent=2))
    return out


def cgm_report(g: pd.DataFrame) -> pd.DataFrame:
    """Standard CGM metrics per patient (international consensus), on measured values only."""
    def metrics(p: pd.DataFrame) -> pd.Series:
        x = p["cgm"].dropna()                                  # only real measurements
        mean = x.mean()
        return pd.Series({
            "days": len(p) / 288,
            "data_pct": 100 * len(x) / len(p),                 # target: >= 70 %
            "mean_mgdl": mean,
            "tir_pct": 100 * x.between(70, 180).mean(),        # time in range, target: > 70 %
            "tbr70_pct": 100 * (x < 70).mean(),                # time below 70, target: < 4 %
            "tbr54_pct": 100 * (x < 54).mean(),                # time below 54, target: < 1 %
            "tar180_pct": 100 * (x > 180).mean(),              # time above 180, target: < 25 %
            "cv_pct": 100 * x.std() / mean,                    # variability, target: <= 36 %
            "gmi_pct": 3.31 + 0.02392 * mean,                  # estimated HbA1c from mean glucose
        })

    rep = g.groupby(["patient_id", "age_group"]).apply(metrics).round(1).reset_index()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rep.to_csv(REPORT_DIR / "cgm_report.csv", index=False)
    return rep


def qc_plot(g: pd.DataFrame, patient: str = "P01") -> Path:
    """One night + morning of one patient: raw sensor vs cleaned vs true glucose."""
    p = g[g["patient_id"] == patient].copy()
    day = p.loc[p["is_compression"], "time_local"].dt.normalize().iloc[0]   # first night with a compression low
    w = p[(p["time_local"] >= day) & (p["time_local"] < day + pd.Timedelta(hours=14))]

    sim = pd.read_parquet(Path("data/sim/clean.parquet"))                    # answer key, only for this plot
    sim = sim[sim["patient_id"] == patient]
    sim["time_local"] = sim["time"].dt.tz_localize(HOME_TZ)
    w = w.merge(sim[["time_local", "bg_true"]], on="time_local", how="left")

    ink, muted, blue, orange = "#0b0b0b", "#8a8984", "#2a78d6", "#eb6834"
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(70, 180, color="#f1f0ec", zorder=0)                          # target range band
    ax.plot(w["time_local"], w["cgm_raw"], color=muted, lw=1, label="Raw sensor (as received)")
    ax.plot(w["time_local"], w["cgm_clean"], color=blue, lw=2, label="Cleaned CGM (target)")
    ax.plot(w["time_local"], w["bg_true"], color=orange, lw=1.5, ls="--", label="True glucose (simulator)")
    rem = w[w["is_compression"]]
    ax.scatter(rem["time_local"], rem["cgm_raw"], s=36, facecolors="none", edgecolors=ink,
               lw=1, zorder=3, label="Removed: compression low")
    for _, r in w[w["carbs_g"] > 0].iterrows():                              # logged meals
        ax.axvline(r["time_local"], color=muted, lw=0.8, ls=":")
        ax.text(r["time_local"], ax.get_ylim()[1], f" {r['carbs_g']:.0f} g", va="top",
                fontsize=9, color=ink)

    ax.set_title(f"{patient}: compression low at night and breakfast - raw vs cleaned vs true",
                 loc="left", fontsize=12, color=ink)
    ax.set_ylabel("Glucose (mg/dL)", color=ink)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M", tz=HOME_TZ))
    ax.tick_params(colors=muted)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(muted)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()

    out = REPORT_DIR / "figures" / f"qc_{patient}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# 8. Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Block 1: extract and standardize ===")
    cgm = load_cgm()
    logbook = load_logbook()
    fs = load_fingerstick()
    patients = load_patients()

    # checks: all times are now UTC, all glucose in mg/dL
    print("\nFirst row of each table (all times now in UTC):")
    print(cgm.head(1))
    print(logbook.head(1))
    print(fs.head(1))
    print("\nTime range:", cgm["time_utc"].min(), "->", cgm["time_utc"].max())
    print("Fingerstick glucose range (mg/dL):", round(fs["glucose"].min()), "-", round(fs["glucose"].max()))

    print("\n=== Block 2: clean the CGM signal ===")
    g = to_grid(cgm)
    print(f"grid       : {len(g)} slots, {g['cgm_raw'].isna().sum()} empty (missing values)")
    g = remove_warmup(g)
    g = remove_compression(g)
    g = fill_short_gaps(g)
    print(f"after clean: {g['cgm_clean'].isna().sum()} slots still empty (long gaps, warm-up, compression)")
    print("\nCheck against the answer key:")
    check_against_truth(g)
    print("\nExample: P01, first sensor warm-up")
    print(g[["patient_id", "time_local", "cgm_raw", "is_warmup", "cgm_clean"]].iloc[22:27])

    print("\n=== Block 3: causal Kalman filter ===")
    g = add_kalman(g, logbook)
    check_kalman(g)
    print("\nExample: P01 breakfast (glucose rising)")
    cols = ["time_local", "cgm", "cgm_kf", "roc_kf", "kf_sd", "kf_boost"]
    print(g.loc[g["patient_id"] == "P01", cols].iloc[103:111].round({"cgm_kf": 1, "roc_kf": 1, "kf_sd": 1}).to_string())

    print("\n=== Block 4: events, features and targets ===")
    g = add_events(g, logbook)
    g = add_features(g, patients)
    g = add_targets(g)
    print(f"events on grid: bolus {(g['bolus_u'] > 0).sum()}, carbs {(g['carbs_g'] > 0).sum()}, "
          f"hypo {(g['hypo_g'] > 0).sum()}")
    print("\nExample: P01 breakfast (inputs -> target)")
    cols = ["time_local", "cgm", "carbs_g", "bolus_u", "cob_g", "iob_u", "y_30", "low_30"]
    print(g.loc[g["patient_id"] == "P01", cols].iloc[104:111].round({"cob_g": 1, "iob_u": 1}).to_string())
    print("\nShare of rows with a low in the next 30 min, by age group:")
    print(g.groupby("age_group")["low_30"].mean().mul(100).round(1).astype(str) + " %")

    print("\n=== Block 5: load and check ===")
    out = save_table(g)
    print(f"saved {out}  ({len(g)} rows, {len(KEEP_COLUMNS)} columns) + metadata .json")
    rep = cgm_report(g)
    print("\nCGM report (mean per age group) -> reports/cgm_report.csv")
    print(rep.groupby("age_group")[["data_pct", "mean_mgdl", "tir_pct", "tbr70_pct",
                                    "cv_pct", "gmi_pct"]].mean().round(1).to_string())
    print("\nQC plot ->", qc_plot(g, "P01"))