"""
export_devices.py - Step 2b: turn the clean simulation into messy "device" files.

Block 1: CGM export.

Input:   data/sim/clean.parquet        (from simulate.py)
Output:  data/raw/cgm.csv              (what the app receives from the CGM sensor)
         data/truth/cgm_flags.parquet  (where we put which artifact -> to check the ETL later)

Run:     python src/export_devices.py
"""

# 1. import 

from pathlib import Path

import numpy as np 
import pandas as pd 

# 2. setting 
SIM_FILE = Path("data/sim/clean.parquet")
RAW_DIR = Path("data/raw")
TRUTH_DIR = Path("data/truth")

TZ = "Europe/Vienna"               # patients live in Vienna (local time)
PER_DAY = 288                      # 5-min steps per day (24 h x 12)

SENSOR_CHANGE_DAY = 10             # new sensor at day 10 ->to have second warm-up
CHANGE_GAP_STEPS = 6               # 30 min without data while changing the sensor
WARMUP_STEPS = 24                  # first 2 h of each sensor are less accurate
SHORT_GAP_RATE = 0.02              # 2 % of single values are lost (Bluetooth)
DUPLICATE_RATE = 0.01              # 1 % of rows are uploaded twice
JITTER_SEC = 60                    # device clock is up to +-60 s off the 5-min grid
CGM_MIN, CGM_MAX = 40, 400         # the sensor cannot report outside this range


CARB_ERROR_SD = 0.20               # people misjudge carbs by about +-20 %. they forgot to log it on time. 
MEAL_MISS_RATE = 0.10              # 10 % of meals are never logged
HYPO_MISS_RATE = 0.30              # 30 % of hypo treatments are never logged
LOG_DELAY_MAX_MIN = 15             # manual entries are typed 0-15 min after eating
BOLUS_MIN_U = 0.05                 # smaller insulin "spikes" are not a real bolus

# 3. Function: make the CGM of one patinet messy 

def make_cgm(p: pd.DataFrame, rng: np.random.Generator):
    """Return (export table, flag table) for one patient.

    p   : clean rows of one patient (sorted by time, one row per 5 min)
    rng : random generator of this patient (fixed seed -> reproducible)
    """
    n = len(p)
    cgm = p["cgm"].to_numpy(dtype=float).copy()   # start from the simulated sensor signal

    # flags: True where we inserted an artifact (only for checking, never for training)
    warmup = np.zeros(n, dtype=bool)
    compression = np.zeros(n, dtype=bool)
    reported = np.ones(n, dtype=bool)              # False = value never reached the app
    sensor_no = np.ones(n, dtype=int)              # 1 = first sensor, 2 = after sensor change

    # 3.1 Sensor change at day 10: 30 min gap, then a new sensor
    change = SENSOR_CHANGE_DAY * PER_DAY
    reported[change - CHANGE_GAP_STEPS:change] = False
    sensor_no[change:] = 2

    # 3.2 Warm-up error: first 2 h of each sensor read too high and are noisy.
    #     Error = 30 mg/dL at start, fading away (exponential), plus random noise.
    for start in [0, change]:
        k = np.arange(WARMUP_STEPS)                            # 0, 1, ..., 23
        idx = start + k
        cgm[idx] += 30 * np.exp(-k / 8) + rng.normal(0, 12, WARMUP_STEPS)
        warmup[idx] = True

    # 3.3 Compression lows: patient lies on the sensor at night.
    #     Shape: fast drop (15 min) -> flat (15-30 min) -> fast recovery (15 min).
    for _ in range(rng.integers(1, 4)):                        # 1 to 3 events
        day = rng.choice([d for d in range(1, 14) if d != SENSOR_CHANGE_DAY])
        i0 = day * PER_DAY + rng.integers(12, 60)              # start between 01:00 and 05:00
        depth = rng.uniform(40, 70)                            # how deep the fake drop is (mg/dL)
        hold = rng.integers(3, 7)                              # flat part: 3-6 steps
        shape = np.concatenate([np.linspace(0, 1, 4)[1:],      # down: 1/3, 2/3, 1
                                np.ones(hold),                 # flat
                                np.linspace(1, 0, 4)[:-1]])    # up: 1, 2/3, 1/3
        idx = i0 + np.arange(len(shape))
        cgm[idx] -= depth * shape
        compression[idx] = True

    # 3.4 Gaps: many short dropouts + 1-2 long signal losses (1-3 h)
    reported &= rng.random(n) > SHORT_GAP_RATE
    for _ in range(rng.integers(1, 3)):
        i0 = rng.integers(PER_DAY, n - 40)                     # not on day 0
        reported[i0:i0 + rng.integers(12, 37)] = False         # 12-36 steps = 1-3 h

    # 3.5 Device limits and integer values (sensors report whole mg/dL)
    cgm = np.clip(np.round(cgm), CGM_MIN, CGM_MAX).astype(int)

    # 3.6 Flag table (truth) - one row per 5-min step, also for missing values
    flags = pd.DataFrame({"patient_id": p["patient_id"].to_numpy(), "time": p["time"].to_numpy(),
                          "sensor_no": sensor_no, "warmup": warmup,
                          "compression": compression, "reported": reported})

    # 3.7 Export table: only reported values, device clock in UTC with jitter
    pid = p["patient_id"].iloc[0]
    t_local = p["time"][reported]
    jitter = pd.to_timedelta(rng.uniform(-JITTER_SEC, JITTER_SEC, len(t_local)), unit="s")
    t_utc = (t_local + jitter).dt.tz_localize(TZ).dt.tz_convert("UTC")

    out = pd.DataFrame({
        "patient_id": pid,
        "sensor_id": [f"SG-{pid}-{s}" for s in sensor_no[reported]],   # sensor serial
        "timestamp_utc": t_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),       # e.g. 2026-01-05T07:00:12Z
        "glucose_mg_dl": cgm[reported],
    })

    # 3.8 Duplicates: some rows are uploaded twice (e.g. after a sync retry)
    dup = out.sample(frac=DUPLICATE_RATE, random_state=int(rng.integers(1_000_000)))
    out = pd.concat([out, dup]).sort_values("timestamp_utc")

    return out, flags


# 4. Function: build the logbook of ONE patient that is normally from meals and pump.  I considered pump log to be exact and automatic
def make_logbook(p: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Return the logbook events of one patient (meals, boluses, hypo treatments, basal rate).

    Manual entries (carbs, hypo treatment) are imperfect: estimation error, delay, sometimes missing.
    Pump entries (bolus, basal rate) are recorded automatically and are exact.
    """
    pid = p["patient_id"].iloc[0]
    events = []                                    # list of (time, event, value, unit, source)

    # 4.1 Basal rate: the pump runs a constant background insulin. One entry at the start.
    basal_u = p["insulin_u"].min()                 # insulin per 5-min step without any bolus
    events.append((p["time"].iloc[0], "basal_rate", round(basal_u * 12, 2), "U/h", "pump"))

    # 4.2 Boluses: insulin clearly above the basal rate (pump -> exact, automatic)
    bolus = p[p["insulin_u"] - basal_u > BOLUS_MIN_U]
    for t, u in zip(bolus["time"], bolus["insulin_u"] - basal_u):
        events.append((t, "bolus", round(u, 1), "U", "pump"))

    # 4.3 Meals: typed by the person -> wrong estimate, rounded to 5 g, sometimes forgotten
    meals = p[p["carbs_g"] > 0]
    for t, g in zip(meals["time"], meals["carbs_g"]):
        if rng.random() < MEAL_MISS_RATE:
            continue                               # forgotten -> no entry (bolus stays!)
        estimate = g * (1 + rng.normal(0, CARB_ERROR_SD))
        estimate = max(5, 5 * round(estimate / 5))  # people round to 5 g, at least 5 g
        delay = pd.Timedelta(minutes=rng.uniform(0, LOG_DELAY_MAX_MIN))
        events.append((t + delay, "carbs", estimate, "g", "manual"))

    # 4.4 Hypo treatments: 15 g fast carbs (exact, e.g. glucose tablets), often not logged
    rescues = p[p["rescue_g"] > 0]
    for t, g in zip(rescues["time"], rescues["rescue_g"]):
        if rng.random() < HYPO_MISS_RATE:
            continue
        delay = pd.Timedelta(minutes=rng.uniform(0, LOG_DELAY_MAX_MIN))
        events.append((t + delay, "hypo_treatment", g, "g", "manual"))

    # 4.5 Table with local time + UTC offset, e.g. 2026-01-05T08:37:05+0100
    lb = pd.DataFrame(events, columns=["time", "event", "value", "unit", "source"])
    lb["time"] = lb["time"] + pd.to_timedelta(rng.uniform(0, 60, len(lb)), unit="s")  # seconds
    lb = lb.sort_values("time")
    lb["timestamp_local"] = lb["time"].dt.tz_localize(TZ).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    lb["patient_id"] = pid
    return lb[["patient_id", "timestamp_local", "event", "value", "unit", "source"]]
# 5. main 
if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    clean = pd.read_parquet(SIM_FILE).sort_values(["patient_id", "time"])

    cgm_all, flags_all, logbook_all = [], [], []
    for pid, p in clean.groupby("patient_id"):
        rng = np.random.default_rng(int(pid[1:]))       # "P07" -> seed 7 (reproducible)
        out, flags = make_cgm(p, rng)
        cgm_all.append(out)
        flags_all.append(flags)
        logbook_all.append(make_logbook(p, rng))

    cgm = pd.concat(cgm_all, ignore_index=True)
    flags = pd.concat(flags_all, ignore_index=True)
    cgm.to_csv(RAW_DIR / "cgm.csv", index=False)
    flags.to_parquet(TRUTH_DIR / "cgm_flags.parquet", index=False)

    # quick checks
    print("cgm.csv:", len(cgm), "rows   (clean simulation:", len(clean), "rows)")
    print(cgm.head(3))
    print("\nArtifacts in total:")
    print("  missing 5-min values :", (~flags["reported"]).sum())
    print("  duplicate rows       :", cgm.duplicated().sum())
    print("  warm-up steps        :", flags["warmup"].sum())
    print("  compression steps    :", flags["compression"].sum(),
          f"({flags.groupby('patient_id')['compression'].sum().gt(0).sum()} patients)")

    
    logbook = pd.concat(logbook_all, ignore_index=True)
    logbook.to_csv(RAW_DIR / "logbook.csv", index=False)
    print("\nlogbook.csv:", len(logbook), "rows")
    print(logbook.head(6))
    print("\nEvents logged vs. happened:")
    print("  meals  :", (logbook["event"] == "carbs").sum(), "of", (clean["carbs_g"] > 0).sum())
    print("  hypos  :", (logbook["event"] == "hypo_treatment").sum(), "of", (clean["rescue_g"] > 0).sum())
    print("  boluses:", (logbook["event"] == "bolus").sum())
