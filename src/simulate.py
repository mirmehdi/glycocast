"""
simulate.py - Step 2a: simulate ALL virtual patients and save the clean data as parquet for aws 

Output:
    data/sim/clean.parquet   clean simulation, one row per patient per 5 min (ground truth)
    data/raw/patients.csv    one row per patient: ID, age group, age, weight, daily insulin

Run:
    python -W ignore src/simulate.py --days 1 --n 1    # quick test: 3 patients, 1 day
    python -W ignore src/simulate.py                   # full: 30 patients, 14 days

Regulatory: 
The data should be prepared according to GDPD: Pseudonymization, safe harbor, 
ICD-10 code, E10 (type 1 diabetes) in patients.csv, to show EHR interoperability,
, Minimum necessary
"""

# ----------------------------------------------------------------------------
# 1. Imports
# ----------------------------------------------------------------------------
import argparse                            # read options from the command line (--days, --n)
import time                                # measure how long the simulation takes
from datetime import datetime, timedelta   # start time and duration for simulation
from multiprocessing import Pool           # run several patients at the same time (CPU cores)
from pathlib import Path                   # clean, OS-independent file paths

import pandas as pd                        # tables (DataFrames)

# simglucose building blocks: each one is a "part" of the virtual world
from simglucose.patient.t1dpatient import T1DPatient, PATIENT_PARA_FILE, Action   # the body, its parameter file, an action (insulin + carbs)
from simglucose.sensor.cgm import CGMSensor                               # the CGM sensor
from simglucose.actuator.pump import InsulinPump                          # the insulin pump
from simglucose.simulation.scenario_gen import RandomScenario             # the meals
from simglucose.controller.basal_bolus_ctrller import BBController, CONTROL_QUEST  # therapy + its parameter file
from simglucose.simulation.env import T1DSimEnv                           # puts all parts together
from simglucose.simulation.sim_engine import SimObj                       # runs the clock

pd.set_option("display.width", 120)        # print wide tables on one line

# ----------------------------------------------------------------------------
# 2. Settings
# ----------------------------------------------------------------------------
START = datetime(2026, 1, 5)               # simulation starts at midnight, 5 Jan 2026 (local time)
STEP_MIN = 5                               # one row = 5 minutes
GROUPS = ["child", "adolescent", "adult"]  # the 3 age groups in simglucose (10 patients each)
# Hypo treatment ("15-15 rule"): below 70 mg/dL eat 15 g fast carbs, wait 15 min before the next one
RESCUE_BELOW = 70        # mg/dL
RESCUE_CARBS_G = 15      # grams of fast carbs
RESCUE_WAIT_MIN = 15     # minutes to wait before the next rescue snack
SIM_DIR = Path("data/sim")                 # clean simulation (ground truth)
RAW_DIR = Path("data/raw")                 # files that look like real device exports



# ----------------------------------------------------------------------------
# 2b. Class: the virtual world + a patient who treats lows (15-15 rule)
# ----------------------------------------------------------------------------
class RescueEnv(T1DSimEnv):
    """Same as T1DSimEnv, but the patient eats 15 g fast carbs when glucose < 70.

    No insulin is given for these carbs (people do not bolus for hypo treatment).
    The rescue carbs are stored separately in `rescue_hist` (grams per 5-min step).
    """

    def _reset(self):
        super()._reset()                 # do everything the normal world does at start
        self.last_rescue = -999          # minute of the last rescue snack (-999 = never)
        self.rescue_now = 0.0            # rescue grams inside the current 5-min step
        self.rescue_hist = []            # rescue grams per 5-min step (our new column)

    def mini_step(self, action):
        """One MINUTE of the world (the clock calls this 5 times per 5-min step)."""
        meal = self.scenario.get_action(self.time).meal                         # planned meal (g/min)
        insulin = self.pump.basal(action.basal) + self.pump.bolus(action.bolus)  # insulin (U/min)

        # the person checks: am I low, and did I wait 15 min since the last snack?
        rescue = 0.0
        low = self.patient.observation.Gsub < RESCUE_BELOW
        waited = self.patient.t - self.last_rescue >= RESCUE_WAIT_MIN
        if low and waited:
            rescue = RESCUE_CARBS_G           # eat 15 g in this minute
            self.last_rescue = self.patient.t
        self.rescue_now += rescue

        # the body gets the planned meal + the rescue carbs + the insulin
        self.patient.step(Action(insulin=insulin, CHO=meal + rescue))

        # report only the PLANNED meal back -> the therapy does not bolus for rescue carbs
        bg = self.patient.observation.Gsub
        cgm = self.sensor.measure(self.patient)
        return meal, insulin, bg, cgm

    def step(self, action, **kwargs):
        """One 5-min step: run the normal step, then store the rescue grams of this step."""
        self.rescue_now = 0.0
        out = super().step(action, **kwargs)
        self.rescue_hist.append(self.rescue_now)
        return out
# ----------------------------------------------------------------------------
# 3. Function: simulate one patient  (same as Step 1)
# ----------------------------------------------------------------------------
def simulate_patient(name: str, days: int, seed: int) -> pd.DataFrame:
    """Run one virtual patient for `days` days. Returns one row per 5 min."""
    patient = T1DPatient.withName(name)                          # the body
    sensor = CGMSensor.withName("GuardianRT", seed=seed)         # 5-min CGM with noise
    pump = InsulinPump.withName("Insulet")                       # the pump
    scenario = RandomScenario(start_time=START, seed=seed)       # random meals
    env = RescueEnv(patient, sensor, pump, scenario)             # the virtual world (+ hypo treatment)
    controller = BBController()                                  # manual basal-bolus therapy

    sim = SimObj(env, controller, timedelta(days=days), animate=False)
    sim.simulate()                                               # run the clock

    df = sim.results().reset_index()                             # time index -> column
    df = df.iloc[:-1].copy()                                     # last row has no action yet
    df["rescue_g"] = env.rescue_hist                             # rescue carbs per 5-min step
    return df                                         # last row has no action yet


# ----------------------------------------------------------------------------
# 4. Function: build the patient list (who, which ID, which seed)
# ----------------------------------------------------------------------------
def build_patient_table(n_per_group: int) -> pd.DataFrame:
    """One row per patient with a pseudonymous ID, a seed and basic demographics."""
    # simglucose names: child#001 ... child#010, adolescent#001 ..., adult#001 ...
    names = [f"{g}#{i:03d}" for g in GROUPS for i in range(1, n_per_group + 1)]

    pts = pd.DataFrame({"sim_name": names})
    pts["patient_id"] = [f"P{i:02d}" for i in range(1, len(pts) + 1)]   # P01, P02, ... (pseudonymous)
    pts["seed"] = range(1, len(pts) + 1)                                # fixed seed per patient
    pts["age_group"] = pts["sim_name"].str.split("#").str[0]            # "child#001" -> "child"

    # demographics from simglucose's own parameter files
    quest = pd.read_csv(CONTROL_QUEST)            # Age, TDI (total daily insulin), CR, CF
    params = pd.read_csv(PATIENT_PARA_FILE)       # BW (body weight) + many model constants
    pts = pts.merge(quest[["Name", "Age", "TDI"]], left_on="sim_name", right_on="Name")
    pts = pts.merge(params[["Name", "BW"]], on="Name")
    pts = pts.rename(columns={"Age": "age", "TDI": "daily_insulin_u", "BW": "weight_kg"})
    pts["weight_kg"] = pts["weight_kg"].round(1)
    pts["daily_insulin_u"] = pts["daily_insulin_u"].round(1)

    return pts[["patient_id", "sim_name", "seed", "age_group", "age", "weight_kg", "daily_insulin_u"]]


# ----------------------------------------------------------------------------
# 5. Function: worker for ONE patient (runs inside a parallel process)
# ----------------------------------------------------------------------------
def run_one(args) -> pd.DataFrame:
    """Simulate one patient and return a clean, tidy table."""
    patient_id, sim_name, seed, days = args       # one tuple in, because Pool.map passes one argument

    df = simulate_patient(sim_name, days, seed)

    # tidy table with clear names and units
    out = pd.DataFrame({
        "patient_id": patient_id,
        "time": df["Time"],                                   # local time, every 5 min
        "bg_true": df["BG"],                                  # true glucose, no sensor noise (mg/dL)
        "cgm": df["CGM"],                                     # sensor glucose with noise (mg/dL)
        "carbs_g": df["CHO"] * STEP_MIN,                      # g/min x 5 min = grams in this step
        "insulin_u": df["insulin"] * STEP_MIN,                # U/min x 5 min = units in this step
        "rescue_g": df["rescue_g"],                           # hypo-treatment carbs in this step (g)
    })
    print(f"  done {patient_id} ({sim_name})", flush=True)   # flush = print immediately
    return out


# ----------------------------------------------------------------------------
# 6. Main: run everything
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # read options from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14, help="days per patient")
    ap.add_argument("--n", type=int, default=10, help="patients per age group (max 10)")
    opt = ap.parse_args()

    # make the output folders if they do not exist yet
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # 6.1 who do we simulate?
    pts = build_patient_table(opt.n)
    print(f"Simulating {len(pts)} patients x {opt.days} days ...")

    # 6.2 run all patients in parallel: Pool() uses all CPU cores
    jobs = [(r.patient_id, r.sim_name, r.seed, opt.days) for r in pts.itertuples()]
    t0 = time.time()
    with Pool() as pool:
        results = pool.map(run_one, jobs)         # list of tables, same order as `jobs`
    print(f"Simulation took {(time.time() - t0) / 60:.1f} min")

    # 6.3 stack all patients into ONE table and save it
    clean = pd.concat(results, ignore_index=True)
    clean.to_parquet(SIM_DIR / "clean.parquet", index=False)

    # 6.4 save the patient table (without the simulator name and seed -> not "device" data)
    pts.drop(columns=["sim_name", "seed"]).to_csv(RAW_DIR / "patients.csv", index=False)

    # 6.5 quick checks
    print("\nclean.parquet:", clean.shape, " rows per patient:", len(clean) // len(pts))
    summary = clean.groupby("patient_id").agg(
        mean_bg=("bg_true", "mean"),
        pct_below70=("bg_true", lambda x: 100 * (x < 70).mean()),
        pct_70_180=("bg_true", lambda x: 100 * x.between(70, 180).mean()),
        meals=("carbs_g", lambda x: (x > 0).sum()),
        rescues=("rescue_g", lambda x: (x > 0).sum()),
    ).round(1)
    summary = summary.join(pts.set_index("patient_id")["age_group"])
    print(summary)
    print("\nBy age group:")
    print(summary.groupby("age_group")[["mean_bg", "pct_below70", "pct_70_180"]].mean().round(1))