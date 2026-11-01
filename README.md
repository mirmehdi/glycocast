# GlycoCast – CGM glucose forecasting and low-glucose alarm

An end-to-end data science pipeline on continuous glucose monitoring (CGM) data:
**simulated patients → messy device exports → ETL with signal processing → glucose forecast,
low-glucose alarm and daily pattern profiles**, validated patient-wise with clinical metrics.

It mirrors the kind of features mySugr / Roche Diabetes Care ships as certified medical
software (glucose prediction, low-glucose warning) under EU MDR / FDA.

> Status: work in progress. See [Roadmap](#roadmap) for what is done and what is next.

---

## Why this project

People with type-1 diabetes manage their glucose all day. Lows (< 70 mg/dL) are dangerous,
especially at night. A good forecast and an early warning give the user time to act
(eat carbs, pause insulin). For a medical feature, accuracy alone is not enough. The system must be:

| Requirement | What it means here |
|---|---|
| **Safe** | Few missed lows (high alarm sensitivity) |
| **Useful** | Few false alarms per day (no alarm fatigue) |
| **Robust** | Works on messy real-world data and when carb logs are missing |
| **Honest** | Gives an uncertainty range, not only one number |
| **Validated & traceable** | Tested on unseen patients, per subgroup; reproducible data and models |

---

## What the system delivers

1. **Glucose forecast** – 30 min ahead (also 15 / 60 / 120 min), with a 10–90 % uncertainty range.
2. **Low-glucose alarm** – warns when glucose is likely to drop below 70 mg/dL within 30 min.
3. **Daily pattern profile** – typical day patterns per patient (e.g. night lows, dawn rise,
   post-meal spikes) as a basis for digital biomarkers.

---

## Data

All data is **simulated** with [simglucose](https://github.com/jxx123/simglucose), the open
Python version of the UVA/Padova type-1 diabetes simulator (FDA-accepted for pre-clinical
in-silico testing).

**Virtual cohort:** 30 patients (10 children, 10 adolescents, 10 adults), 14 days each,
CGM every 5 min, manual basal-bolus therapy, random meals, and a patient who treats lows
(15-15 rule: 15 g fast carbs below 70 mg/dL, no bolus).

**Device exports** – three files that look like what an app receives, each messy on purpose:

| File | Content | Built-in problems |
|---|---|---|
| `cgm.csv` | CGM every 5 min | UTC time, ±60 s jitter, duplicates, short & long gaps, sensor warm-up error (day 0 and sensor change at day 10), night compression lows |
| `logbook.csv` | Meals, boluses, hypo treatments | Local time with offset, carb estimation error (±20 %), ~10 % meals not logged |
| `fingerstick.csv` | 3–4 meter values per day | mmol/L instead of mg/dL, German date format, no time zone, ~5 % meter error |

A hidden ground truth (`data/sim/clean.parquet`, artifact flags) is kept **only** to check
whether the ETL removes the artifacts. It is never used for training.

---

## Pipeline

```
simulate.py          etl.py                                   train.py
-----------          ------                                   --------
simglucose  ──►  raw device CSVs  ──►  clean 5-min table  ──►  models  ──►  metrics + report
(30 pts)         (messy on purpose)    (Parquet, versioned)    (patient-wise validation)
                                       + CGM report per patient
```

**ETL steps:** unit conversion (mmol/L × 18) · time-zone alignment · de-duplication ·
5-min grid · remove sensor warm-up · detect compression lows · fill only short gaps (≤ 15 min) ·
Kalman filter (sensor fusion / denoising) · align meals and insulin to the grid ·
features (rate of change, insulin on board, carbs on board, time of day).

**CGM report per patient:** time in range (70–180), time below range, CV, GMI.

---

## Modeling

A model ladder – each step answers one question, all on the **same split and metrics**:

| Level | Model | Question |
|---|---|---|
| L0 | Persistence (future = now) | What is the minimum to beat? |
| L1 | Linear trend / Kalman extrapolation | Is signal processing alone enough? |
| L2 | Ridge on last 60 min of CGM | Does a simple linear model help? |
| L3 | LightGBM (CGM features) | Does non-linearity help? |
| L4 | LightGBM + insulin & carbs on board | Do insulin and carb inputs help? |
| L5 | Causal TCN, multimodal (CGM + insulin + carbs channels), modality dropout, quantile output | Does deep learning help, and can it give uncertainty and survive missing carbs? |

**Low alarm, three ways:** threshold on the forecast · direct classifier · lower quantile < 70.
For each, a threshold sweep gives the **sensitivity vs false-alarms-per-day** trade-off.

**Analysis:** PCA on daily CGM profiles ("eigen-days") for patient phenotypes;
SHAP for explainability.

### Validation
- **Patient-wise split**, stratified by age group (test patients never seen in training).
- **Leakage demo:** random row split vs patient-wise split.
- **Ablation:** model performance with carb input removed.
- **Subgroups:** age group · day vs night · glucose range (< 70 / 70–180 / > 180).

### Metrics
RMSE · MAE · Clarke error grid zones A+B · alarm sensitivity · false alarms per day · lead time.

---

## Results

*To be filled after training.*

| Model | RMSE 30 min | Clarke A+B | Alarm sensitivity | False alarms / day |
|---|---|---|---|---|
| Persistence | – | – | – | – |
| LightGBM | – | – | – | – |
| TCN (quantile) | – | – | – | – |

---

## Project structure

```
glycocast/
├── src/
│   ├── simulate.py      # Step 2: simulate patients, export device files
│   ├── etl.py           # Step 3: clean, align, Kalman, features, CGM report
│   └── train.py         # Step 4: models, validation, metrics
├── data/                # not in git – fully reproducible from fixed seeds
│   ├── sim/             # clean simulation (ground truth)
│   ├── raw/             # messy device exports + patients.csv
│   └── processed/       # clean training table (Parquet)
├── notebooks/           # EDA
├── reports/             # project report, figures
├── requirements.txt
└── requirements-lock.txt
```

---

## Setup and run

```bash
conda create -n glycast python=3.11 -y
conda activate glycast
pip install -r requirements.txt          # includes setuptools<81 (needed by simglucose's old gym)

python -W ignore src/simulate.py --days 1 --n 1   # quick test: 3 patients, 1 day
python -W ignore src/simulate.py                  # full: 30 patients x 14 days
```

`-W ignore` hides harmless warnings from the old `gym` package.
Data is not stored in git; fixed random seeds make every run identical.

---

## Limitations

- Simulated physiology: no exercise, stress, illness or changing insulin sensitivity.
  Results are an **optimistic upper bound** for real patients.
- Only 30 virtual patients from one model family.
- Simplified therapy (basal-bolus formula) and hypo treatment (fixed 15-15 rule).

## Privacy and regulation (planned)

Pseudonymous patient IDs from the start. Planned: data minimization in the training table,
date shifting, a data dictionary with sensitivity classes, and an AWS setup in an EU region
with encryption at rest and in transit. Context: GDPR (health data = special category),
HIPAA for US use, EU MDR / FDA for the medical feature.

---

## Roadmap

- [x] Business and data understanding
- [x] Project setup (conda, git, structure)
- [x] Simulation of 30 patients with hypo treatment
- [ ] Messy device exports
- [ ] ETL + Kalman filter + CGM report
- [ ] EDA + eigen-days (PCA)
- [ ] Models L0–L4, leakage demo, alarm, subgroups
- [ ] TCN with quantiles and modality dropout
- [ ] AWS: S3 data lake + Athena
- [ ] Privacy and regulatory section

**Vision:** at mySugr scale, pretrain the TCN self-supervised on millions of unlabeled CGM
days and fine-tune it per task – a foundation model for CGM.

---

*Author: Dr.-Ing. Mir Mehdi Seyedebrahimi*