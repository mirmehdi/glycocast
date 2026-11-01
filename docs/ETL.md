# ETL – from messy device files to a clean training table

`src/etl.py` turns three device exports into **one clean, versioned training table**
on a 5-minute grid. Every cleaning step is checked against the simulator's answer key.

```bash
python src/etl.py      # ~10 s, reads data/raw/, writes data/processed/ and reports/
```

---

## Inputs

| File | Source | Time format | Unit | Problems built in |
|---|---|---|---|---|
| `cgm.csv` | CGM sensor | UTC (`...Z`), ±60 s jitter | mg/dL | duplicates, gaps, warm-up error, compression lows, slow drift |
| `logbook.csv` | Pump (bolus, basal) + manual entries (carbs, hypo treatment) | local + offset (`+0100`) | U, g | carb error ±20 %, 10 % meals and 30 % hypo treatments not logged, entries 0–15 min late |
| `fingerstick.csv` | Blood glucose meter | local, **no time zone**, `dd.mm.yyyy HH:MM` | **mmol/L** | meter error ~5 % |
| `patients.csv` | User profile | – | – | – |

---

## Pipeline

| Block | Step | What it does | Why |
|---|---|---|---|
| **1 Extract** | Standardize | All times → UTC, all glucose → mg/dL (× 18), remove exact duplicates | One common "language" for all devices |
| **2 Clean CGM** | 5-min grid | Snap times to the nearest 5 min, average values in the same slot, build a full grid | Gaps become visible; "30 min ahead" has a clear meaning |
| | Warm-up | Remove the first 2 h of each sensor (`sensor_id`) | New sensors read too high |
| | Compression lows | Rule: at night, fall ≥ 30 mg/dL in 15 min **and** recovery within 60 min → remove | Fake lows would train false alarms |
| | Short gaps | Straight line for gaps ≤ 15 min; longer gaps stay empty | Fill error is small; long gaps would invent data |
| **3 Kalman** | Causal filter | 2 states (glucose, speed), past values only; higher process noise for 90 min after a logged meal or bolus | Rate-of-change feature, gap bridging, uncertainty |
| **4 Features** | Events | Boluses, carbs, hypo treatments summed per 5-min slot | Align events with CGM |
| | On board | Insulin (τ = 60 min), carbs (τ = 40 min), hypo carbs (τ = 15 min), exponential decay | Turns single events into "still active" signals |
| | Context | Time of day (sin/cos), age group, missing mask | Daily rhythm, subgroups, data quality |
| | Targets | Cleaned CGM in 15/30/60/120 min; `low_30` = below 70 in the next 30 min; flag if the target was really measured | The "correct answers" for training |
| **5 Load** | Save | `data/processed/train_v1.parquet` + `train_v1.json` (git commit, input file hashes, settings) | Traceability (MDR / FDA) |
| | Report | Standard CGM metrics per patient → `reports/cgm_report.csv` | Clinical summary |
| | QC plot | Raw vs cleaned vs true glucose → `reports/figures/qc_P01.png` | Visual proof that cleaning works |

---

## Design principles

1. **Targets may use hindsight, inputs may not.** The target (`cgm_clean`) may use future
   values (gap filling, compression detection). Every input feature uses only data known
   at that moment → no leakage from the future.
2. **Measure, don't assume.** Each cleaning step is compared with the answer key.
3. **Nothing is fitted on test patients.** Kalman settings were chosen on training
   patients only (P01–P07, P11–P17, P21–P27).
4. **Don't invent data.** Forgotten carb logs stay 0; long gaps stay empty.
5. **Traceable output.** Versioned file name + metadata with code version, input hashes and settings.

---

## Validation results

| Check | Result |
|---|---|
| Duplicates removed | 1,170 of 1,170 inserted |
| Warm-up removed | 1,440 of 1,440 steps |
| Compression lows | **recall 91 %, precision 86 %** (simple rule, 15 lines) |
| Short-gap fill (≤ 15 min) | error 1.2 mg/dL vs sensor noise 8.9 mg/dL |
| Data available after cleaning | 95.5 % per patient (consensus target ≥ 70 %) |
| Events on grid | bolus 1,550 · carbs 1,379 · hypo 527 (3 pairs merged into one slot) |

### Key finding: the Kalman filter does not denoise this sensor

| Signal | Error vs true glucose |
|---|---|
| Raw CGM | 8.9 mg/dL |
| Kalman (causal) | 9.3 mg/dL |

The sensor error is **slow drift**, not jitter: its autocorrelation is 0.96 after 5 min
and still 0.47 after 30 min. Smoothing cannot remove drift; it only adds delay.
This is why real sensors need **fingerstick calibration**.
→ The filter is kept only for the **rate-of-change** feature, where it is measurably better
(error 0.37 vs 0.43 mg/dL/min for a 15-min difference).

### CGM report (mean per age group)

| Age group | Data % | Mean (mg/dL) | Time in range % | Time < 70 % | CV % | GMI % |
|---|---|---|---|---|---|---|
| Child | 95.5 | 141.7 | 75.9 | 4.6 | 35.1 | 6.7 |
| Adolescent | 95.5 | 139.0 | 82.2 | 1.5 | 28.0 | 6.6 |
| Adult | 95.5 | 124.8 | 94.3 | 1.0 | 22.4 | 6.3 |

Children: CGM shows 4.6 % time below 70, the true value is 3.4 % →
**sensor noise inflates time below range.** Low-range accuracy matters most for the alarm.

### Class balance for the alarm (`low_30`)

Child 8.1 % · Adolescent 2.6 % · Adult 1.9 % of rows → strong imbalance;
evaluate with sensitivity and false alarms per day, not accuracy.

---

## Output table (`train_v1.parquet`, 120,959 rows × 32 columns)

| Group | Columns |
|---|---|
| Keys | `patient_id`, `age_group`, `time_utc`, `time_local`, `sensor_id` |
| CGM | `cgm_raw` (as received), `cgm` (warm-up + compression removed, **input**), `cgm_clean` (+ short gaps filled, **target only**), `cgm_missing`, `is_warmup`, `is_compression` |
| Kalman | `cgm_kf`, `roc_kf` (mg/dL per min), `kf_sd` |
| Events | `bolus_u`, `carbs_g`, `hypo_g`, `basal_u_h` |
| On board | `iob_u`, `cob_g`, `hob_g` |
| Time | `tod_sin`, `tod_cos` |
| Targets | `y_15`, `y_30`, `y_60`, `y_120`, `y_*_measured`, `low_30` |

---

## Limitations and next steps

- **Compression detection uses the future** (recovery). Fine for cleaning training data;
  a live product needs a causal version (e.g. physiologically impossible fall rate).
- **Meter time zone is assumed** (home zone) – wrong if the patient travels.
- **No drift correction yet.** Next step: a 3-state Kalman filter (glucose, speed, sensor bias)
  that fuses fingersticks – i.e. calibration.
- **Real data has no answer key** → validate against fingersticks and clinician labels.
