GlycoCast – CGM glucose forecasting pipeline

Small end-to-end project: simulated type-1 diabetes data -> ETL with signal processing -> 30-min glucose forecast and low-glucose alarm, validated patient-wise.

CRISP-DM
Business understanding – safe, useful glucose forecast / low warning (EU MDR / FDA context)
Data understanding – CGM (5 min), fingerstick, insulin, carbs; known device artifacts
Data preparation – src/etl.py
Modeling – src/train.py
Evaluation – RMSE, Clarke error grid A+B, sensitivity, false alarms/day, lead time
Deployment – AWS S3 + Athena (data lake)
Run
bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -W ignore src/simulate.py --days 3   # quick test
python -W ignore src/simulate.py            # full: 10 patients x 14 days

Data is not stored in git. It is fully reproducible from fixed random seeds.