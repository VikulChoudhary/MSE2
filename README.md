# HELIOS | Autonomous Self-Healing Fraud Detection MLOps

**Detect · Diagnose · Recover · Verify**

A working beginner-friendly **credit-card fraud-detection MLOps** project that uses **scikit-learn Logistic Regression with an automatic pure-Python fallback**. It adapts the earlier HELIOS Iris project to six transaction risk features and uses **logistic regression** plus a FastAPI dashboard, a SQLite experiment/incident store, JSON model versioning, PSI data drift monitoring, rule-driven recovery, rollback, and recovery verification.

> **Dataset transparency:** The bundled 1,600 transactions are entirely **synthetic, reproducible, and fictional**. No credit-card numbers, real transactions or personal information. The generated labels come from a deterministic synthetic risk rule with small noise. **Do not use this model to approve/decline actual transactions.** The validation metrics only apply to the artificial holdout; they do not represent real-world fraud performance.

## First run (Windows / VS Code / PowerShell)

1. Unzip into a **new, separate `HELIOS` folder**. Do not copy the new files over your old Iris `runtime/` directory. Model artifacts/schemas are incompatible.
2. From inside the new `HELIOS` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.demo
python -m uvicorn main:app --reload
```

3. Dashboard: http://127.0.0.1:8000/ ; Swagger API: http://127.0.0.1:8000/docs .

If your existing `.venv` is in the **parent folder**, activate it instead with `..\.venv\Scripts\Activate.ps1`. Do not create two virtual environments unnecessarily.

The server initializes a quality-gated champion **model v1** on the first run. Runtime artifacts are stored in the `runtime/` folder (which is excluded from the ZIP). The service starts a monitor every 60 seconds by default.

## Your ML pipeline

```
Synthetic CSV (1,600 labeled transactions)
   -> deterministic stratified 75/25 training/validation split
   -> scikit-learn Logistic Regression (or automatic pure-Python fallback; fraud = 1)
   -> evaluate accuracy, fraud precision, recall, F1 and confusion counts
   -> quality gates: accuracy >= 0.90 and fraud recall >= 0.65
                      and <= 0.03 accuracy regression vs. champion
   -> accepted model stored as JSON model_vN.json and activated
   -> /predict API / dashboard records predictions into SQLite
   -> monitor: latency + prediction patterns + feature PSI drift
               + live metrics when authenticated labeled feedback arrives
   -> root cause rules -> retrain / rollback / repair / human attention
   -> recovery verification and incident tracking
```

The synthetic data has six features:

| Feature | Meaning | Example |
| --- | --- | --- |
| `amount` | Synthetic transaction amount (USD) | `65` |
| `hour` | Hour of day, 0–23 | `12` |
| `merchant_risk` | Fictional risk indicator, 0–1 | `0.15` |
| `international` | 1 if international, otherwise 0 | `0` |
| `transactions_24h` | Transactions in preceding day | `2` |
| `account_age_days` | Age of account in days | `650` |

`is_fraud` is the binary target: **0 = legitimate, 1 = fraud**. Class probability is a demo model output, not a real calibrated fraud probability.

## Train, monitor, run recovery

In a **second terminal** inside HELIOS, with the environment activated:

```powershell
python train.py            # evaluate and conditionally deploy model v2
python run_cycle.py        # run one monitoring/diagnosis/recovery cycle
python -m pytest -q        # automated integration tests
python -m scripts.demo     # full isolated self-healing demonstration
```

`python -m scripts.demo` does **not** modify your live models or database; it uses a temporary directory. The demo tests prediction, induced drift, safe refusal to retrain without labels, repair of a simulated pipeline failure, manual retraining, rollback, and incident history.

## Try prediction in Swagger

Open http://127.0.0.1:8000/docs -> POST `/predict` -> Try it out and submit:

```json
{
  "amount": 65,
  "hour": 12,
  "merchant_risk": 0.15,
  "international": 0,
  "transactions_24h": 2,
  "account_age_days": 650
}
```

A high-risk *synthetic example* is in `examples/suspicious_request.json`.

The API responds with `prediction` (0/1), `class_name`, `fraud_probability`, `confidence`, `model_version`, `latency_ms`, and `prediction_id`. `confidence` is the probability the model assigned to the returned label, and the decision threshold for fraud is **0.80**, deliberately chosen for this synthetic demo.

To simulate strong feature drift, send the suspicious example 35 times, then GET `/drift`. For drift alerts, a minimum of **30 recent predictions** is required. Drift is calculated using Population Stability Index (PSI), and is **not** direct evidence of fraud or poor accuracy.

## Administrative endpoints (local only)

Admin operations are disabled until you set an API key **before starting the server**:

```powershell
$env:HELIOS_ADMIN_API_KEY = "choose-your-own-long-local-secret"
$env:HELIOS_DEMO_MODE = "true"
python -m uvicorn main:app --reload
```

Keep this key private and only expose the server locally. Send it as `X-Admin-Key` on protected requests. Demonstration-only fault injection requires `HELIOS_DEMO_MODE=true`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | API/model/storage health |
| `POST /predict` | Classify and record fictional transaction |
| `POST /feedback` | Add a trusted label to a previous prediction (admin) |
| `GET /monitor` | Latency, feature drift, class patterns, live labeled quality |
| `GET /drift` | PSI feature drift scores |
| `GET /models` | Model versions and champion |
| `GET /experiments` | Local experiment history |
| `GET /incidents` | Recovery incident history |
| `POST /admin/recovery-cycle` | Diagnose and safely recover (admin) |
| `POST /admin/retrain` | Quality-gated retraining (admin) |
| `POST /admin/rollback` | Restore previously deployed healthy model (admin) |
| `POST /admin/verify` | Verify model and pipeline health (admin) |
| `POST /admin/demo/pipeline-failure` | Inject demo-only pipeline fault (admin) |
| `GET /metrics/prometheus` | Prometheus-compatible counters and latency |

### Safety rules

- Data drift **alone** does not automatically justify retraining. Reliable labels must be available before recovery can train from feedback. Otherwise the incident is `needs_attention`.
- Model acceptance requires **both accuracy and fraud recall**, in addition to no excessive regression. Majority-class accuracy alone is not sufficient.
- This project's pipeline repair is a **controlled local demonstration** of known faults, not generalized cloud/k8s repair.
- Drift after a retrain is reassessed: holdout performance passing is not automatically considered drift resolution.
- `feedback` is an authenticated label input in this educational app. A real fraud system needs trusted, delayed, audited outcome verification and strict data access controls.
- Standard-library logistic regression is intentionally simple and tuned for the synthetic data; performance on real transactions is unknown.

## Dataset regeneration

```powershell
python -m scripts.generate_demo_data
```

This regenerates the exact same 1,600 artificial rows deterministically. Re-running it is unnecessary for a fresh install. Do not change or replace the CSV while the server is running.

## Project layout

```
HELIOS/
├── helios/
│   ├── api.py                # FastAPI endpoints and request validation
│   ├── service.py            # Prediction and background monitoring
│   ├── training.py           # Stratified split, metrics, quality gates
│   ├── training_model.py     # Pure-Python weighted logistic regression
│   ├── monitoring.py         # PSI feature drift and live metrics
│   ├── recovery.py           # Root-cause and bounded recovery rules
│   ├── registry.py           # JSON model registry, champion, rollback
│   ├── database.py           # SQLite storage
│   ├── config.py             # Environment-variable configuration
│   ├── data/synthetic_fraud_transactions.csv
│   └── static/index.html     # Fraud detection control center
├── examples/                 # Legitimate and suspicious JSON inputs
├── scripts/demo.py           # End-to-end isolated demonstration
├── scripts/generate_demo_data.py
├── tests/                     # Automated tests
├── train.py                   # Manual train/deploy CLI
├── run_cycle.py               # One monitoring/recovery cycle
├── main.py                    # ASGI entrypoint
├── .env.example               # Config values (not auto-loaded)
├── Dockerfile / docker-compose.yml
└── requirements.txt           # No sklearn/numpy/joblib
```

## Optional experiment tracking and Docker

SQLite records all experiments by default. Optional MLflow is configured using `requirements-mlflow.txt` and `HELIOS_MLFLOW_TRACKING_URI` (separate from the minimal install). Docker and GitHub Actions CI configuration are bundled.

## Boundaries

This is a reproducible local **educational research prototype**; it does not interface with banks or real payment systems, and does not have production-grade authentication, data governance, monitoring infrastructure, or real-world predictive validation. No actual credit-card data is included.

## scikit-learn and the Windows DLL problem

The original ML pipeline used scikit-learn, so this edition includes it in `requirements.txt`.
On some locked-down Windows computers, importing scikit-learn fails with an
**Application Control policy has blocked this file** error. This is a Windows
security policy, not a missing package. Do **not** disable security protections.

HELIOS uses `LogisticRegression` from scikit-learn when it works. If a native
DLL is blocked, HELIOS automatically trains with its standard-library Python
implementation instead, prints a clear message, and continues running. Both
backends produce the same portable JSON model format, so prediction/recovery
can work without importing scikit-learn. The training output includes `backend`.

In PowerShell, choose a backend before running commands:

```powershell
$env:HELIOS_ML_BACKEND = "auto"       # Default: try sklearn, fall back if blocked
python train.py
$env:HELIOS_ML_BACKEND = "python"     # Always use pure Python (most reliable)
python train.py
$env:HELIOS_ML_BACKEND = "sklearn"    # Try sklearn; fallback on DLL/import error
python train.py
```

For a quick check:

```powershell
python -c "from sklearn.linear_model import LogisticRegression; print('sklearn import OK')"
```

If this fails with a DLL policy error, you can still run HELIOS using the
`python` backend. To use scikit-learn itself, Windows/organization policy must
permit the dependency; Python code cannot bypass that policy.

## Beginner map: which file does what?

- `train.py`: start a training experiment
- `helios/training.py`: read CSV, split data, score, and apply quality gates
- `helios/training_model.py`: Logistic Regression fit/predict; two backends
- `main.py`: start FastAPI
- `helios/api.py`: API routes for prediction, monitoring, retraining
- `helios/monitoring.py`: monitor latency and drift
- `helios/recovery.py`: diagnose issues and select recovery actions
- `helios/registry.py`: save models, change champion, roll back
- `run_cycle.py`: run a single monitoring + recovery cycle
- `scripts/demo.py`: demonstrate the full flow in an isolated environment
- `tests/`: tests (run `python -m pytest -q`)

**Important:** synthetic transactions and holdout scores are for coursework,
not for real transaction decisions.
