# HELIOS: simple start guide (Windows PowerShell)

## What it does

HELIOS trains a small model to flag **synthetic** credit-card transactions as
`legitimate` (0) or `fraud` (1). It serves predictions through FastAPI,
records what the model sees, detects drift or failures, chooses a rule-based
recovery action, and checks whether that recovery worked.

## First run

Open the extracted `HELIOS` folder in VS Code. In a terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.demo
python -m uvicorn main:app --reload
```

- Dashboard: http://127.0.0.1:8000
- Test predictions: http://127.0.0.1:8000/docs
- Stop the API: Ctrl+C

## What command runs the ML training pipeline?

Open another terminal with the environment activated:

```powershell
python train.py
```

This reads the CSV -> splits labeled data -> trains logistic regression ->
evaluates accuracy and **fraud recall** -> checks quality gates -> registers
and deploys a new version if it passes. Model and database records are kept
in `runtime/`.

## How do I test self-healing?

```powershell
python -m scripts.demo
```

This safely shows 8 stages: model bootstrap, prediction, drift detection,
refusal to retrain without labels, pipeline repair, retraining, rollback and
incident history. The demo uses a temporary directory; the live database
is not modified.

```powershell
python run_cycle.py
```

This runs one real monitor/diagnose/recover cycle against your local runtime.

## If sklearn gives the Windows DLL blocked error

The model will fall back automatically, but you can force the safe backend:

```powershell
$env:HELIOS_ML_BACKEND = "python"
python train.py
```

To try scikit-learn again:

```powershell
$env:HELIOS_ML_BACKEND = "auto"
python train.py
```

`backend` in training output will be `sklearn` or `pure_python`. Both are
Logistic Regression. A blocked DLL is an OS security decision; the Python
fallback avoids the need to change Windows security policy.

## Where is the code?

- `main.py` - FastAPI app entry point
- `train.py` - run training
- `helios/training.py` - load data, calculate metrics, quality gate
- `helios/training_model.py` - beginner-friendly Logistic Regression
- `helios/monitoring.py` - monitor drift and model health
- `helios/recovery.py` - root cause and self-healing rules
- `helios/registry.py` - model versions + rollback
- `scripts/demo.py` - guided demo

**Disclaimer:** bundled data and labels are generated for education, not
real bank operations. Do not use this model for real payment decisions.
