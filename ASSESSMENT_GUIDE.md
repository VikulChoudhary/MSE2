# HELIOS: MLOps Practical Assessment (10 Marks)

This is a small, standalone **A–E** assessment workflow added to the existing HELIOS fraud-detection app. It keeps the old FastAPI dashboard, tests, Docker and GitHub Actions intact.

## Windows PowerShell commands

From the `HELIOS` project folder with your virtual environment active:

```powershell
python -m scripts.make_new_batch
python assessment.py
python -m pytest -q
```

For your own CSV batch, use `python assessment.py --batch "path/to/new.csv"`. Supply columns `amount,hour,merchant_risk,international,transactions_24h,account_age_days,is_fraud`; optionally `transaction_id`. `is_fraud` must contain 0 and 1. **Use labeled data for the evaluation.**

Evidence (after execution) is saved under `runtime/assessment/`:

- `assessment_report.json`: validation size/distribution, champion and candidate metrics, gate checks and decision
- `assessment_report.txt`: short printable summary
- `candidate_model.json`: separately saved *candidate* weights, not deployed
- Terminal output: all A–E stages

## Requirements mapping

| Task | Implementation | Evidence |
|---|---|---|
| A: Load and validate new data | `assessment.py:load_and_validate` | Required columns, missing values, duplicate IDs/rows, row count, label distribution |
| B: Evaluate existing model | `Registry.active_model()` and `Trainer.evaluate()` | Champion metrics on the new batch's held-out subset, JSON report |
| C: Retrain | `dataset()` + new training subset → `FraudModel.fit()` | Independent `candidate_model.json` and optional MLflow tracking |
| D: Quality gate | Accuracy >=90%, recall >=65%, accuracy within 3 points of champion, recall >= champion | `ELIGIBLE_FOR_PROMOTION` or `KEEP_CHAMPION`; recorded in report |
| E: Demonstrate | Run commands and inspect files | Terminal + JSON/TXT report + candidate artifact |

**Why split the new batch?** We use approximately 75% of the NEW batch to retrain, and 25% exclusively for a fair comparison. If we trained on all of it and then reported accuracy on that same data, the candidate would have an unfair advantage (data leakage). Both champion and candidate are evaluated on the same unseen new-data subset.

**What is the champion?** The previously trained deployed model (e.g. v3 from your local runtime). A candidate is an *alternative* model trained on existing training data plus fresh labeled examples. The candidate is saved separately and is **never automatically promoted** by `assessment.py`.

**Why fraud recall?** False negatives are real frauds that the model missed. Fraud recall specifically measures how many fraud cases are caught. We also require accuracy and non-regression to avoid replacing a strong champion with a broadly worse model.

**MLflow:** Only used if you separately installed `requirements-mlflow.txt` and configured `HELIOS_MLFLOW_TRACKING_URI` to a working local server. Otherwise it is optional and the JSON report always records evidence.

**Dataset warning:** `data/new_fraud_batch.csv` is artificially generated, not real bank data. No real financial claims should be made from its metrics. Existing bundled `helios/data/synthetic_fraud_transactions.csv` is the historical data.

## 45-second viva explanation

> Task A validates a separately collected and labeled new transaction batch, checking schema, missing values, duplicate IDs and class counts. Task B loads the currently deployed champion from HELIOS's model registry and evaluates it using accuracy, precision, recall and F1 on the new batch's held-out samples. In Task C, I combine the historical training set with the new batch's training portion, reuse the same fraud model and feature standardization process, and save a new candidate as JSON. Task D compares both models on the same unseen examples and writes a quality-gate decision. It never overwrites the champion automatically. Task E is demonstrated by the command output, candidate artifact and evaluation reports.
