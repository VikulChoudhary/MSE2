# HELIOS architecture | Synthetic credit-card fraud detection

## Data provenance

The included CSV contains **1,600 generated, fictional transactions**, with no real payment identifiers. `python -m scripts.generate_demo_data` deterministically reproduces it. Labels follow a synthetic rule (weighted amount, merchant risk, international status, transaction velocity, age, hour with 1.2% label noise). This rule deliberately makes a learnable **teaching exercise**, not a realistic fraud distribution.

## Major components

1. `training.py` parses the CSV and makes a deterministic stratified 75/25 holdout split, preserving fraud prevalence.
2. `training_model.py` learns a **class-weighted, standard-library logistic regression** classifier, stores feature means/scales, and uses a demo decision threshold of 0.80.
3. `registry.py` saves JSON model artifacts and an atomic `champion.json` active-version pointer. `database.py` records metadata, deployments, experiments, predictions, feedback and incidents.
4. `api.py` receives and validates the six-feature transaction request, exposes control-plane routes, and serves `static/index.html`.
5. `monitoring.py` records rolling prediction patterns and latency, PSI drift against training feature profiles, and metrics computed from authenticated outcomes when available.
6. `recovery.py` performs deterministic diagnosis and a bounded set of recovery actions; it refuses to claim successful resolution without verification.
7. `service.py` initializes the baseline model, recovers a broken startup artifact if a verified earlier version is available, and runs the periodic recovery loop.

## Lifecycle

Synthetic CSV -> split -> train -> evaluate (accuracy >= 0.90, fraud recall >= 0.65, nonregression within 0.03) -> register and activate -> API inference -> monitoring -> diagnose -> [repair, retrain, rollback, or request labels/review] -> verify -> persist incident result.

## Security / correctness limits

- **No real money decisions:** Example features and performance do not generalize to real transactions.
- No untrusted remote shell execution or arbitrary infrastructure repair.
- Authenticated admin endpoints use one environment-configured key suitable only for localhost demo; no enterprise RBAC.
- Retained feedback is assumed trusted; real systems require delayed label auditing, leakage prevention, and privacy controls.
- Holdout accuracy and fraud recall are measured only on a small fixed synthetic holdout; additional temporal evaluation would be required for realistic usage.
- Drift may remain after retraining; it is reported as needing attention rather than falsely resolved.
