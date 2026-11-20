"""Tasks A-D: validate a new batch, evaluate champion, retrain candidate, gate.

Run: python assessment.py
Artifacts are written to runtime/assessment; the champion is NEVER replaced.
"""
import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from helios.config import Settings
from helios.service import HeliosService
from helios.training import FEATURE_NAMES, dataset, Trainer
from helios.training_model import FraudModel

ROOT = Path(__file__).resolve().parent
REQUIRED = [*FEATURE_NAMES, 'is_fraud']


def load_and_validate(path):
    """Task A: check schema, blanks, duplicate IDs/rows and valid target labels."""
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError('Batch CSV has no header')
        missing_columns = sorted(set(REQUIRED) - set(reader.fieldnames))
        if missing_columns:
            raise ValueError(f'Missing required columns: {missing_columns}')
        rows = list(reader)
    if len(rows) < 20:
        raise ValueError('Need at least 20 labeled new transactions')
    ids, signatures, features, labels = set(), set(), [], []
    for index, row in enumerate(rows, 2):
        if any(row.get(key) is None or not row[key].strip() for key in REQUIRED):
            raise ValueError(f'Missing value on CSV line {index}')
        if 'transaction_id' in row:
            identifier = row['transaction_id'].strip()
            if not identifier:
                raise ValueError(f'Missing transaction_id on line {index}')
            if identifier in ids:
                raise ValueError(f'Duplicate transaction_id: {identifier}')
            ids.add(identifier)
        try:
            vector = [float(row[name]) for name in FEATURE_NAMES]
        except ValueError as exc:
            raise ValueError(f'Non-numeric feature on line {index}') from exc
        if not all(math.isfinite(x) for x in vector):
            raise ValueError(f'Invalid infinite or NaN value on line {index}')
        if vector[0] < 0 or not 0 <= vector[1] <= 23 or not 0 <= vector[2] <= 1 or vector[3] not in (0, 1) or vector[4] < 0 or vector[5] < 0:
            raise ValueError(f'Feature outside expected range on line {index}')
        if row['is_fraud'].strip() not in ('0', '1'):
            raise ValueError(f'Target must be 0 or 1 on line {index}')
        label = int(row['is_fraud'])
        signature = (tuple(vector), label)
        if signature in signatures:
            raise ValueError(f'Duplicate transaction features and target on line {index}')
        signatures.add(signature)
        features.append(vector)
        labels.append(label)
    if set(labels) != {0, 1}:
        raise ValueError('New batch must contain both legitimate and fraud examples')
    distribution = dict(sorted(Counter(labels).items()))
    print('TASK A - NEW BATCH VALIDATED')
    print('Rows:', len(rows), '| Target distribution:', distribution)
    print('Required columns, missing values and duplicates: PASS')
    return features, labels, distribution


def split_new_batch(features, labels):
    """Keep every fourth example per label for *unseen* evaluation.

    Other new examples can be used to retrain; avoids evaluating on examples
    the candidate just learned from. Both models use the SAME evaluation slice.
    """
    groups = {0: [], 1: []}
    for x, y in zip(features, labels):
        groups[y].append((x, y))
    train, evaluation = [], []
    for label in (0, 1):
        if len(groups[label]) < 4:
            raise ValueError('At least four examples of each class required')
        for i, item in enumerate(groups[label]):
            (evaluation if i % 4 == 0 else train).append(item)
    return ([x for x, y in train], [y for x, y in train],
            [x for x, y in evaluation], [y for x, y in evaluation])


def log_mlflow(settings, report, candidate_file):
    """Optional: use existing HELIOS MLflow tracking URI if configured."""
    if not settings.mlflow_tracking_uri:
        return 'Not configured (optional)'
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment('HELIOS-assessment-fraud')
        with mlflow.start_run(run_name='new-batch-candidate'):
            mlflow.log_param('batch', report['new_batch_file'])
            mlflow.log_param('champion_version', report['champion_version'])
            mlflow.log_param('candidate_backend', report['candidate_backend'])
            for group in ('champion_metrics', 'candidate_metrics'):
                mlflow.log_metrics({f'{group}.{k}': v for k, v in report[group].items()
                                    if isinstance(v, (int, float))})
            mlflow.log_artifact(str(candidate_file))
            mlflow.set_tag('quality_gate', report['quality_gate']['decision'])
        return 'Logged to MLflow'
    except Exception as exc:
        return f'Optional MLflow logging skipped: {exc}'


def run(batch, output_dir=None, settings=None):
    settings = settings or Settings()
    output = Path(output_dir) if output_dir else settings.data_dir / 'assessment'
    output.mkdir(parents=True, exist_ok=True)

    # Task A: validate the entire new batch.
    new_x, new_y, distribution = load_and_validate(batch)
    new_train_x, new_train_y, new_eval_x, new_eval_y = split_new_batch(new_x, new_y)

    # Task B: evaluate the saved champion model on unseen, labeled new data.
    service = HeliosService(settings)
    service.bootstrap()
    champion, champion_version = service.registry.active_model()
    champion_metrics = Trainer.evaluate(champion, new_eval_x, new_eval_y)
    print('TASK B - CHAMPION MODEL', champion_version, champion_metrics)

    # Task C: existing training rows + new training slice, same preprocessing.
    existing_x, _existing_holdout_x, existing_y, _existing_holdout_y = dataset()
    combined_x = existing_x + new_train_x
    combined_y = existing_y + new_train_y
    candidate = FraudModel().fit(combined_x, combined_y)
    candidate_metrics = Trainer.evaluate(candidate, new_eval_x, new_eval_y)
    candidate_path = output / 'candidate_model.json'
    candidate_path.write_text(json.dumps(candidate.to_dict(), indent=2), encoding='utf-8')
    print('TASK C - CANDIDATE SAVED:', candidate_path)

    # Task D: both models compared on the same unseen labeled evaluation rows.
    gates = {
        'min_accuracy_90_percent': candidate_metrics['accuracy'] >= settings.candidate_accuracy_threshold,
        'min_fraud_recall_65_percent': candidate_metrics['fraud_recall'] >= settings.min_fraud_recall,
        'accuracy_not_worse_than_champion_by_3_points': candidate_metrics['accuracy'] >= champion_metrics['accuracy'] - settings.max_allowed_regression,
        'fraud_recall_not_below_champion': candidate_metrics['fraud_recall'] >= champion_metrics['fraud_recall'],
    }
    eligible = all(gates.values())
    decision = 'ELIGIBLE_FOR_PROMOTION' if eligible else 'KEEP_CHAMPION'
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'new_batch_file': str(batch),
        'new_batch_size': len(new_y),
        'target_distribution': {'legitimate_0': distribution.get(0, 0), 'fraud_1': distribution.get(1, 0)},
        'new_batch_training_rows': len(new_train_y),
        'new_batch_evaluation_rows': len(new_eval_y),
        'existing_training_rows': len(existing_y),
        'combined_training_rows': len(combined_y),
        'champion_version': champion_version,
        'champion_metrics': champion_metrics,
        'candidate_backend': candidate.backend,
        'candidate_metrics': candidate_metrics,
        'candidate_file': str(candidate_path),
        'quality_gate': {'checks': gates, 'decision': decision,
                         'metric_justification': 'Fraud recall is essential because missing a fraudulent transaction is costly; accuracy and non-regression avoid promoting a broadly weaker model.',
                         'auto_promotion': False},
    }
    report['mlflow'] = log_mlflow(settings, report, candidate_path)
    report['champion_version_after'] = service.registry.active_version()
    assert report['champion_version_after'] == champion_version, 'Champion unexpectedly changed!'
    report_path = output / 'assessment_report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = [
        'HELIOS MODEL MONITORING & RETRAINING ASSESSMENT',
        'TASK A - New batch validation: PASS',
        f'Batch rows: {len(new_y)}; legitimate: {distribution.get(0,0)}; fraud: {distribution.get(1,0)}',
        f'New batch train/evaluation: {len(new_train_y)} / {len(new_eval_y)} (held-out to prevent leakage)',
        f'TASK B - Champion model version: {champion_version}',
        f'Champion metrics: {json.dumps(champion_metrics)}',
        f'TASK C - Candidate trained on {len(combined_y)} rows, artifact: {candidate_path}',
        f'Candidate metrics: {json.dumps(candidate_metrics)}',
        f'TASK D - Gate checks: {json.dumps(gates)}',
        f'Decision: {decision}; champion remains version {champion_version}',
        'Metric choice: prioritize fraud recall; require adequate accuracy and no degradation versus champion.',
        f'MLflow: {report["mlflow"]}',
    ]
    (output / 'assessment_report.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('TASK D - QUALITY GATE:', decision)
    print('Champion unchanged: version', report['champion_version_after'])
    print('REPORT:', report_path)
    print('TASK E - Show candidate_model.json, assessment_report.json, and these terminal logs.')
    return report


def main():
    parser = argparse.ArgumentParser(description='HELIOS assessment A-E')
    parser.add_argument('--batch', type=Path, default=ROOT / 'data' / 'new_fraud_batch.csv')
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    if not args.batch.exists():
        raise SystemExit(f'Batch not found: {args.batch}. Run python -m scripts.make_new_batch first.')
    run(args.batch, args.output_dir)


if __name__ == '__main__':
    main()
