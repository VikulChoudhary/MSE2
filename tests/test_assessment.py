import csv
import json

import pytest
from helios.config import Settings
from scripts.make_new_batch import main as generate
from assessment import load_and_validate, run


def test_assessment_generates_artifacts_and_preserves_champion(tmp_path, monkeypatch):
    from scripts import make_new_batch
    monkeypatch.setattr(make_new_batch, 'OUTPUT', tmp_path / 'batch.csv')
    generate()
    settings = Settings(data_dir=tmp_path / 'runtime', auto_monitor=False)
    report = run(tmp_path / 'batch.csv', settings=settings)
    assert (tmp_path / 'runtime/assessment/candidate_model.json').exists()
    assert (tmp_path / 'runtime/assessment/assessment_report.txt').exists()
    assert report['champion_version'] == report['champion_version_after']
    assert report['new_batch_evaluation_rows'] > 0
    assert report['new_batch_training_rows'] > 0
    assert report['quality_gate']['decision'] in ('ELIGIBLE_FOR_PROMOTION', 'KEEP_CHAMPION')
    assert json.loads((tmp_path / 'runtime/assessment/assessment_report.json').read_text())['quality_gate']['auto_promotion'] is False


def test_assessment_rejects_missing_values_and_duplicate_ids(tmp_path):
    columns = ['transaction_id','amount','hour','merchant_risk','international','transactions_24h','account_age_days','is_fraud']
    path = tmp_path / 'bad.csv'
    rows = [[f'id{i}', 100+i, 12, 0.2, 0, 3, 100, i%2] for i in range(24)]
    rows[4][1] = ''
    with path.open('w', newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)
    with pytest.raises(ValueError, match='Missing value'):
        load_and_validate(path)
    rows[4][1]=104
    rows[4][0]=rows[3][0]
    with path.open('w', newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)
    with pytest.raises(ValueError, match='Duplicate transaction_id'):
        load_and_validate(path)
