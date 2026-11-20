"""Generate a NEW reproducible synthetic fraud batch for the MLOps assessment.

The generated records are fictional and independent of the original training CSV.
"""
import csv
import math
import random
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / 'data' / 'new_fraud_batch.csv'


def main():
    rng = random.Random(20261120)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(240):
        amount = round(math.exp(rng.gauss(4.28, 1.1)), 2)
        hour = rng.randrange(24)
        merchant_risk = round(rng.betavariate(2.3, 4.6), 4)
        international = int(rng.random() < 0.18)
        velocity = min(50, int(rng.expovariate(1 / 3.7)))
        age = rng.randrange(1, 1600)
        score = (0.0009 * amount + 2.5 * merchant_risk + 0.85 * international
                 + 0.12 * velocity - 0.0003 * age + 0.015 * (23 - hour))
        label = int((score > 2.1) != (rng.random() < 0.012))
        rows.append([f'new-{i:04d}', amount, hour, merchant_risk,
                     international, velocity, age, label])
    with OUTPUT.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['transaction_id', 'amount', 'hour', 'merchant_risk',
                         'international', 'transactions_24h', 'account_age_days', 'is_fraud'])
        writer.writerows(rows)
    print(f'Generated {len(rows)} fictional new transactions at {OUTPUT}')


if __name__ == '__main__':
    main()
