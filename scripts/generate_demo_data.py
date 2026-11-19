"""Regenerate synthetic, non-sensitive transaction data, deterministically.

Data is artificial and should never be described as real credit-card activity.
"""
import csv
import math
import random
from pathlib import Path


def main():
    rng = random.Random(20261119)
    noise = random.Random(144)
    rows = []
    for i in range(1600):
        amount = round(math.exp(rng.gauss(4.1, 1.08)), 2)
        hour = rng.randrange(24)
        merchant_risk = round(rng.betavariate(2, 5), 4)
        international = int(rng.random() < 0.13)
        velocity = min(50, int(rng.expovariate(1 / 3.2)))
        age = rng.randrange(1, 1600)
        # Draw and intentionally discard a random number used by the original
        # educational generator, preserving exact feature reproducibility.
        _ = rng.random()
        score = (0.0009 * amount + 2.5 * merchant_risk + 0.85 * international
                 + 0.12 * velocity - 0.0003 * age + 0.015 * (23 - hour))
        is_fraud = int((score > 2.1) != (noise.random() < 0.012))
        rows.append([amount, hour, merchant_risk, international, velocity, age, is_fraud])
    path = Path(__file__).resolve().parent.parent / "helios" / "data" / "synthetic_fraud_transactions.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["amount", "hour", "merchant_risk", "international",
                         "transactions_24h", "account_age_days", "is_fraud"])
        writer.writerows(rows)
    print(f"Created {len(rows)} synthetic transactions in {path}")


if __name__ == "__main__":
    main()
