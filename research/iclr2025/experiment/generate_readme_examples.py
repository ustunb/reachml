# experiment/generate_readme_examples.py

import argparse
import numpy as np
import pandas as pd

CONTRIBUTION_LIMIT = 6500


def generate_readme_examples(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n):
        age = rng.integers(18, 81)  # 18–80
        employed = rng.integers(0, 2)

        if employed:
            monthly_income = rng.uniform(3000, 15000)
        else:
            monthly_income = rng.uniform(0, 1000)

        portfolio_allocation = rng.uniform(0.0, 1.0)

        # retirement_savings tied to age & employment
        if age >= 73 or not employed:
            # capped / plateaued
            retirement_savings = rng.uniform(0.7, 1.1) * (73 * CONTRIBUTION_LIMIT)
        else:
            base = age * CONTRIBUTION_LIMIT
            noise = rng.uniform(-0.2, 0.2) * CONTRIBUTION_LIMIT
            retirement_savings = max(0, base + noise)

        # provisional label from a simple score
        score = (
            0.4 * (monthly_income / 15000)
            + 0.3 * employed
            + 0.2 * (retirement_savings / (80 * CONTRIBUTION_LIMIT))
            + 0.1 * (1 - abs(portfolio_allocation - 0.6))
        )

        rows.append(
            dict(
                loanApproved=False,  # will set later
                age=age,
                employed=employed,
                monthly_income=monthly_income,
                retirement_savings=retirement_savings,
                portfolio_allocation=portfolio_allocation,
                _score=score,
            )
        )

    df = pd.DataFrame(rows)

    # turn score into balanced labels: median split
    threshold = df["_score"].median()
    df["loanApproved"] = df["_score"] >= threshold
    df = df.drop(columns=["_score"])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="readme_example_data.csv",
        help="where to write the synthetic dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = generate_readme_examples(n=args.n, seed=args.seed)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
