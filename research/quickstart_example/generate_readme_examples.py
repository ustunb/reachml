import numpy as np
import pandas as pd

contribution_limit = 6500

def generate_readme_examples(n=100, seed=0):
    rng = np.random.default_rng(seed)
    examples = []

    for _ in range(n):
        # 1. Basic demographics
        age = rng.integers(18, 81)  # 18–80 inclusive
        employed = rng.integers(0, 2)  # 0 or 1

        # 2. Monthly income: if unemployed, small or zero
        if employed:
            monthly_income = rng.uniform(3000, 15000)
        else:
            monthly_income = rng.uniform(0, 1000)

        # 3. Portfolio allocation between 0 and 1
        portfolio_allocation = rng.uniform(0.0, 1.0)

        # 4. Retirement savings, obeying linkage and stop conditions
        #    Base: age * contribution_limit with some variation
        if age >= 73 or not employed:
            # No further contributions, maybe plateau
            retirement_savings = rng.uniform(
                0.8, 1.2
            ) * (73 * contribution_limit)
        else:
            # Allow contributions up to age * limit
            base = age * contribution_limit
            # Add noise up to ±20% of current limit
            noise = rng.uniform(-0.2, 0.2) * contribution_limit
            retirement_savings = max(0, base + noise)

        loanApproved = bool(rng.uniform() < 0.7)  # arbitrary placeholder label

        examples.append(
            dict(
                loanApproved=loanApproved,
                age=age,
                employed=employed,
                monthly_income=monthly_income,
                retirement_savings=retirement_savings,
                portfolio_allocation=portfolio_allocation,
            )
        )

    return pd.DataFrame(examples)


# Example usage
if __name__ == "__main__":
    df = generate_readme_examples(20, seed=42)
    print(df.head())
    df.to_csv("readme_example_data.csv", index=False)
