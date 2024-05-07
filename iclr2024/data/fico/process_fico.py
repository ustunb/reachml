import pathlib
import pandas as pd

abs_path = pathlib.Path(__file__).parent.resolve()
RAW_PATH = abs_path / "fico_encoded.csv"
PROCESSED_PATH = abs_path / "fico_data.csv"

df = pd.read_csv(RAW_PATH)
df["RiskPerformance"] = df["RiskPerformance"].replace("Good", 1).values
df["RiskPerformance"] = df["RiskPerformance"].replace("Bad", 0).values

# # drop special characters
# df = df[~df.eq(-7).any(1)]
# df = df[~df.eq(-8).any(1)]
# df = df[~df.eq(-9).any(1)]

df.to_csv(PROCESSED_PATH, header=True, index=False)
print("results saved!")
