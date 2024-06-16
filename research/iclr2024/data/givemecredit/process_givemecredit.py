import pathlib
import pandas as pd

abs_path = pathlib.Path(__file__).parent.resolve()
RAW_PATH = abs_path / "givemecredit_raw.csv"
PROCESSED_PATH = abs_path / "givemecredit_data.csv"

df = pd.read_csv(RAW_PATH)
df["SeriousDlqin2yrs"] = df["SeriousDlqin2yrs"].replace({0: 1, 1: 0})
df = df.rename(columns={"SeriousDlqin2yrs": "NotSeriousDlqin2yrs"})

df.to_csv(PROCESSED_PATH, header=True, index=False)
print("results saved!")
