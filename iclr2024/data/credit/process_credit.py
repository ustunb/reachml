import pathlib

import pandas as pd

from scripts import utils as utils

abs_path = pathlib.Path(__file__).parent.resolve()
RAW_PATH = abs_path / "credit_raw.csv"
PROCESSED_PATH = abs_path / "credit_data.csv"

df = pd.read_csv(RAW_PATH)
df = df.drop(["id"], axis=1)

NTD_to_USD = 30.48  # see https://www.poundsterlinglive.com/bank-of-england-spot/historical-spot-exchange-rates/usd/USD-to-TWD-2023
monetary_features = list(
    filter(
        lambda x: ("bill_amt" in x) or ("pay_amt" in x) or ("limit_bal" in x),
        df.columns,
    )
)
df[monetary_features] = (
    df[monetary_features].applymap(lambda x: x / NTD_to_USD).round(-1).astype(int)
)

df["is_male"] = utils.filter_cond(df["sex"] == 1)

df["education_graduate"] = 0
df["education_bachelors"] = 0
df["education_hs"] = 0

df["education_graduate"][df["education"] == 1] = 1  # Graduate
df["education_bachelors"][df["education"] == 2] = 1  # University
df["education_hs"][df["education"] == 3] = 1  # HS

df["martial_status_married"] = utils.filter_cond(df["marriage"] == 1)
df["martial_status_single"] = utils.filter_cond(df["marriage"] == 2)
df["martial_status_other"] = utils.filter_cond(df["marriage"] == 3)

df["pay_0_is_paid_duly"] = utils.filter_cond(df["pay_0"] == -1)
df["pay_0_was_refunded"] = utils.filter_cond(df["pay_0"] == -2)

df["pay_2_is_paid_duly"] = utils.filter_cond(df["pay_2"] == -1)
df["pay_2_was_refunded"] = utils.filter_cond(df["pay_2"] == -2)

df["pay_3_is_paid_duly"] = utils.filter_cond(df["pay_3"] == -1)
df["pay_3_was_refunded"] = utils.filter_cond(df["pay_3"] == -2)

df["pay_4_is_paid_duly"] = utils.filter_cond(df["pay_4"] == -1)
df["pay_4_was_refunded"] = utils.filter_cond(df["pay_4"] == -2)

df["pay_5_is_paid_duly"] = utils.filter_cond(df["pay_5"] == -1)
df["pay_5_was_refunded"] = utils.filter_cond(df["pay_5"] == -2)

df["pay_6_is_paid_duly"] = utils.filter_cond(df["pay_6"] == -1)
df["pay_6_was_refunded"] = utils.filter_cond(df["pay_6"] == -2)

df = df.rename(
    columns={
        "pay_0": "pay_0_months_delayed",
        "pay_1": "pay_1_months_delayed",
        "pay_2": "pay_2_months_delayed",
        "pay_3": "pay_3_months_delayed",
        "pay_4": "pay_4_months_delayed",
        "pay_5": "pay_5_months_delayed",
        "pay_6": "pay_6_months_delayed",
    }
)

df = df.replace(-1, 0)
df = df.replace(-2, 0)

df = df.drop(columns=["sex", "education", "marriage"])

df = df.dropna()

y = df["will_default_next_month"]
X = df.drop(columns=["will_default_next_month"])

new_df = pd.concat([y, X], axis=1, join="inner")


new_df.to_csv(PROCESSED_PATH, header=True, index=False)
print("results saved!")
