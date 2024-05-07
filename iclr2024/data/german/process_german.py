import pathlib
import pandas as pd

abs_path = pathlib.Path(__file__).parent.resolve()
RAW_PATH = abs_path / "german_raw.csv"
PROCESSED_PATH = abs_path / "german_data.csv"

# load data
raw_df = pd.read_csv(filepath_or_buffer = RAW_PATH,
                 names = [
                     "checking_acct",
                     "duration",
                     "credit_history",
                     "purpose",
                     "credit_amt",
                     "savings_acct",
                     "employment",
                     "installment_rate",
                     "personal_status",
                     "other_debtors",
                     "residence_since",
                     "property",
                     "age",
                     "other_installments",
                     "housing",
                     "existing_credits",
                     "job",
                     "liable_persons",
                     "telephone",
                     "foreign_worker",
                     "credit_risk",
                     ])

#
df = pd.DataFrame()

# outcome
df["is_credit_risk_good"] = (raw_df["credit_risk"] == 1)

# age
df["age"] = raw_df["age"]

# gender and marital status
df["male_divorced"] = raw_df["personal_status"] == "A91"
df["male_single"] = raw_df["personal_status"] == "A93"
df["male_married"] = raw_df["personal_status"] == "A94"
df["female_divorced/married"] = raw_df["personal_status"] == "A92"
df["female_single"] = raw_df["personal_status"] == "A95"

# foreign worker
df["is_foreign_worker"] = raw_df["foreign_worker"] == "A201"

# dependents
df["liable_persons"] = raw_df["liable_persons"]

df["residence_since"] = raw_df["residence_since"]

# housing
df["housing_is_renter"] = raw_df["housing"] == "A151"
df["housing_is_owner"] = raw_df["housing"] == "A152"
df["housing_is_free"] = raw_df["housing"] == "A153"

# contact information
df["has_telephone"] = raw_df["telephone"] == "A192"

# real estate ownership
df["real_estate"] = raw_df["property"] == "A121"
df["building_society"] = raw_df["property"] == "A122"
df["car_or_other"] = raw_df["property"] == "A123"
df["unknown_property"] = raw_df["property"] == "A124"

# job
df["job_is_unemployed"] = (raw_df["job"] == "A171")
df["job_is_unskilled"] = (raw_df["job"] == "A172")
df["job_is_skilled"] = (raw_df["job"] == "A173")
df["job_is_management"] = (raw_df["job"] == "A174")

# loan terms
df["duration"] = raw_df["duration"]
df["credit_amt"] = raw_df["credit_amt"]
df["installment_rate"] = raw_df["installment_rate"]

# loan purpose
df["purpose_is_car_new"] = raw_df["purpose"] == "A40"
df["purpose_is_car_used"] = raw_df["purpose"] == "A41"
df["purpose_is_furniture"] = raw_df["purpose"] == "A42"
df["purpose_is_radio/tv"] = raw_df["purpose"] == "A43"
df["purpose_is_domestic_appliances"] = raw_df["purpose"] == "A44"
df["purpose_is_repairs"] = raw_df["purpose"] == "A45"
df["purpose_is_education"] = raw_df["purpose"] == "A46"
df["purpose_is_vacation"] = raw_df["purpose"] == "A47"
df["purpose_is_retraining"] = raw_df["purpose"] == "A48"
df["purpose_is_business"] = raw_df["purpose"] == "A49"
df["purpose_is_other"] = raw_df["purpose"] == "A410"

# credit history
df["credit_history_no_credits_taken"] = raw_df["credit_history"] == "A30"
df["credit_history_all_credits_paid_duly"] = raw_df["credit_history"] == "A31"
df["credit_history_all_credits_paid_till_now"] = raw_df["credit_history"] == "A32"
df["credit_history_delay_in_paying_credits"] = raw_df["credit_history"] == "A33"
df["credit_history_critical_credit_history"] = raw_df["credit_history"] == "A34"

# credits
df["existing_credits"] = raw_df["existing_credits"]

# installments
df["has_bank_installments"] = raw_df["other_installments"] == "A141"
df["has_store_installments"] = raw_df["other_installments"] == "A142"
df["has_none_installments"] = raw_df["other_installments"] == "A143"

# checking account
df["checking_acct_le_0"] = raw_df["checking_acct"] == "A11"
df["checking_acct_bt_0_200"] = raw_df["checking_acct"] == "A12"
df["checking_acct_ge_200"] = raw_df["checking_acct"] == "A13"
df["no_checking_acct"] = raw_df["checking_acct"] == "A14"

# savings account
df["savings_acct_le_100"] = raw_df["savings_acct"] == "A61"
df["savings_acct_bt_100_499"] = raw_df["savings_acct"] == "A62"
df["savings_acct_bt_500_999"] = raw_df["savings_acct"] == "A63"
df["savings_acct_ge_1000"] = raw_df["savings_acct"] == "A64"
df["no_savings_acct"] = raw_df["savings_acct"] == "A65"

# employment
df["unemployed"] = raw_df["employment"] == "A71"
df["employed_le_1_yr"] = raw_df["employment"] == "A72"
df["employed_bt_1_4_yr"] = raw_df["employment"] == "A73"
df["employed_bt_4_7_yr"] = raw_df["employment"] == "A74"
df["employed_geq_7_yr"] = raw_df["employment"] == "A75"

# loan
df["no_other_debtors"] = raw_df["other_debtors"] == "A101"
df["co-applicant"] = raw_df["other_debtors"] == "A102"
df["guarantor"] = raw_df["other_debtors"] == "A103"


df = df.dropna()

y = df["is_credit_risk_good"]
cur_X = df.drop(columns=["is_credit_risk_good"])
df = pd.concat([y, cur_X], axis=1, join="inner")
df.to_csv(PROCESSED_PATH, header=True, index=False)
print("processed german dataset")
