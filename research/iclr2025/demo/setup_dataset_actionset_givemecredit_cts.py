# todo: update so that this is script (like german_cts)
#  continuous features + continuous actionability constraints

import os
import sys

# fmt:off
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

import pprint

pp = pprint.PrettyPrinter(depth=2)

from src.ext import fileutils
from src.ext.data import BinaryClassificationDataset
from src.paths import *
from src.utils import check_processing_loss

from reachml import ActionSet
from reachml.constraints import *

settings = {
    "data_name": "givemecredit_cts",
    "action_set_names": ["complex_nD"],
    "check_processing_loss": False,
    "fold_id": "K05N01",
    "random_seed": 2338,
    }


def process_dataset(raw_df):
    """
    `NoSeriousDlqin2yrs`:Person did not experience 90 days past due delinquency or worse
    `Age`: Age of borrower in years
    `NumberOfDependents`: Number of dependents in family excluding themselves (spouse, children etc.)
    #
    `MonthlyIncome`: Monthly income
    #
    `DebtRatio`: Monthly debt payments, alimony, living costs divided by monthy gross income
    `RevolvingUtilizationOfUnsecuredLines`: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits
    #
    `NumberRealEstateLoansOrLines`: Number of mortgage and real estate loans including home equity lines of credit
    `NumberOfOpenCreditLinesAndLoans`: Number of Open loans (installment like car loan or mortgage) + Lines of credit (e.g. credit cards)
    #
    `NumberOfTime30-59DaysPastDueNotWorse`: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
    `NumberOfTime60-89DaysPastDueNotWorse`: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
    `NumberOfTimes90DaysLate`:Number of times borrower has been 90 days or more past due.
    """

    raw_df = pd.DataFrame(raw_df)
    raw_df = raw_df[raw_df.age >= 21]  # note: one person has age == 0
    # todo: remove these - I commented them out for now - I think it's better to keep outliers if you can
    # df = df[df.MonthlyIncome <= 30000] # tally(np.round(loaded.df['MonthlyIncome'].values/1000))
    # df = df[df.RevolvingUtilizationOfUnsecuredLines <= 10000]  #tally(np.round(loaded.df['RevolvingUtilizationOfUnsecuredLines'].values/1000))
    # df = df[df.NumberOfDependents <= 10]

    df = pd.DataFrame()
    df["NotSeriousDlqin2yrs"] = raw_df["NotSeriousDlqin2yrs"]

    # Age
    df["Age"] = raw_df["age"]
    df["NumberOfDependents"] = raw_df["NumberOfDependents"]

    # Debt Ratio
    df["DebtRatio"] = raw_df["DebtRatio"]

    # Monthly Income
    df["MonthlyIncome"] = raw_df["MonthlyIncome"]    

    # Creditline Utilization
    df["CreditLineUtilization"] = raw_df["RevolvingUtilizationOfUnsecuredLines"]

    # Real Estate Loans
    df["NumberRealEstateLoansOrLines"] = raw_df["NumberRealEstateLoansOrLines"]

    # Open Credit Lines
    df["NumberOfOpenCreditLinesAndLoans"] = raw_df["NumberOfOpenCreditLinesAndLoans"]

    # Late Payments
    df["HistoryOfLatePayment"] = np.any(raw_df[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"]].values > 0, axis = 1)

    # Late Payments (Mutable)
    df["HistoryOfLatePaymentInPast2Years"] = np.any(raw_df[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"]].values > 0, axis = 1)

    # Delinquency
    df["HistoryOfDelinquency"] = np.any(raw_df[["NumberOfTimes90DaysLate"]].values > 0, axis = 1)
    
    # Delinquency (Mutable)
    df["HistoryOfDelinquencyInPast2Years"] = np.any(raw_df[["NumberOfTimes90DaysLate"]].values > 0, axis = 1)


    # todo: consider adding MonthlyIncome == 0
    # todo: add features `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90Days+Late`
    # NOTE `NumberOfTimes90Days` is exactly what we're trying to predict
    # several options to choose from:
    #
    # 1. Create AnyLatePaymentIndicators Like:
    #  - HistoryOfLatePayment = any(`NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90Days+Late`)
    #  - HistoryOfLatePaymentOverLast2Years <- mutable variant of the above (which adds a causal constraint on Age)
    #
    # 2. Create DelinquentPaymentsIndicators Like:
    # - HistoryOfDelinquency = `NumberOfTimes90Days+Late` > 0
    # - HistoryOfDelinquencyInPastTwoYears <- mutable variant of the above (which adds a causal constraint on Age)
    # -
    # 3. Keep features in current form (as counts)
    #
    # 4. Convert to nested counts (NumberOfTimesAtLeast30DaysLate, NumberOfTimesAtLeast60DaysLast, NumberOfTimesAtLeast90DaysLate)
    return df

def complex_nD(data):

    A = ActionSet(data.X_df)
    immutable_features = [
        "Age",
        "NumberOfDependents",
        "HistoryOfLatePayment",
        "HistoryOfDelinquency",
        ]
    A[immutable_features].actionable = False
    A["Age"].step_direction = 1
    A["NumberOfDependents"].step_direction = 1

    A['HistoryOfLatePaymentInPast2Years'].lb = 0.0
    A['HistoryOfLatePaymentInPast2Years'].ub = 1.0
    A['HistoryOfLatePaymentInPast2Years'].step_direction = -1

    A['HistoryOfDelinquencyInPast2Years'].lb = 0.0
    A['HistoryOfDelinquencyInPast2Years'].ub = 1.0
    A['HistoryOfDelinquencyInPast2Years'].step_direction = -1

    A["NumberRealEstateLoansOrLines"].lb = 0
    A["NumberRealEstateLoansOrLines"].ub = 100
    A["NumberRealEstateLoansOrLines"].step_direction = -1

    A["NumberOfOpenCreditLinesAndLoans"].lb = 0
    A["NumberOfOpenCreditLinesAndLoans"].ub = 100
    A["NumberOfOpenCreditLinesAndLoans"].step_direction = -1
    A["NumberOfOpenCreditLinesAndLoans"].step_lb = -3

    A["DebtRatio"].lb = 0
    # A["DebtRatio"].step_direction = -1
    A["DebtRatio"].step_lb = -1
    A["DebtRatio"].step_ub = 1

    # A["MonthlyIncome"].step_direction = 1
    A["MonthlyIncome"].step_ub = 5000
    A["MonthlyIncome"].step_lb = -5000

    A["CreditLineUtilization"].lb = 0.0
    # A["CreditLineUtilization"].step_direction = -1
    A["CreditLineUtilization"].step_lb = -1
    A["CreditLineUtilization"].step_ub = 1

    # should be a constraint that says if loans decrease, debt ratio must also decrease

    A.constraints.add(
        constraint=DirectionalLinkage(
            names=["NumberRealEstateLoansOrLines", "NumberOfOpenCreditLinesAndLoans"],
            scales=[1.0, 1.0],
            keep_bounds=True
        )
    )

    A.constraints.add(
        constraint=DirectionalLinkage(
            names=["HistoryOfLatePaymentInPast2Years", "Age"],
            scales=[1.0, 2.0],
            keep_bounds=True
        )
    )

    A.constraints.add(
        constraint=DirectionalLinkage(
            names=["HistoryOfDelinquencyInPast2Years", "Age"],
            scales=[1.0, 2.0],
            keep_bounds=True
        )
    )


    # todo: Add Causal Constraint between RealEstateLoans -> OpenCreditLinesAndLoans
    # note: CreditLinesAndLoans = RealEstateLoans + OtherLinesOfCredit by definition
    # so changing RealEstateLoans should change CreditLines commensurately
    #   increase AnyRealEstateLoans from 0 to 1 ->
    #   increasing (AnyCreditLinesAndLoans, MultipleAnyCreditLinesAndLoans) from (0,0) -> (1,0) OR from (1,0) -> (1,1)
    # The specific constraint will depend on how you encode this information
    #   ReachabilityConstraint if you use binary
    #   DirectionalLink if you use counts

    # todo: Add Causal Constraint between HistoryOfLatePaymentOverLastTwoYears and Age
    # Changing HistoryOfLatePayment from 0 to 1 will increase Age by 1
    return A


# load raw dataset
loaded = BinaryClassificationDataset.read_csv(
        data_file=data_dir / "givemecredit" / "givemecredit"
        )

# process dataset
data_df = process_dataset(raw_df=loaded.df)

# create processed dataset
data = BinaryClassificationDataset.from_df(data_df)
data.generate_cvindices(
        strata=data.y,
        total_folds_for_cv=[1, 3, 4, 5],
        replicates=1,
        seed=settings["random_seed"],
        )
print(f"n = {data.n} points, of which {np.unique(data.X, axis = 0).shape[0]} are unique")

# train models to check change in feature processing
# may have to standardize data for logreg
if settings["check_processing_loss"]:
    df_raw = loaded.df.loc[data_df.index]
    data_raw = BinaryClassificationDataset.from_df(df_raw)
    data_raw.cvindices = data.cvindices
    comp_results = check_processing_loss(
            data,
            data_raw,
            model_type="logreg",
            rebalance=None,
            fold_id="K05N01",
            fold_num_test=5,
            )
    pp.pprint(["TRAIN", comp_results["model"]['train'], "TEST", comp_results["model"]['test']])
    pp.pprint(["RAW_TRAIN", comp_results["model_raw"]['train'], "RAW_TEST", comp_results["model_raw"]['test']])

# save dataset
fileutils.save(
    data,
    path=get_data_file(settings["data_name"], action_set_name="complex_nD"),
    overwrite=True,
    check_save=True
)

# create ActionSet
action_set = complex_nD(data)

# save ActionSet
fileutils.save(
    action_set,
    path=get_action_set_file(settings["data_name"], action_set_name="complex_nD"),
    overwrite=True,
    check_save=True
)
