import os
import sys

# fmt:off
import numpy as np
import pandas as pd
from src.paths import *
from src import fileutils
from src.data import BinaryClassificationDataset
from reachml import ActionSet, ReachableSetDatabase
from reachml.constraints import *
from scripts.utils import check_processing_loss, check_responsiveness, tabulate_actions, tally
import pprint
pp = pprint.PrettyPrinter(depth=2)


settings = {
    "data_name": "givemecredit",
    "action_set_names": ["simple_1D", "complex_1D", "complex_nD"],
    "check_processing_loss": True,
    "generate_reachable_sets": False,
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
    #df["Age"] = raw_df["age"]
    # df["NumberOfDependents"] = raw_df["NumberOfDependents"]

    df["Age_leq_24"] = (raw_df["age"] <= 24)
    df["Age_bt_25_to_30"] = (raw_df["age"] >= 25) & (raw_df["age"] < 30)
    df["Age_bt_30_to_59"] = (raw_df["age"] >= 30) & (raw_df["age"] <= 59)
    df["Age_geq_60"] = raw_df["age"] >= 60

    # Dependents
    df["NumberOfDependents_eq_0"] = raw_df["NumberOfDependents"] == 0
    df["NumberOfDependents_eq_1"] = raw_df["NumberOfDependents"] == 1
    df["NumberOfDependents_geq_2"] = raw_df["NumberOfDependents"] >= 2
    df["NumberOfDependents_geq_5"] = raw_df["NumberOfDependents"] >= 5

    # Debt Ratio
    df["DebtRatio_geq_1"] = raw_df["DebtRatio"] >= 1

    cash_thresholds = (3, 5, 10)  # old versions: (3, 7, 11)
    income = raw_df["MonthlyIncome"] / 1000
    for t in cash_thresholds:
        df[f"MonthlyIncome_geq_{t}K"] = income >= t

    utilization = raw_df["RevolvingUtilizationOfUnsecuredLines"]
    utilization_thresholds = (0.1, 0.2, 0.5, 0.7, 1.0)  # old versions: (3, 7, 11)
    for t in utilization_thresholds:
        df[f"CreditLineUtilization_geq_{t*100}"] = utilization >= t

    # todo: consider adding MonthlyIncome == 0
    df["AnyRealEstateLoans"] = raw_df["NumberRealEstateLoansOrLines"] >= 1
    df["MultipleRealEstateLoans"] = raw_df["NumberRealEstateLoansOrLines"] >= 2
    df["AnyCreditLinesAndLoans"] = raw_df["NumberOfOpenCreditLinesAndLoans"] >= 1
    df["MultipleCreditLinesAndLoans"] = raw_df["NumberOfOpenCreditLinesAndLoans"] >= 2

    df["HistoryOfLatePayment"] = np.any(raw_df[["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse"]].values > 0, axis = 1)
    df["HistoryOfDelinquency"] = np.any(raw_df[["NumberOfTimes90DaysLate"]].values > 0, axis = 1)
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

    # Debt Ratio
def simple_1D(data):
    A = ActionSet(data.X_df)
    immutable_features = [
        #"Age",
        #"NumberOfDependents",
        "Age_leq_24",
        "Age_bt_25_to_30",
        "Age_bt_30_to_59",
        "Age_geq_60",
        "NumberOfDependents_eq_0",
        "NumberOfDependents_eq_1",
        "NumberOfDependents_geq_2",
        "NumberOfDependents_geq_5",
        "HistoryOfLatePayment",
        "HistoryOfDelinquency",
        ]
    A[immutable_features].actionable = False
    #A["Age"].step_direction = 1
    #A["NumberOfDependents"].step_direction = 1
    #
    # todo: History Of Late Payments
    #
    A["AnyRealEstateLoans"].lb = 0.0
    A["AnyRealEstateLoans"].ub = 1.0
    A["MultipleRealEstateLoans"].lb = 0.0
    A["MultipleRealEstateLoans"].ub = 1.0
    A["AnyCreditLinesAndLoans"].lb = 0.0
    A["AnyCreditLinesAndLoans"].ub = 1.0
    A["MultipleCreditLinesAndLoans"].lb = 0.0
    A["MultipleCreditLinesAndLoans"].ub = 1.0
    return A

def complex_1D(data):
    A = simple_1D(data)
    A["DebtRatio_geq_1"].step_direction = -1
    A["AnyRealEstateLoans"].step_direction = -1
    A["MultipleRealEstateLoans"].step_direction = -1
    A["AnyCreditLinesAndLoans"].step_direction = -1
    A["MultipleCreditLinesAndLoans"].step_direction = -1
    income_variables = [s for s in A.names if "MonthlyIncome_geq_" in s]
    A[income_variables].step_direction = 1
    return A

def complex_nD(data):

    A = complex_1D(data)

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[s for s in A.names if "MonthlyIncome_geq_" in s],
                    step_direction=1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[s for s in A.names if "Utilization_geq_" in s],
                    step_direction=-1,
                    )
            )


    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=["AnyRealEstateLoans", "MultipleRealEstateLoans"],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=["AnyCreditLinesAndLoans", "MultipleCreditLinesAndLoans"],
                    step_direction=-1,
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
        data_file=data_dir / settings["data_name"] / f"{settings['data_name']}"
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
if settings["check_processing_loss"]:
    df_raw = loaded.df.loc[data_df.index]
    data_raw = BinaryClassificationDataset.from_df(df_raw)
    data_raw.cvindices = data.cvindices
    comp_results = check_processing_loss(
            data,
            data_raw,
            model_type="logreg",
            rebalance="over",
            fold_id="K05N01",
            fold_num_test=5,
            )
    pp.pprint(["TRAIN", comp_results["model"]['train'], "TEST", comp_results["model"]['test']])
    pp.pprint(["RAW_TRAIN", comp_results["model_raw"]['train'], "RAW_TEST", comp_results["model_raw"]['test']])

# create actionset
for name in settings["action_set_names"]:

    if name == "simple_1D":
        action_set = simple_1D(data)
    elif name == "complex_1D":
        action_set = complex_1D(data)
    elif name == "complex_nD":
        action_set = complex_nD(data)

    # more
    try:
        assert action_set.validate(data.X)
    except AssertionError:
        violations_df = action_set.validate(data.X, return_df=True)
        violations = ~violations_df.all(axis=0)
        violated_columns = violations[violations].index.tolist()
        print(violated_columns)
        raise AssertionError()

    # save dataset
    fileutils.save(
            data,
            path=get_data_file(settings["data_name"], action_set_name=name),
            overwrite=True,
            check_save=False,
            )

    # save actionset
    fileutils.save(
            action_set,
            path=get_action_set_file(settings["data_name"], action_set_name=name),
            overwrite=True,
            check_save=True,
            )

    # generate reachable set
    print(tabulate_actions(action_set))
    if settings["generate_reachable_sets"] and name == "complex_nD":
        db = ReachableSetDatabase(
                action_set=action_set,
                path=get_reachable_db_file(
                        data_name=settings["data_name"], action_set_name=name
                        ),
                )
        generation_stats = db.generate(data.X)
        print(generation_stats.n_points.describe())
        df = db.audit(X = data.X, clf = comp_results['model']['model'], scaler = comp_results['model']['scaler'], target = 1)
        recourse_df = df.query("yhat == False")
        n_total = len(recourse_df)
        n_responsive = recourse_df["recourse"].sum(axis=0)
        n_fixed = n_total-n_responsive
        p_fixed = n_fixed/n_total

        print(f"predictions without recourse: {p_fixed*100:1.1f}% ({n_fixed}/{n_total})")

        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', 500)

        score_tables = []
        names = [a.name for a in action_set if a.actionable]
        feature_indices = action_set.get_feature_indices([a.name for a in action_set if a.actionable])
        for status in [True, False]:
            subset_df = recourse_df.loc[recourse_df["recourse"] == status]
            reachable_scores = np.array(subset_df['reachable_scores'].to_list()).mean(axis = 0)
            reachable_scores = {n: np.round(s * 100, 1) for n, s in zip(action_set.names, reachable_scores)}
            recourse_scores = np.array(subset_df['recourse_scores'].to_list()).mean(axis = 0)
            recourse_scores = {n: np.round(s * 100, 1) for n, s in zip(action_set.names, recourse_scores)}
            fixed_scores = np.array(subset_df['immutability_scores'].to_list()).mean(axis = 0)
            fixed_scores = {n: np.round(s * 100, 1) for n, s in zip(action_set.names, fixed_scores)}
            score_tables += [
                #pd.DataFrame.from_dict(fixed_scores, orient = "index", columns = [f"fixed_{status}"]),
                pd.DataFrame.from_dict(recourse_scores, orient = "index", columns = [f"recourse_{status}"]),
                pd.DataFrame.from_dict(reachable_scores, orient = "index", columns = [f"reachable_{status}"]),
                ]

        score_df = pd.concat(score_tables, axis = 1)
        for status in [True, False]:
            score_df[f"constraint_score_{status}"] = score_df[f"reachable_{status}"] - score_df[f"recourse_{status}"]
        score_df.loc[names][["constraint_score_True", "constraint_score_False"]]

        print(tally(recourse_df["n_reachable"]))
        print(recourse_df["n_reachable"].describe())
