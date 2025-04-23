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
from src.ext.training import extract_predictor
from src.paths import *
from src.utils import check_processing_loss, tabulate_actions

from reachml import ActionSet, ReachableSetDatabase
from reachml.constraints import *

settings = {
    "data_name": "givemecredit",
    #"action_set_names": ["simple_1D", "complex_1D", "complex_nD"],
    "action_set_names": ["complex_nD"],
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
            rebalance=None,
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

        predictor = extract_predictor(comp_results['model']['model'], scaler = comp_results['model']['scaler'])

        # run sample audit
        audit_df = db.audit(X = data.X, clf = comp_results['model']['model'], scaler = comp_results['model']['scaler'], target = 1)

        # print audit results
        recourse_df = audit_df.query("yhat == False")
        n_total = len(recourse_df)
        n_responsive = recourse_df["recourse"].sum(axis=0)
        n_fixed = n_total-n_responsive
        p_fixed = n_fixed/n_total
        print(f"predictions without recourse: {p_fixed*100:1.1f}% ({n_fixed}/{n_total})")
        print('reachable point distribution')
        print(recourse_df["n_reachable"].describe())
        print('reachable point distribution')
        pp.pprint(tally(recourse_df['n_feasible']))

        # # analysis for predictions without recourse
        #
        # clf = comp_results['model']['model']
        # scaler = comp_results['model']['scaler']
        # weights = pd.DataFrame({
        #     'names': data.names.X,
        #     'coefficient': np.true_divide(clf.coef_,  scaler.scale_).flatten()
        #     }).set_index("names")
        # n_feasible_threshold = 0
        #
        # fixed_idx = audit_df.query("yhat == False").query(f"n_feasible == {n_feasible_threshold}").index.tolist()
        # fixed_features = data.X_df.loc[fixed_idx].T
        # fixed_features["weights"] = weights
        # fixed_features = fixed_features[["weights"] + fixed_features.columns[0:-1].tolist()]
        #
        # fixed_dfs = {}
        # for i in fixed_idx:
        #     x = data.X[i, :]
        #     fixed_dfs[i] = db[x].describe(predictor = predictor, target = 1)
        #     #fixed_dfs[i]['w'] = weights
        #     #fixed_dfs[i] = fixed_dfs[i][["x", "w", "n_total", "n_target"]]
        #     print("="*70)
        #     print(f"x[{i}] | n_feasible={n_feasible_threshold}")
        #     print("-"*70)
        #     print(fixed_dfs[i])
        #     print("="*70)
        #

        # # subpopulation analysis
        # n_feasible_min = 1
        # n_feasible_max = 10
        # subset_idx = (audit_df.query("yhat == False").
        #              query(f"n_feasible >= {n_feasible_min}").
        #              query(f"n_feasible <= {n_feasible_max}").
        #              index.tolist())
        # other_idx = set(audit_df.query("yhat == False").index.to_list()) - set(subset_idx)
        # score_df = pd.DataFrame.from_dict(
        #         {'names': action_set.names,
        #          'recourse_subset': np.array(audit_df.loc[subset_idx]['recourse_scores'].to_list()).mean(axis = 0),
        #          'recourse_other': np.array(audit_df.loc[other_idx]['recourse_scores'].to_list()).mean(axis = 0)
        #             }).set_index('names')
        # score_df['diff'] = (score_df['recourse_other'] - score_df['recourse_subset'])/score_df['recourse_other']
        # print(score_df)
