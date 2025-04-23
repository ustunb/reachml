import os
import sys

sys.path.append(os.getcwd())

import itertools

import numpy as np
import pandas as pd

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)

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
    "data_name": "fico",
    "action_set_names": ["simple_1D", "complex_1D", "complex_nD"],
    "check_processing_loss": False,
    "generate_reachable_sets": False,
    "fold_id": "K05N01",
    "random_seed": 2338,
}


# fmt: off
def process_dataset(raw_df):
    # raw_df=loaded.df
    raw_df = pd.DataFrame(raw_df)
    cols = [
        "RiskPerformance",
        "AverageMInFile",
        "ExternalRiskEstimate",
        "MaxDelqEver",
        "MaxDelq2PublicRecLast12M",
        "MSinceMostRecentDelq",
        "NumTotalTrades",
        "NetFractionInstallBurden",
        "NetFractionRevolvingBurden",
        "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen",
        "PercentInstallTrades",
        "NumInstallTradesWBalance",
        "NumRevolvingTradesWBalance",
        "NumBank2NatlTradesWHighUtilization",
        ]
    raw_df = raw_df[cols]

    # filter out rows without credit history
    # -9 No Bureau Record or No Investigation - i.e, no credit history/score information is available.
    no_bureau_record = raw_df.eq(-9).any(axis=1)
    raw_df = raw_df[~no_bureau_record]

    # filter out rows without usable/valid trades/inquiries
    # -8 No Usable/Valid Accounts Trades or Inquiries
    # Usable or valid for Accounts/Trades means inactive, or very old.
    # For inquiries, this can mean that the account has no “hard” inquiries, i.e. you are not actively searching for credit.
    # However, if your bank pulled your credit score to send you a pre-approved credit card, the bank’s inquiry is deemed not valid.

    # no_usable_cols = [col for col in cols if col not in ["NumInstallTradesWBalance"]]
    # no_usable_valid_trades = raw_df[no_usable_cols].eq(-8).any(1)
    # print(np.mean(no_usable_valid_trades))
    # raw_df = raw_df[~no_usable_valid_trades]

    no_usable_trades = raw_df.eq(-8).any(axis=1)
    raw_df = raw_df[~no_usable_trades]

    # -7 Condition not Met (e.g. No Inquiries, No Delinquencies)
    # No other cols have -7 except for "MaxDelqEver", "MSinceMostRecentDelq"
    df = pd.DataFrame()
    df["RiskPerformance"] = raw_df["RiskPerformance"]
    risk_estimate_thresholds = (40, 50, 60, 70, 80)
    for t in risk_estimate_thresholds:
        df[f"ExternalRiskEstimate_geq_{t}"] = (raw_df["ExternalRiskEstimate"] >= t)

    # Years of Account History
    df["YearsOfAccountHistory"] = round(raw_df["MSinceOldestTradeOpen"] / 12)

    AvgYearsInFile = round(raw_df["AverageMInFile"] / 12)
    df["AvgYearsInFile_geq_3"] = AvgYearsInFile >= 3
    df["AvgYearsInFile_geq_5"] = AvgYearsInFile >= 5
    df["AvgYearsInFile_geq_7"] = AvgYearsInFile >= 7

    # Most Recent Trade
    YearsSinceMostRecentTrade = round(raw_df["MSinceMostRecentTradeOpen"] / 12)
    df["MostRecentTradeWithinLastYear"] = YearsSinceMostRecentTrade <= 1
    df["MostRecentTradeWithinLast2Years"] = YearsSinceMostRecentTrade <= 2

    # Delinquencies
    df["AnyDerogatoryComment"] = raw_df["MaxDelqEver"] == 2
    df["AnyTrade120DaysDelq"] = raw_df["MaxDelqEver"] == 3
    df["AnyTrade90DaysDelq"] = raw_df["MaxDelqEver"] == 4
    df["AnyTrade60DaysDelq"] = raw_df["MaxDelqEver"] == 5
    df["AnyTrade30DaysDelq"] = raw_df["MaxDelqEver"] == 6

    # Recent Delinquencies
    NoRecentDelq = raw_df["MSinceMostRecentDelq"] == -7
    df["NoDelqEver"] = NoRecentDelq
    for t in [1, 3, 5]:
        df[f"YearsSinceLastDelqTrade_leq_{t}"] = NoRecentDelq & (raw_df["MSinceMostRecentDelq"] <= (t * 12))

    # Trades
    trade_thresholds = [2, 3, 5, 7]
    NumInstallTrades = raw_df["NumTotalTrades"] * (raw_df["PercentInstallTrades"] / 100)
    for t in trade_thresholds:
        df[f"NumInstallTrades_geq_{t}"] = NumInstallTrades >= t
        df[f"NumInstallTradesWBalance_geq_{t}"] = raw_df["NumInstallTradesWBalance"] >= t
        df[f"NumRevolvingTrades_geq_{t}"] = (raw_df["NumTotalTrades"] - NumInstallTrades) >= t
        df[f"NumRevolvingTradesWBalance_geq_{t}"] = raw_df["NumRevolvingTradesWBalance"] >= t

    # Delinquencies
    net_fraction_names = ["NetFractionInstallBurden", "NetFractionRevolvingBurden"]
    net_fraction_thresholds = (10, 20, 50)
    for name, t in itertools.product(net_fraction_names, net_fraction_thresholds):
        df[f"{name}_geq_{t}"] = raw_df[name] >= t

    # Utilization
    df["NumBank2NatlTradesWHighUtilizationGeq2"] = raw_df["NumBank2NatlTradesWHighUtilization"] >= 2
    df = df.astype(float)
    return df


def simple_1D(data):
    A = ActionSet(data.X_df)
    immutable_features = [a.name for a in A if "ExternalRiskEstimate" in a.name]
    immutable_features += [
        #
        "AnyDerogatoryComment",
        "NoDelqEver",
        #
        "YearsOfAccountHistory",
        #
        "AnyTrade120DaysDelq",
        "AnyTrade90DaysDelq",
        "AnyTrade60DaysDelq",
        "AnyTrade30DaysDelq",
        ]
    A[immutable_features].actionable = False
    return A


def complex_1D(data):
    A = simple_1D(data)
    A["YearsOfAccountHistory"].step_direction = 1
    A["YearsOfAccountHistory"].step_ub = 2
    A["NumBank2NatlTradesWHighUtilizationGeq2"].step_direction = -1

    for n in A.name:
        decreasing_cols = [
             "YearsSinceLastDelqTrade_leq",
             "NetFractionRevolvingBurden_geq",
             "NetFractionInstallBurden_geq",
             "NumRevolvingTradesWBalance_geq",
             "NumRevolvingTrades_geq",
             "NumInstallTradesWBalance_geq",
             "NumInstallTrades_geq"
                ]
        if any(col in n for col in decreasing_cols):
            A[n].step_direction = -1
        elif "AvgYearsInFile_geq" in n:
            A[n].step_direction = 1

    return A


def complex_nD(data):
    A = complex_1D(data)

    for threshold in [2, 3, 5, 7]:
        A.constraints.add(
                constraint=DirectionalLinkage(
                        names=[f"NumRevolvingTradesWBalance_geq_{threshold}", f"NumRevolvingTrades_geq_{threshold}"],
                        scales=[1.0, 1.0],
                        keep_bounds=True,
                        )
                )

        A.constraints.add(
                constraint=DirectionalLinkage(
                        names=[f"NumInstallTradesWBalance_geq_{threshold}", f"NumInstallTrades_geq_{threshold}"],
                        scales=[1.0, 1.0],
                        keep_bounds=True,
                        )
                )

    for t in [1, 3, 5]:
        A.constraints.add(constraint = DirectionalLinkage(
                names = [f'YearsSinceLastDelqTrade_leq_{t}', 'YearsOfAccountHistory'],
                scales = [1, -t]
                ))

    # If MostRecentTradeWithinLastYear Greater Than 1 then MostRecentTradeWithinLast2Years Equals 1
    A.constraints.add(
            constraint=ReachabilityConstraint(
                    names=["MostRecentTradeWithinLastYear", "MostRecentTradeWithinLast2Years"],
                    values=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
                    reachability=[[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 0, 1]],
                    )
            )

    # todo: add causal constraints to say
    #  "any recent trade within last n years" -> increases years_of_account_history by n years

    # todo: add mutability switches for missing variables
    # A.constraints.add(
    #     constraint=MutabilitySwitch(
    #         switch="NetFractionInstallBurden_missing",
    #         on_value=1,
    #         targets=[n for n in action_set.name if "NetFractionInstallBurden_geq" in n],
    #     )
    # )

    # Encoding Constraints
    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "YearsSinceLastDelqTrade_leq" in n],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "AvgYearsInFile_geq" in n],
                    step_direction=1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "NetFractionRevolvingBurden_geq" in n],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "NetFractionInstallBurden_geq" in n],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "NumRevolvingTradesWBalance_geq" in n],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "NumRevolvingTrades_geq" in n],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "NumInstallTradesWBalance_geq" in n],
                    step_direction=-1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[n for n in A.name if "NumInstallTrades_geq" in n],
                    step_direction=-1,
                    )
            )

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

assert not any(data.X_df.min(0) == data.X_df.max(0))
np.unique(data.X, axis = 0).shape

# train models to check change in model processing
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

for name in settings["action_set_names"]:

    if name == "simple_1D":
        action_set = simple_1D(data)
    elif name == "complex_1D":
        action_set = complex_1D(data)
    elif name == "complex_nD":
        action_set = complex_nD(data)

    # check that action set matches bounds and constraints
    print(tabulate_actions(action_set))
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
    if settings["generate_reachable_sets"] and name == "complex_nD":
        # fmt:off
        db = ReachableSetDatabase(
                action_set=action_set,
                path=get_reachable_db_file(data_name=settings["data_name"], action_set_name=name),
                )
        generation_stats = db.generate(data.X, overwrite=True)
        print(generation_stats.n_points.describe())

        predictor = extract_predictor(comp_results['model']['model'], scaler = comp_results['model']['scaler'])

        # run sample audit
        audit_df = db.audit(X = data.X, clf = comp_results['model']['model'], scaler = comp_results['model']['scaler'], target = 1)
        recourse_df = audit_df.query("yhat == False")
        n_total = len(recourse_df)
        n_responsive = recourse_df["recourse"].sum(axis=0)
        n_fixed = n_total-n_responsive
        p_fixed = n_fixed/n_total
        print(f"predictions without recourse: {p_fixed*100:1.1f}% ({n_fixed}/{n_total})")
