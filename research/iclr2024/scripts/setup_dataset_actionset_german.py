# fmt: off
import os
import sys

import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


from src.paths import *
from src import fileutils
from src.data import BinaryClassificationDataset
from reachml import ActionSet, ReachableSetDatabase
from reachml.constraints import *

import pprint as pp
from scripts.utils import check_processing_loss, tabulate_actions, tally, tally_predictions
from src.training import extract_predictor

settings = {
    "data_name": "german",
    "action_set_names": ["simple_1D", "complex_1D", "complex_nD"],
    "check_processing_loss": True,
    "generate_reachable_sets": False,
    "fold_id": "K05N01",
    "random_seed": 2338,
    }

def process_dataset(raw_df):

    df = pd.DataFrame()
    df["IsCreditRiskGood"] = raw_df["is_credit_risk_good"]

    df["Age"] = raw_df["age"]
    # df["Age_leq_24"] = (raw_df["age"] <= 24)
    # df["Age_bt_25_to_30"] = (raw_df["age"] >= 25) & (raw_df["age"] < 30)
    # df["Age_bt_30_to_59"] = (raw_df["age"] >= 30) & (raw_df["age"] <= 59)
    # df["Age_geq_60"] = raw_df["age"] >= 60

    df["Male"] = raw_df[["male_divorced", "male_single", "male_married"]].any(axis=1)
    df["Single"] = raw_df[["male_single", "female_single"]].any(axis=1)

    df["ForeignWorker"] = raw_df["is_foreign_worker"]
    df["YearsAtResidence"] = raw_df["residence_since"]
    df["LiablePersons"] = raw_df["liable_persons"]

    df["Housing_is_Renter"] = raw_df["housing_is_renter"]
    df["Housing_is_Owner"] = raw_df["housing_is_owner"]
    df["Housing_is_Free"] = raw_df["housing_is_free"]

    df["Job_is_Unskilled"] = raw_df["job_is_unskilled"]
    df["Job_is_Skilled"] = raw_df["job_is_skilled"]
    df["Job_is_Management"] = raw_df["job_is_management"]

    #df["Unemployed"] = raw_df["unemployed"]
    employed_bt_0_to_1 = raw_df[["unemployed", "employed_bt_1_4_yr", "employed_bt_4_7_yr", "employed_geq_7_yr"]].sum(axis=1) == 0
#    df["YearsEmployed_geq_0"] = employed_bt_0_to_1 + raw_df[["employed_bt_1_4_yr", "employed_bt_4_7_yr", "employed_geq_7_yr"]].any(axis=1)
    df["YearsEmployed_geq_1"] = raw_df[["employed_bt_1_4_yr", "employed_bt_4_7_yr", "employed_geq_7_yr"]].any(axis=1)
    # df["YearsEmployed_geq_4"] = raw_df[["employed_bt_4_7_yr", "employed_geq_7_yr"]].any(axis=1)
    # df["YearsEmployed_geq_7"] = raw_df["employed_geq_7_yr"]

    # Loan Amount
    credit_amt = raw_df["credit_amt"]
    threshold_amount_thresholds = [1000, 2000, 5000, 10000]
    for t in threshold_amount_thresholds:
        df[f"CreditAmt_geq_{t}K"] = credit_amt >= t

    # Loan Duration
    #df["LoanDuration_geq_10m"] = raw_df["duration"] >= 10
    df["LoanDuration_leq_6"] = raw_df["duration"] <= 6
    df["LoanDuration_geq_12"] = raw_df["duration"] >= 12
    df["LoanDuration_geq_24"] = raw_df["duration"] >= 24
    df["LoanDuration_geq_36"] = raw_df["duration"] >= 36
    # Loan Rate
    #df["LoanRate_geq_2"] = raw_df["installment_rate"] >= 2
    df["LoanRate"] = raw_df["installment_rate"]
    #df["HasCoApplicant"] = raw_df["co-applicant"]
    df["HasGuarantor"] = raw_df["guarantor"]

    # Loan Purpose (immutable)
    df["LoanRequiredForBusiness"] = raw_df["purpose_is_business"]
    df["LoanRequiredForEducation"] = raw_df[["purpose_is_education", "purpose_is_retraining"]].any(axis=1)
    df["LoanRequiredForCar"] = raw_df[["purpose_is_car_new", "purpose_is_car_used"]].any(axis=1)
    df["LoanRequiredForHome"] = raw_df[["purpose_is_furniture", "purpose_is_radio/tv", "purpose_is_domestic_appliances", "purpose_is_repairs"]].any(axis=1)
    #df["LoanRequiredForOther"] = ~df[["LoanRequiredForHome", "LoanRequiredForCar", "LoanRequiredForEducation", "LoanRequiredForBusiness"]].any(axis=1)

    # CreditHistory
    df["NoCreditHistory"] = raw_df["credit_history_no_credits_taken"]
    df["HistoryOfLatePayments"] = raw_df[["credit_history_delay_in_paying_credits"]]
    df["HistoryOfDelinquency"] = raw_df[["credit_history_critical_credit_history"]]
    df["HistoryOfBankInstallments"] = raw_df[["has_bank_installments"]]
    df["HistoryOfStoreInstallments"] = raw_df[["has_store_installments"]]
    #df["ExisingCredits"] = raw_df["existing_credits"]

    # Checking Account
    # df["CheckingAcct_none"] = raw_df["no_checking_acct"]
    # df["CheckingAcct_leq_0"] = raw_df["checking_acct_le_0"]
    # df["CheckingAcct_bt_0_200"] = raw_df["checking_acct_bt_0_200"]
    # df["CheckingAcct_geq_200"] = raw_df["checking_acct_ge_200"]
    df["CheckingAcct_exists"] = raw_df["no_checking_acct"] == False
    #df["CheckingAcct_geq_any"] = raw_df[["checking_acct_le_0", "checking_acct_bt_0_200", "checking_acct_ge_200"]].any(axis = 1)
    df["CheckingAcct_geq_0"] = raw_df[["checking_acct_bt_0_200", "checking_acct_ge_200"]].any(axis = 1)
    # df["CheckingAcct_geq_200"] = raw_df[["checking_acct_ge_200"]]

    # Savings Account
    #df["SavingsAcct_none"] = raw_df["no_savings_acct"]
    # df["SavingsAcct_leq_100"] = raw_df["savings_acct_le_100"]
    # df["SavingsAcct_bt_100_499"] = raw_df["savings_acct_bt_100_499"]
    # df["SavingsAcct_bt_500_999"] = raw_df["savings_acct_bt_500_999"]
    # df["SavingsAcct_geq_1000"] = raw_df["savings_acct_ge_1000"]
    df["SavingsAcct_exists"] = raw_df["no_savings_acct"] == False
    df["SavingsAcct_geq_100"] = raw_df[["savings_acct_bt_100_499", "savings_acct_bt_500_999", "savings_acct_ge_1000"]].any(axis = 1)
    #df["SavingsAcct_geq_500"] = raw_df[["savings_acct_bt_500_999", "savings_acct_ge_1000"]].any(axis = 1)
    #df["SavingsAcct_geq_1000"] = raw_df[["savings_acct_ge_1000"]].any(axis = 1)
    return df

def simple_1D(data):
    A = ActionSet(data.X_df)
    immutable_features = (
            [a.name for a in A if "Age" in a.name] +
            [a.name for a in A if "Job_is" in a.name] +
            [a.name for a in A if "Housing_is_" in a.name] +
            [a.name for a in A if "CreditAmt_geq_" in a.name] +
            [a.name for a in A if "LoanDuration_" in a.name] +
            [a.name for a in A if "LoanRate" in a.name] +
            [
                "Male",
                "Single",
                "ForeignWorker",
                "LiablePersons",
                #
                #
                #"LoanRate_geq_2",
                #
                "NoCreditHistory",
                "HistoryOfLatePayments",
                "HistoryOfDelinquency",
                #"HistoryOfInstallments",
                #
                "LoanRequiredForCar",
                "LoanRequiredForHome",
                "LoanRequiredForEducation",
                "LoanRequiredForBusiness",
                #
                #"HasGuarantor",
                #"HasCoApplicant",
                ]
    )
    A[immutable_features].actionable = False
    A["YearsAtResidence"].lb = 0
    A["YearsAtResidence"].ub = 7
    A["YearsAtResidence"].step_ub = 2
    return A

def complex_1D(data):
    A = simple_1D(data)
    age_features = [a.name for a in A if "Age" in a.name]
    A[age_features].step_direction = 1
    A["YearsAtResidence"].step_direction = 1
    A["HasGuarantor"].step_direction = 1
    A["HistoryOfBankInstallments"].step_direction = 1
    A["HistoryOfStoreInstallments"].step_direction = 1
    #A["HistoryOfInstallments"].step_direction = 1
    A[[a.name for a in A if "YearsEmployed_" in a.name]].step_direction = 1
    A[[a.name for a in A if "SavingsAcct_" in a.name]].step_direction = 1
    A[[a.name for a in A if "CheckingAcct_" in a.name]].step_direction = 1
    return A

def complex_nD(data):
    A = complex_1D(data)

    A.constraints.add(
            constraint=DirectionalLinkage(names=["YearsAtResidence", "Age"], scales=[1, 1])
            )

    A.constraints.add(
            constraint=DirectionalLinkage(names=["YearsEmployed_geq_1", "Age"], scales=[1, 1])
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[a.name for a in A if "CheckingAcct_" in a.name],
                    step_direction=1,
                    )
            )

    A.constraints.add(
            constraint=ThermometerEncoding(
                    names=[a.name for a in A if "SavingsAcct_" in a.name],
                    step_direction=1,
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

# create actionset
for name in settings["action_set_names"]:

    if name == "simple_1D":
        action_set = simple_1D(data)
    elif name == "complex_1D":
        action_set = complex_1D(data)
    elif name == "complex_nD":
        action_set = complex_nD(data)

    print(tabulate_actions(action_set))
    # check that action set matches bounds and constraints
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
                path=get_reachable_db_file(
                        data_name=settings["data_name"], action_set_name=name
                        ),
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

        clf = comp_results['model']['model']
        scaler = comp_results['model']['scaler']
        weights = pd.DataFrame({
            'names': data.names.X,
            'coefficient': np.true_divide(clf.coef_,  scaler.scale_).flatten()
            }).set_index("names")

        # reachable set distribution
        print('reachable point distribution')
        print(recourse_df["n_reachable"].describe())

        print('reachable point distribution')
        pp.pprint(tally(recourse_df['n_feasible']))

        # analysis for predictions without recourse
        n_feasible_threshold = 0
        fixed_idx = audit_df.query("yhat == False").query(f"n_feasible == {n_feasible_threshold}").index.tolist()
        fixed_features = data.X_df.loc[fixed_idx].T
        fixed_features["weights"] = weights
        fixed_features = fixed_features[["weights"] + fixed_features.columns[0:-1].tolist()]

        fixed_dfs = {}
        for i in fixed_idx:
            x = data.X[i, :]
            fixed_dfs[i] = db[x].describe(predictor = predictor, target = 1)
            #fixed_dfs[i]['w'] = weights
            #fixed_dfs[i] = fixed_dfs[i][["x", "w", "n_total", "n_target"]]
            print("="*70)
            print(f"x[{i}] | n_feasible={n_feasible_threshold}")
            print("-"*70)
            print(fixed_dfs[i])
            print("="*70)


        # # population level analysis
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
