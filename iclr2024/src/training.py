from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import warnings

from src import metrics

warnings.filterwarnings("ignore")


def sample_processing(
    data, label_encoding=None, rescale=False, rebalance=None, seed=None
):
    """
    process training and testing data for training -- use for rescaling etc.
    :param data: BinaryClassificationDataset
    :param remap_labels = {-1: 0
    :param rebalancing_train: None or 'over' (oversampling, 'under' (undersampling)
    :param rebalancing_test: None or 'over' (oversampling, 'under' (undersampling)
    :param seed: random seed used for resampling
    :return: X_train, y_train, X_test, y_test for training
    """
    assert rebalance in (None, "over", "under")
    if label_encoding is not None:
        assert len(label_encoding) == 2
        data.training.update_classes(values=label_encoding)
        data.test.update_classes(values=label_encoding)
    else:
        label_encoding = tuple(data.classes)

    args = (
        {
            "seed": seed,
            "label_encoding": label_encoding,
            "rescale": rescale,
            "rebalance": rebalance,
        },
    )

    X_train, y_train = data.training.X, data.training.y
    X_test, y_test = data.test.X, data.test.y

    if rebalance is not None:
        if rebalance == "over":
            resampler = RandomOverSampler(random_state=seed)
        elif rebalance == "under":
            resampler = RandomUnderSampler(random_state=seed)
        X_train, y_train = resampler.fit_resample(X_train, y_train)
        X_test, y_test = resampler.fit_resample(X_test, y_test)

    scaler = None
    if rescale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    out = {
        "args": args,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

    return out


def train_logreg(data, seed=None, label_encoding=None, rescale=True, rebalance=None):

    processed = sample_processing(
        data,
        label_encoding=label_encoding,
        rescale=rescale,
        rebalance=rebalance,
        seed=seed,
    )
    pool = RandomizedSearchCV(
        estimator=LogisticRegression(random_state=seed),
        param_distributions={
            "penalty": [None],
            "solver": ["saga"],
            "max_iter": [3_000_000],
        },
        n_iter=10,
        cv=3,
        verbose=1,
        random_state=seed,
        n_jobs=-1,
    )
    pool.fit(processed["X_train"], processed["y_train"])
    clf = pool.best_estimator_

    out = {
        "model_type": "logreg",
        "sample_processing_args": processed["args"],
        "model": clf,
        "scaler": processed["scaler"],
        "train": get_clf_stats(processed["X_train"], processed["y_train"], clf),
        "test": get_clf_stats(processed["X_test"], processed["y_test"], clf),
    }
    return out


def train_logreg_vanilla(
    data, seed=None, label_encoding=None, rescale=True, rebalance=None
):
    processed = sample_processing(
        data,
        label_encoding=label_encoding,
        rescale=rescale,
        rebalance=rebalance,
        seed=seed,
    )
    clf = LogisticRegression(
        random_state=seed, penalty=None, solver="saga", max_iter=3_000_000, verbose=0
    )
    clf.fit(processed["X_train"], processed["y_train"])
    out = {
        "model_type": "logreg",
        "sample_processing_args": processed["args"],
        "model": clf,
        "scaler": processed["scaler"],
        "train": get_clf_stats(processed["X_train"], processed["y_train"], clf),
        "test": get_clf_stats(processed["X_test"], processed["y_test"], clf),
    }
    return out


def train_rf(data, seed=None, label_encoding=None, rescale=False, rebalance=None):

    processed = sample_processing(
        data,
        label_encoding=label_encoding,
        rescale=rescale,
        rebalance=rebalance,
        seed=seed,
    )

    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    pool = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=seed),
        param_distributions={
            "n_estimators": [int(x) for x in np.linspace(start=300, stop=2000, num=10)],
            "max_depth": [None] + [int(x) for x in np.linspace(5, 110, num=11)],
            "min_samples_split": [2, 5, 7, 10, 12],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
    )

    pool.fit(processed["X_train"], processed["y_train"])
    clf = pool.best_estimator_

    out = {
        "model_type": "rf",
        "sample_processing_args": processed["args"],
        "model": clf,
        "scaler": processed["scaler"],
        "train": get_clf_stats(processed["X_train"], processed["y_train"], clf),
        "test": get_clf_stats(processed["X_test"], processed["y_test"], clf),
    }

    return out


def train_xgb(data, seed=None, label_encoding=(0, 1), rescale=False, rebalance=None):

    # replace all -1 values with 0
    processed = sample_processing(
        data,
        label_encoding=label_encoding,
        rescale=rescale,
        rebalance=rebalance,
        seed=seed,
    )

    pool = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=seed),
        param_distributions={
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "max_depth": [int(x) for x in np.linspace(5, 110, num=11)] + [None],
            "n_estimators": [int(x) for x in np.linspace(start=300, stop=2000, num=10)],
            "max_depth": [i for i in range(0, 12)],
            "min_child_weight": [1, 5, 15],
            "gamma": [i for i in range(0, 10)],
            "booster": ["gbtree", "gblinear", "dart"],
            "subsample": np.arange(0.5, 1.1, 0.1),
            "grow_policy": [0, 1],
        },
        n_iter=10,
        cv=3,
        verbose=2,
        random_state=seed,
        n_jobs=-1,
    )

    pool.fit(processed["X_train"], processed["y_train"])
    clf = pool.best_estimator_
    out = {
        "model_type": "xgb",
        "sample_processing_args": processed["args"],
        "model": clf,
        "scaler": processed["scaler"],
        "train": get_clf_stats(processed["X_train"], processed["y_train"], clf),
        "test": get_clf_stats(processed["X_test"], processed["y_test"], clf),
    }

    return out


def get_clf_stats(X, y, model):

    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)
    if y_probs.shape[1] == 2:
        y_probs = y_probs[:, 1]

    error = metrics.compute_error(y, y_pred)

    auc = metrics.compute_auc(y, y_probs)
    # todo: auc = skmetrics.roc_auc_score(y, probs) for train_rf/XGB if needed?

    loss = metrics.compute_log_loss(y, y_probs)
    ece = metrics.compute_ece(y, y_probs)
    stats = {
        "auc": round(auc, 4),
        "log_loss": round(loss, 4),
        "error": round(error, 4),
        "ece": round(ece, 1),
        "n": len(y),
        "n_pos_pred": len(y) - np.greater(y_pred, 0).sum(),
        "p_pos_pred": np.greater(y_pred, 0).mean(),
    }

    return stats


def extract_predictor(clf, scaler=None):
    if scaler is None:
        predictor = lambda x: clf.predict(x)
    else:
        reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
        rescale = lambda x: scaler.transform(reformat(x))
        predictor = lambda x: clf.predict(rescale(x))
    return predictor


def probs(model, pt, outcome_prob=1, scaler=None):
    pt = [pt] if scaler is None else scaler.transform([pt])
    out = model.predict_proba(pt)
    if outcome_prob == 1:
        return "{0:.2%}".format(out[1])
    else:
        return "{0:.2%}".format(out[0])
