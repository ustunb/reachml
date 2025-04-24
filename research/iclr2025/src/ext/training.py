import warnings

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import metrics

warnings.filterwarnings("ignore")

# Parameter distributions for hyperparameter search for each model type.
PARAMS_LOGREG = {
    "penalty": [None],
    "solver": ["saga"],
    "max_iter": [3_000_000],
}

PARAMS_RF = {
    "n_estimators": [int(x) for x in np.linspace(300, 2000, num=10)],
    "max_depth": [None] + [int(x) for x in np.linspace(5, 110, num=11)],
    # "min_samples_split": [2, 5, 7, 10, 12],
    # "min_samples_leaf": [1, 2, 4, 6, 8],
    "min_samples_split": [int(x) for x in np.arange(30, 50)],
    "min_samples_leaf": [int(x) for x in np.arange(10, 30)],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
}

PARAMS_XGB = {
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [i for i in range(3, 12)],  # discrete range used for demonstration
    "n_estimators": [int(x) for x in np.arange(100, 600, 100)],
    "min_child_weight": [1, 3, 5, 7, 9, 15],
    "gamma": [i for i in range(0, 10)],
    "booster": ["gbtree"],  # TreeSHAP only supports gbtree
    "subsample": np.arange(0.5, 1.1, 0.1),
    "grow_policy": ["depthwise", "lossguide"],
    "reg_alpha": [0, 0.01, 0.1],
    "reg_lambda": [1, 1.1, 1.2],
}

# PARAMS_XGB = {
#     # "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
#     "learning_rate": [0.01, 0.05, 0.1],
#     "max_depth": [i for i in range(0, 12)],  # discrete range used for demonstration
#     "n_estimators": [int(x) for x in np.linspace(300, 2000, num=10)],
#     "min_child_weight": [1, 5, 15],
#     "gamma": [i for i in range(0, 10)],
#     "booster": ["gbtree"],  # TreeSHAP only supports gbtree
#     "subsample": np.arange(0.5, 1.1, 0.1),
#     "grow_policy": ["depthwise", "lossguide"],
# }

CLF_MAP = {
    "logreg": [LogisticRegression, PARAMS_LOGREG],
    "rf": [RandomForestClassifier, PARAMS_RF],
    "xgb": [XGBClassifier, PARAMS_XGB],
}


def sample_processing(
    data, label_encoding=None, rescale=False, rebalance=None, seed=None
):
    """
    Process training and testing data, performing optional label encoding, rescaling, and rebalancing.

    Args:
        data: BinaryClassificationDataset.
        label_encoding (tuple, optional): Mapping of original labels to new labels.
        rescale (bool): Whether to standardize features.
        rebalance (str or None): Options: None, "over", or "under" for resampling.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Contains processed X_train, y_train, X_test, y_test, scaler, and processing parameters.
    """
    assert rebalance in (None, "over", "under"), (
        "rebalance must be one of None, 'over', or 'under'"
    )
    if label_encoding is not None:
        assert len(label_encoding) == 2, "label_encoding must have two elements"
        data.training.update_classes(values=label_encoding)
        data.test.update_classes(values=label_encoding)
    else:
        label_encoding = tuple(data.classes)

    args = {
        "seed": seed,
        "label_encoding": label_encoding,
        "rescale": rescale,
        "rebalance": rebalance,
    }

    X_train, y_train = data.training.X, data.training.y
    X_test, y_test = data.test.X, data.test.y
    X_valid, y_valid = data.validation.X, data.validation.y

    if rebalance is not None:
        resampler = (
            RandomOverSampler(random_state=seed)
            if rebalance == "over"
            else RandomUnderSampler(random_state=seed)
        )
        X_train, y_train = resampler.fit_resample(X_train, y_train)
        X_test, y_test = resampler.fit_resample(X_test, y_test)
        if X_valid.shape[0] > 0:
            X_valid, y_valid = resampler.fit_resample(X_valid, y_valid)

    scaler = None
    if rescale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if X_valid.shape[0] > 0:
            X_valid = scaler.transform(X_valid)

    return {
        "args": args,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_valid": X_valid if X_valid.shape[0] > 0 else None,
        "y_valid": y_valid if X_valid.shape[0] > 0 else None,
    }


def _build_output(processed, model, model_type):
    """
    Helper function for building standard output dictionary for training functions.

    Args:
        processed (dict): Output dictionary from sample_processing.
        model: Trained classifier.
        model_type (str): Identifier for the model type.

    Returns:
        dict: Contains model type, processing args, model, scaler, and training/test statistics.
    """
    return {
        "model_type": model_type,
        "sample_processing_args": processed["args"],
        "model": model,
        "scaler": processed["scaler"],
        "train": get_clf_stats(processed["X_train"], processed["y_train"], model),
        "test": get_clf_stats(processed["X_test"], processed["y_test"], model),
    }


def train_model(
    data,
    model_type,
    seed=None,
    label_encoding=None,
    rescale=False,
    rebalance=None,
    **kwargs,
):
    """
    Train a model of the given class using randomized search for hyperparameters and wrap with a scaler if provided.

    Args:
        data: BinaryClassificationDataset.
        model_type (str): Identifier for the model type.
        seed (int, optional): Random seed.
        label_encoding (tuple, optional): Mapping for label encoding.
        rescale (bool): Whether to scale the data. For LR this is typically True. RF and XGB default to False.
        rebalance (str or None): Resampling option: None, 'over', or 'under'.

    Returns:
        dict: A dictionary containing the model type, sample processing arguments, trained model, scaler,
              and training and test statistics.
    """
    assert model_type in CLF_MAP, f"model_type must be one of {list(CLF_MAP.keys())}"

    model_class, param_distributions = CLF_MAP[model_type]

    processed = sample_processing(
        data,
        label_encoding=label_encoding,
        rescale=rescale,
        rebalance=rebalance,
        seed=seed,
    )

    pool = RandomizedSearchCV(
        estimator=model_class(random_state=seed),
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        verbose=2,
        scoring="roc_auc",
        random_state=seed,
        n_jobs=-1,
    )

    if kwargs.get("early_stopping_rounds") is not None:
        assert model_type == "xgb", (
            "early_stopping_rounds only supported for XGBClassifier"
        )
        assert processed["X_valid"] is not None, (
            "early_stopping_rounds requires validation data"
        )
        pool.set_params(
            estimator__early_stopping_rounds=kwargs["early_stopping_rounds"]
        )
        pool.fit(
            X=processed["X_train"],
            y=processed["y_train"],
            eval_set=[(processed["X_valid"], processed["y_valid"])],
        )
    else:
        pool.fit(X=processed["X_train"], y=processed["y_train"])

    best_estimator = pool.best_estimator_

    # Always wrap model with scaler in a pipeline if a scaler is present.
    if processed["scaler"] is not None:
        clf = Pipeline([("scaler", processed["scaler"]), ("clf", best_estimator)])
    else:
        clf = best_estimator

    return _build_output(processed, clf, model_type)


def get_clf_stats(X, y, model):
    """
    Compute classification statistics for the given model and data.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): True labels.
        model: Trained classifier with predict and predict_proba methods.

    Returns:
        dict: Statistics including AUC, log loss, error, ECE, and prediction details.
    """
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)
    if y_probs.shape[1] == 2:
        y_probs = y_probs[:, 1]

    error = metrics.compute_error(y, y_pred)
    auc = metrics.compute_auc(y, y_probs)
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
    """
    Return a predictor function based on the classifier, optionally applying scaling.

    Args:
        clf: Trained classifier.
        scaler: Optional scaler, if training data was scaled.

    Returns:
        callable: A function that takes input x and returns predictions.
    """
    if scaler is None:
        predictor = lambda x: clf.predict(x)
    else:
        reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
        rescale_input = lambda x: scaler.transform(reformat(x))
        predictor = lambda x: clf.predict(rescale_input(x))
    return predictor


def probs(model, pt, outcome_prob=1, scaler=None):
    """
    Compute and format predicted probability for a single data point.

    Args:
        model: Trained classifier with predict_proba method.
        pt: Single data point.
        outcome_prob (int): Index (1 for positive, otherwise 0) for probability selection.
        scaler: Optional scaler to transform the input data.

    Returns:
        str: Formatted probability as a percentage string.
    """
    pt = [pt] if scaler is None else scaler.transform([pt])
    out = model.predict_proba(pt)
    if outcome_prob == 1:
        return "{0:.2%}".format(out[0][1])
    else:
        return "{0:.2%}".format(out[0][0])
