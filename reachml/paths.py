"""This file defines paths for key directories and files. Contents include:
1. Directory Names: Path objects that specify the directories where we store code, data, results, etc.
2. File Name Generators: functions used to programatically name processed datasets, results, graphs etc.
"""

from pathlib import Path

# Directories

# path to the GitHub repository
repo_dir = Path(__file__).resolve().parent.parent

# path to the Python package
pkg_dir = repo_dir / "reachml/"

# directory where we store datasets
data_dir = repo_dir / "data/"

# path to the Python package
tests_dir = repo_dir / "tests/"

# directory where we store results
results_dir = repo_dir / "results/"

# create local directories if they do not exist
results_dir.mkdir(exist_ok=True)


# Naming Functions
def get_data_csv_file(data_name, **kwargs):
    """:param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / data_name / f"{data_name}_processed.csv"
    return f


def get_data_file(data_name, action_set_name, **kwargs):
    """:param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / f"{data_name}_{action_set_name}.data"
    return f


def get_action_set_file(data_name, action_set_name, **kwargs):
    """:param data_name: string containing name of the dataset
    :return: file name
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = data_dir / f"{data_name}_{action_set_name}.actionset"
    return f


def get_model_file(data_name, action_set_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0

    f = f"{data_name}_{action_set_name}_{model_type}"

    if kwargs.get("n_features") is not None:
        assert isinstance(kwargs.get("n_features"), int)
        assert kwargs.get("meta_features") is not None
        n_feat = kwargs.get("n_features")
        meta_features = kwargs.get("meta_features")
        f += f"_n_feat_{n_feat}_{'+'.join([str(s) for s in meta_features])}"

    if kwargs.get("model_number") is not None:  # for glm
        assert isinstance(kwargs.get("model_number"), int)
        model_number = kwargs.get("model_number")
        f += f"_{model_number}"

    return results_dir / f"{f}.model"


def get_benchmark_results_file(data_name, action_set_name, method_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(action_set_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{method_name}.results"
    return f


def get_audit_results_file(data_name, action_set_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0

    f = f"{data_name}_{action_set_name}_{model_type}"

    if kwargs.get("model_number") is not None:  # for glm
        assert isinstance(kwargs.get("model_number"), int)
        model_number = kwargs.get("model_number")
        f += f"_model_{model_number}"

    if kwargs.get("resp_thresh") is not None:
        resp_thresh = kwargs.get("resp_thresh")
        f += f"_epsilon_{resp_thresh}"

    if kwargs.get("alpha") is not None:
        alpha = kwargs.get("alpha")
        f += f"_alpha_{alpha}"

    if kwargs.get("n") is not None:
        sample_size = kwargs.get("n")
        f += f"_n_{sample_size}"

    if kwargs.get("trial") is not None:
        trial = kwargs.get("trial")
        f += f"_trial_{trial}"

    return results_dir / f"{f}.audit"


def get_demo_results_file(data_name, action_set_name, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0

    f = f"{data_name}_{action_set_name}"

    if kwargs.get("resp_thresh") is not None:
        resp_thresh = kwargs.get("resp_thresh")
        f += f"_epsilon_{resp_thresh}"
    if kwargs.get("model_type") is not None:
        model_type = kwargs.get("model_type")
        f += f"_{model_type}"

    f = f"{f}_demo.results"

    if kwargs.get("csv"):
        f += ".csv"

    return results_dir / f"{f}"


def get_stats_file(data_name, action_set_name, method_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(method_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{method_name}.stats"
    return f


def get_reachable_db_file(data_name, action_set_name, **kwargs):
    """Returns file name of a reachable set dataset.

    :param data_name: string containing name of the dataset
    :param action_set_name: string containing name of the action set
    :param kwargs: used to catch other args when unpacking dictionaies
                   this allows us to call this function as get_results_file_name(**settings)

    :return: Path of results object
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0

    f = f"{data_name}_{action_set_name}"

    return results_dir / f"{f}.database"


def get_explainer_file(data_name, model_type, explainer_type, action_set_name=None, **kwargs):
    """Returns file name of a explainer object (i.e. lime or shap).

    :param data_name: dataset name
    :param model_type: model type
    :param explainer_name: explainer name
    :param return_both: if True, returns both the original explainer and actionAwareExplainer associated with action_set_name
    :param action_set_name: action set name (optional)
    """
    if "actionAware" in explainer_type:
        header = f"{data_name}_{action_set_name}_{model_type}_{explainer_type}"
    else:
        header = f"{data_name}_{model_type}_{explainer_type}"
    f = results_dir / f"{header}.explainer"
    return f


def get_metrics_file(data_name, action_set_name, model_type=None, explainer_type=None, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    header = f"{data_name}_{action_set_name}"
    if model_type is not None:
        header = f"{header}_{model_type}"
    if explainer_type is not None:
        header = f"{header}_{explainer_type}"
    f = results_dir / f"{header}.metrics"
    return f


def get_rij_file(data_name, action_set_name, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_rij.df"
    return f


def get_plot_data_file(data_name, action_set_name, model_type, explainer_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    assert isinstance(model_type, str) and len(model_type) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{explainer_type}_plot_data.df"
    return f


def get_plot_file(data_name, action_set_name, model_type, explainer_type, plot_name, **kwargs):
    """Return file name of a plot (without extension, extension set by plotting script)."""
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    assert isinstance(model_type, str) and len(model_type) > 0
    assert isinstance(plot_name, str)
    f = (
        results_dir
        / f"{data_name}_{action_set_name}_{model_type}_{explainer_type}_plot_{plot_name}"
    )
    return f
