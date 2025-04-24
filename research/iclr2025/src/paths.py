"""
This file defines paths for key directories and files. Contents include:
1. Directory Names: Path objects that specify the directories where we store code, data, results, etc.
2. File Name Generators: functions used to programatically name processed datasets, results, graphs etc.
"""

from pathlib import Path

# Directories

# path to the GitHub repository
paper_dir = Path(__file__).resolve().parent.parent
repo_dir = paper_dir.parent.parent

# directory where we store datasets
data_dir = paper_dir / "data/"

# path to the Python package
tests_dir = repo_dir / "tests/"

# directory where we store results
results_dir = paper_dir / "results/"

# directory where we store plots
plot_dir = paper_dir / "plots/"

# directory of reporting package #TODO: check if necessary
reporting_dir = paper_dir / "reporting/"

# directory where we store templates #TODO: check if necessary
templates_dir = reporting_dir / "templates/"

# path to reachml code
rs_repo_dir = repo_dir / "reach/"

# create local directories if they do not exist
results_dir.mkdir(exist_ok=True)


# Naming Functions
def get_data_csv_file(data_name, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / data_name / f"{data_name}_processed.csv"
    return f


def get_data_file(data_name, action_set_name, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    f = data_dir / f"{data_name}_{action_set_name}.data"
    return f


def get_action_set_file(data_name, action_set_name, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :return: file name
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = data_dir / f"{data_name}_{action_set_name}.actionset"
    return f


def get_model_file(data_name, action_set_name, model_type, is_raw=False, **kwargs):
    file_type = "raw" if is_raw else "processed"
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{file_type}.model"
    return f


def get_explainer_file(
    data_name, model_type, explainer_type, action_set_name=None, **kwargs
):
    """
    returns file name of a explainer object (i.e. lime or shap)

    :param data_name: dataset name
    :param model_type: model type
    :param explainer_name: explainer name
    :param return_both: if True, returns both the original explainer and actionAwareExplainer associated with action_set_name
    :param action_set_name: action set name (optional)
    """
    if "actionAware" in explainer_type:
        f = (
            results_dir
            / f"{data_name}_{action_set_name}_{model_type}_{explainer_type}.explainer"
        )
    else:
        f = results_dir / f"{data_name}_{model_type}_{explainer_type}.explainer"
    return f


def get_benchmark_results_file(
    data_name, action_set_name, method_name, model_type, **kwargs
):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(action_set_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = (
        results_dir
        / f"{data_name}_{action_set_name}_{model_type}_{method_name}.results"
    )
    return f


def get_audit_results_file(data_name, action_set_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}.audit"
    return f


def get_stats_file(data_name, action_set_name, method_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(action_set_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{method_name}.stats"
    return f


def get_metrics_file(
    data_name, action_set_name, model_type=None, explainer_type=None, **kwargs
):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0

    if model_type is None:
        f = results_dir / f"{data_name}_{action_set_name}.metrics"
    elif explainer_type is None:
        f = results_dir / f"{data_name}_{action_set_name}_{model_type}.metrics"
    else:
        f = (
            results_dir
            / f"{data_name}_{action_set_name}_{model_type}_{explainer_type}.metrics"
        )

    return f


def get_reachable_db_file(data_name, action_set_name, **kwargs):
    """
    returns file name of a reachable set dataset.

    :param data_name: string containing name of the dataset
    :param action_set_name: string containing name of the action set
    :param kwargs: used to catch other args when unpacking dictionaies
                   this allows us to call this function as get_results_file_name(**settings)

    :return: Path of results object
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}.database"
    return f


def get_scorer_file(data_name, action_set_name, **kwargs):
    """
    returns file name of a reachable set dataset.

    :param data_name: string containing name of the dataset
    :param action_set_name: string containing name of the action set
    :param kwargs: used to catch other args when unpacking dictionaies
                   this allows us to call this function as get_results_file_name(**settings)

    :return: Path of results object
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}.scorer"
    return f


def get_resp_df_file(data_name, action_set_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_resp.df"
    return f


def get_plot_data_file(
    data_name, action_set_name, model_type, explainer_type, **kwargs
):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    assert isinstance(model_type, str) and len(model_type) > 0
    f = (
        results_dir
        / f"{data_name}_{action_set_name}_{model_type}_{explainer_type}_plot_data.df"
    )
    return f


def get_plot_file(
    data_name, action_set_name, model_type, explainer_type, plot_name, **kwargs
):
    """
    return file name of a plot (without extension, extension set by plotting script)
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    assert isinstance(model_type, str) and len(model_type) > 0
    assert isinstance(plot_name, str)
    f = (
        plot_dir
        / f"{data_name}_{action_set_name}_{model_type}_{explainer_type}_plot_{plot_name}"
    )
    return f
