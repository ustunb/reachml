"""
This file defines paths for key directories and files. Contents include:
1. Directory Names: Path objects that specify the directories where we store code, data, results, etc.
2. File Name Generators: functions used to programatically name processed datasets, results, graphs etc.
"""

from pathlib import Path
import os

# Directories

# path to the GitHub repository
paper_dir = Path(os.path.join(os.getcwd(), "")).resolve()
repo_dir = Path(os.getcwd()).resolve()

# directory where we store datasets
data_dir = paper_dir / "data/"

# path to the Python package
tests_dir = repo_dir / "tests/"

# directory where we store results
results_dir = paper_dir / "results/"

# directory of reporting package
reporting_dir = paper_dir / "reporting/"

# directory where we store templates
templates_dir = reporting_dir / "templates/"

# todo: remove
models_dir = (
    lambda data_name, action_set_name: results_dir
    / data_name
    / action_set_name
    / "models"
)

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


def get_audit_results_file(
    data_name, action_set_name, method_name, model_type, **kwargs
):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(action_set_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{method_name}.audit"
    return f


def get_stats_file(data_name, action_set_name, method_name, model_type, **kwargs):
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(action_set_name) > 0
    assert isinstance(action_set_name, str) and len(action_set_name) > 0
    f = results_dir / f"{data_name}_{action_set_name}_{model_type}_{method_name}.stats"
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
