import os
import sys
from pathlib import Path

# name of the virtual environment for R
DEFAULT_ENVNAME = "reticulate-python-env"

# Path of the GitHub repository
REPO_DIR = Path(__file__).absolute().parent

# Path to the Python executable used to build the virtual environment
DEFAULT_PATH = Path(sys.executable)

# Names of required R packages
DEFAULT_PKG_LIST_PY = ("pandas", "numpy", "scipy", "dill", "pickle5", "matplotlib", "seaborn", "ipython", "prettytable", "scikit-learn")

def setup_R():
    """
    :param verbose:
    :return:
    """
    f = REPO_DIR / 'setup.R'
    cmd = "Rscript '%s' " % f
    out = os.system(cmd)
    return out


def setup_reticulate(envname = DEFAULT_ENVNAME, python_path = DEFAULT_PATH, pkg_list = DEFAULT_PKG_LIST_PY):
    """
    :param envname:
    setting envname = None will use R defaults

    :param python_path:
    setting python_path = None will use R default

    :param pkg_list:
    setting pkg_list = None will use R default

    :return:
    """

    assert Path(python_path).exists()
    assert isinstance(pkg_list, (list, set, tuple))
    assert all([isinstance(pkg, str) for pkg in pkg_list])

    f = REPO_DIR / 'setup_reticulate.R'
    cmd = "Rscript '%s' " %f

    # parse args
    args = []
    if envname is not None:
        args.append("--envname '%s'" % envname)

    if python_path is not None:
        args.append("--python_path '%s'" % python_path)

    if pkg_list is not None:
        pkg_list = ' '.join(["%s" % pkg for pkg in pkg_list])
        args.append("--packages '%s'" % pkg_list)

    if len(args) > 0:
        cmd = cmd + ' '.join(args)

    out = os.system(cmd)
    return out


if __name__ == "__main__":

    setup_R()
    setup_reticulate()
