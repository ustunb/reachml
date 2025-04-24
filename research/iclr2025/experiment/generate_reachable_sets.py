import os
import sys
from argparse import ArgumentParser

from psutil import Process

# script / source code imports
sys.path.append(os.getcwd())  # add repo dir to path to import source code
from src.ext import fileutils
from src.paths import *

from reachml import ReachableSetDatabase

# script settings / default command line arguments
settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "hard_cap": None,
    "overwrite": True,
    "n": 500,
}

# parse settings from command line when script is not run from iPython console
if Process(pid=os.getppid()).name() not in ("pycharm"):
    p = ArgumentParser()
    p.add_argument("--data_name", type=str, required=True)
    p.add_argument("--action_set_name", type=str, required=True)
    p.add_argument("--hard_cap", type=int, default=settings["hard_cap"])
    p.add_argument("--overwrite", default=settings["overwrite"], action="store_true")
    args, _ = p.parse_known_args()
    settings.update(vars(args))

data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))
database_file = get_reachable_db_file(**settings)
print(f"database file: {database_file}")
database = ReachableSetDatabase(action_set=action_set, path=database_file)
stats = database.generate(data.X, **settings)
print(stats.n_points.describe())
