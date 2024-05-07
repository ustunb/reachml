import sys
import os

sys.path.append(os.getcwd())

from src import fileutils
from src.paths import *
import pandas as pd

results_non_part = fileutils.load(get_audit_results_file("fico", "action_set_10_timed"))
results_part = fileutils.load(get_audit_results_file("fico", "action_set_10"))

non_part_df = pd.DataFrame(data=results_non_part["reachable_sets"])
part_df = pd.DataFrame(data=results_part["reachable_sets"])

non_part_df = non_part_df.drop_duplicates(subset=["id_for_x"])
non_part_df = non_part_df.drop(
    columns=["point", "reachable_set", "action_values", "count", "percent_positive_pts"]
)
part_df = part_df.drop(
    columns=["point", "reachable_set", "action_values", "count", "time"]
)

part_df = part_df.drop_duplicates(subset=["num_reachable_pts"])


# assert len(non_part_df) == len(part_df)
non_part_df["Computation"] = "Brute Force"
non_part_df["Alpha"] = 0.8
part_df["Computation"] = "Decomposition"
part_df["Alpha"] = 0.2


# df = pd.concat([non_part_df, part_df])

df = non_part_df.merge(part_df, on=["id_for_x"], how="inner")
df = df.sort_values(by=["num_reachable_pts_y"])


new_df = pd.DataFrame(
    columns=["id_for_x", "num_reachable_pts", "Computation", "Alpha", "new_id", "pct"]
)

for index, row in df.iterrows():
    # add the point itself

    new_df.loc[len(new_df)] = [
        row["id_for_x"],
        row["num_reachable_pts_x"] + 1,
        row["Computation_x"],
        row["Alpha_x"],
        None,
        None,
    ]
    new_df.loc[len(new_df)] = [
        row["id_for_x"],
        row["num_reachable_pts_y"] + 1,
        row["Computation_y"],
        row["Alpha_y"],
        None,
        None,
    ]

new_id_count = 0
rows_count = 0

for i, row in new_df.iterrows():
    if rows_count == 0:
        new_df.at[i, "new_id"] = new_id_count
        new_df.at[i, "pct"] = (new_id_count / len(part_df)) * 100
        rows_count += 1
    else:
        new_df.at[i, "new_id"] = new_id_count
        new_df.at[i, "pct"] = (new_id_count / len(part_df)) * 100
        rows_count = 0
        new_id_count += 1

new_df = new_df.drop(columns=["id_for_x"])
new_df.to_csv(
    "/Users/avnikothari/Desktop/UCSD/Research/infeasible-recourse/graphs/timing.csv",
    index=False,
)
print("Saved!")
