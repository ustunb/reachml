import pathlib

import numpy as np
import pandas as pd

abs_path = pathlib.Path(__file__).parent.resolve()
bots_df = pd.read_csv(abs_path / "bots/bots.100k.csv")
humans_df = pd.read_csv(abs_path / "humans/humans.100k.csv")

bots_df = bots_df.drop(columns=["screen_name"])
bots_df.insert(loc=0, column="is_human", value=0)
humans_df = humans_df.drop(columns=["screen_name"])
humans_df.insert(loc=0, column="is_human", value=1)

new_df = pd.concat([bots_df, humans_df]).reset_index(drop=True)

source_dict = {
    "0": "source_other",
    "1": "source_web",
    "2": "source_mobile",
    "3": "source_app",
    "4": "source_automation",
    "5": "source_branding",
    "6": "source_news",
}

# Add sources.
source_dummies = pd.DataFrame(np.zeros((len(new_df), 7)), columns=source_dict.values())
for i, t in enumerate(new_df.source_identity):
    for source_code in t.strip("[]").split(";"):
        source = source_dict[source_code]
        source_dummies.iloc[i][source] = 1

new_df = pd.concat([new_df.drop(columns="source_identity"), source_dummies], axis=1)

processed_file = abs_path / "twitterbot_data.csv"
new_df.to_csv(processed_file, header=True, index=False)
print("results saved!")
