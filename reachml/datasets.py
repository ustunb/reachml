import pandas as pd

GH_DATA_URL = "https://raw.githubusercontent.com/ustunb/reachml/main/data/"


def givemecredit_cts_slim(n=100, label=[-1, 1]):
    filename = "givemecredit_cts.csv"
    data_df = pd.read_csv(GH_DATA_URL + filename)
    data_df = data_df.sample(n=n, random_state=0).astype(float)

    if label == [0, 1]:
        data_df.iloc[:, 0] = data_df.iloc[:, 0].replace(-1, 0)

    X, y = data_df.iloc[:, 1:], data_df.iloc[:, 0]

    return X, y


def credit():
    filename = "credit.csv"
    data_df = pd.read_csv(GH_DATA_URL + filename)

    X, y = data_df.iloc[:, 1:], data_df.iloc[:, 0]

    return X, y
