import pathlib
import numpy as np
import pytest
import pandas as pd
from reachml.action_set import ActionSet
from reachml.database import ReachableSetDatabase
from reachml.paths import tests_dir


@pytest.fixture()
def test_case():
    names = ["Age_lt_25", "Age_in_25_to_40", "Age_in_40_to_59", "Age_geq_60"]
    X = pd.read_csv(tests_dir / "credit.csv")[names]

    A = ActionSet(X)

    out = {"X": X, "A": A, "names": names}

    return out


def test_content_addressable_encoding_integer(test_case):
    x1 = np.array([1, 2, 3])
    x2 = np.array([1, 2, 3], dtype=np.float32)

    # precision doesn't matter for integers
    low_precision = 2
    key1 = ReachableSetDatabase(test_case["A"], precision=low_precision).array_to_key(x1)
    key2 = ReachableSetDatabase(test_case["A"], precision=low_precision).array_to_key(x2)
    assert key1 == key2

    high_precision = 6
    key3 = ReachableSetDatabase(test_case["A"], precision=high_precision).array_to_key(x1)
    key4 = ReachableSetDatabase(test_case["A"], precision=high_precision).array_to_key(x2)

    assert key3 == key4


def test_content_addressable_encoding_float(test_case):
    x1 = np.array([0.1, 0.2, 0.3])
    x2 = np.array([0.10000002, 0.2, 0.3])

    low_precision = 4
    key1 = ReachableSetDatabase(test_case["A"], precision=low_precision).array_to_key(x1)
    key2 = ReachableSetDatabase(test_case["A"], precision=low_precision).array_to_key(x2)
    assert key1 == key2

    high_precision = 8
    key3 = ReachableSetDatabase(test_case["A"], precision=high_precision).array_to_key(x1)
    key4 = ReachableSetDatabase(test_case["A"], precision=high_precision).array_to_key(x2)
    assert key3 != key4


def test_generate_database_from_dataframe(test_case, tmpdir):
    db_path = pathlib.Path(tmpdir) / "test_db"

    all_keys = range(5)
    X = test_case["X"].iloc[all_keys]
    U = np.unique(X.values, axis=0)
    A = test_case["A"]

    db1 = ReachableSetDatabase(A, path=db_path, enumerate=True)
    assert len(db1.keys()) == 0
    db1.generate(X)
    assert len(db1.keys()) == len(U)

    for _, x in X.iterrows():
        reachable_set = db1[x]
        assert pytest.approx(reachable_set[0]) == x
        assert len(reachable_set) >= 1


def test_generate_database_from_numpy(test_case, tmpdir):
    db_path = pathlib.Path(tmpdir) / "test_db"

    all_keys = range(5)
    X = test_case["X"].iloc[all_keys].values
    U = np.unique(X, axis=0)
    A = test_case["A"]

    db1 = ReachableSetDatabase(A, path=db_path)
    assert len(db1.keys()) == 0
    db1.generate(X)
    assert len(db1.keys()) == len(U)

    for x in X:
        reachable_set = db1[x]
        assert pytest.approx(reachable_set[0]) == x
        assert len(reachable_set) >= 1


def test_database_persistence(test_case, tmpdir):
    db_path = pathlib.Path(tmpdir) / "test_db"

    all_keys = range(5)
    X = test_case["X"].iloc[all_keys]
    A = test_case["A"]

    db1 = ReachableSetDatabase(A, path=db_path)
    assert len(db1.keys()) == 0
    db1.generate(X)
    prev_keys = db1.keys()
    del db1

    db2 = ReachableSetDatabase(A, path=db_path)
    assert np.all(db2.keys() == prev_keys)


if __name__ == "__main__":
    pytest.main()
