"""
Test Strategy
----
target_type: ["int", "bool", "float"]
target_actionability: [True, False]
target_monotonicity: [0, 1, -1]
n_targets_per_constraint: [1,2],
constraints: [1, multiple_independent, multiple_overlapping]
action_on_source_violates_target_type: [True, False]
action_on_source_violates_target_bounds: [True, False]
"""
import pytest
import pandas as pd
import numpy as np
import itertools
from reachml import ActionSet
from reachml.reachable_set import EnumeratedReachableSet
from reachml.constraints.directional_linkage import DirectionalLinkage
from reachml.utils import SUPPORTED_SOLVERS

sortrows = lambda v: v[np.lexsort(v.T, axis=0), :]


def create_linked_reachable_set(A, x, source, targets, scales):
    d = len(x)
    assert d >= 2
    assert isinstance(targets, list) and len(targets) == (d - 1)
    assert isinstance(scales, list) and len(scales) == d
    scales = scales[1:]
    x_s = x[A.get_feature_indices(source)]
    a_s = A[source].reachable_grid(x_s, return_actions=True)
    a_st = np.tile(a_s[:, None], d - 1) * scales
    source_actions = np.column_stack((a_s, a_st))
    joint_actions = []
    for target in targets:
        target_idx = A.get_feature_indices(target)
        x_t = x[target_idx]
        a_t = A[target].reachable_grid(x_t, return_actions=True)
        target_actions = np.zeros(shape=(len(a_t), d))
        target_actions[:, target_idx] = a_t
        joint_actions.append(
            np.vstack(
                [
                    np.add(source_actions, target_action)
                    for target_action in target_actions
                ]
            )
        )
    joint_actions = np.vstack(joint_actions)
    reachable_set = np.add(joint_actions, x)
    reachable_set = np.fliplr(sortrows(np.fliplr(reachable_set)))
    x_idx = np.flatnonzero(np.all(reachable_set == x, axis=1))
    reachable_set = np.delete(reachable_set, x_idx, axis=0)
    reachable_set = np.insert(reachable_set, 0, x, axis=0)

    return reachable_set


def get_2d_test_case(t):
    X = pd.DataFrame(
        columns=["x0", "x1", "x2"], data=[[0, 0, -1], [1, 0, 0], [1, 1, 2], [1, 1, 3]]
    )
    source = "x0"
    if t == "bool":
        targets = ["x1"]
    elif t == "int":
        targets = ["x2"]
    names = [source] + targets
    X = X[names]
    A = ActionSet(X)
    out = {
        "source": source,
        "targets": targets,
        "X": X,
        "A": A,
    }
    return out


@pytest.fixture(params=[True, False])
def target_actionability(request):
    return request.param


@pytest.fixture(params=[0, 1, -1])
def target_monotonicity(request):
    return request.param


@pytest.fixture(params=["bool", "int"])
def target_type(request):
    return request.param


@pytest.fixture(params=[[1], [-1], [2]])
def target_scale(request):
    return request.param


def test_initialization(
    target_type, target_scale, target_actionability, target_monotonicity
):
    test_case = get_2d_test_case(t=target_type)
    X = test_case["X"]
    A = test_case["A"]
    source = test_case["source"]
    targets = test_case["targets"]
    names = [source] + targets
    n_targets = len(targets)
    assert n_targets == 1
    scales = np.array([1.0] + target_scale)

    # setup actionability, monotonicity
    A[targets].actionable = target_actionability
    A[targets].step_direction = target_monotonicity

    # initialize constraint
    cons = DirectionalLinkage(names=names, scales=scales, keep_bounds=False)

    # test print
    print(str(cons))

    # test parameters
    assert cons.source == names[0]
    assert set(cons.targets) == set(targets)
    assert len(cons.scales) == len(targets)

    # test that all linked features are within the partition
    const_id = A.constraints.add(cons)
    part = A.get_feature_indices(names)
    assert part in A.partition
    assert part in A.actionable_partition

    # add
    assert cons.id == const_id

    # adding it again raises an error
    with pytest.raises(AssertionError):
        A.constraints.add(cons)

    # dropping works
    dropped = A.constraints.drop(const_id)

    # assert constraint is dropped
    assert dropped
    assert cons not in A.constraints

    # cannot drop again
    dropped = A.constraints.drop(const_id)
    assert dropped is False


test_cases_with_valid_type_int = list(
    itertools.product(["int"], [1, 2], [True, False], [0, 1, -1])
)
test_cases_with_valid_type_bool = list(
    itertools.product(["bool"], [1, -1], [True, False], [0, 1, -1])
)
test_cases_with_valid_type = (
    test_cases_with_valid_type_int + test_cases_with_valid_type_bool
)


@pytest.mark.parametrize(
    "target_type,target_scale,target_actionability,target_monotonicity,solver",
    [(*case, solver) for case in test_cases_with_valid_type for solver in SUPPORTED_SOLVERS],
)
def test_enumeration_with_one_target(
    target_type, target_scale, target_actionability, target_monotonicity, solver
):
    # setup test case
    test_case = get_2d_test_case(t=target_type)
    X = test_case["X"]
    A = test_case["A"]
    source = test_case["source"]
    targets = test_case["targets"]
    scales = [1.0] + [target_scale]

    # setup actionability, monotonicity
    A[targets].step_direction = target_monotonicity
    A[targets].actionable = target_actionability

    # setup expected sets
    expected_reachable_sets = {
        tuple(x): create_linked_reachable_set(A, x, source, targets, scales)
        for x in X.values
    }

    # add link constraint
    link_constraint = DirectionalLinkage(
        names=[source] + targets, scales=scales, keep_bounds=False
    )
    A.constraints.add(constraint=link_constraint)

    for idx, x in enumerate(X.values):
        expected_set = expected_reachable_sets.get(tuple(x))
        reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver=solver)
        reachable_set.generate()
        assert reachable_set.complete
        try:
            assert np.isclose(sortrows(reachable_set.X), sortrows(expected_set)).all()
        except AssertionError:
            print(A)
            print(f"\nx={str(x)}")
            print(f"\nexpected_set.X ({expected_set.shape[0]} points)\n{expected_set}")
            print(
                f"\nreachable_set.X ({reachable_set.X.shape[0]} points)\n{reachable_set.X}"
            )
            raise AssertionError()
                
@pytest.mark.parametrize(
    "target_type,target_scale,target_actionability,target_monotonicity",
    test_cases_with_valid_type,
)
def test_enumeration_with_one_target_scip_and_cplex(target_type, target_scale, target_actionability, target_monotonicity):
    try:
        assert "scip" in SUPPORTED_SOLVERS and "cplex" in SUPPORTED_SOLVERS
    except AssertionError:
        print("SCIP and CPLEX are not both supported solvers.")
        pytest.skip()
    # setup test case
    test_case = get_2d_test_case(t=target_type)
    X = test_case["X"]
    A = test_case["A"]
    source = test_case["source"]
    targets = test_case["targets"]
    scales = [1.0] + [target_scale]

    # setup actionability, monotonicity
    A[targets].step_direction = target_monotonicity
    A[targets].actionable = target_actionability

    # setup expected sets
    expected_reachable_sets = {
        tuple(x): create_linked_reachable_set(A, x, source, targets, scales)
        for x in X.values
    }

    # add link constraint
    link_constraint = DirectionalLinkage(
        names=[source] + targets, scales=scales, keep_bounds=False
    )
    A.constraints.add(constraint=link_constraint)

    for idx, x in enumerate(X.values):
        expected_set = expected_reachable_sets.get(tuple(x))
        scip_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="scip")
        scip_reachable_set.generate()
        assert scip_reachable_set.complete
        try:
            assert np.isclose(sortrows(scip_reachable_set.X), sortrows(expected_set)).all()
        except AssertionError:
            print(A)
            print(f"\nx={str(x)}")
            print(f"\nexpected_set.X ({expected_set.shape[0]} points)\n{expected_set}")
            print(
                f"\nreachable_set.X ({scip_reachable_set.X.shape[0]} points)\n{scip_reachable_set.X}"
            )
            raise AssertionError()
        cplex_reachable_set = EnumeratedReachableSet(x=x, action_set=A, solver="cplex")
        cplex_reachable_set.generate()
        assert cplex_reachable_set.complete
        try:
            assert np.isclose(sortrows(cplex_reachable_set.X), sortrows(expected_set)).all()
        except AssertionError:
            print(A)
            print(f"\nx={str(x)}")
            print(f"\nexpected_set.X ({expected_set.shape[0]} points)\n{expected_set}")
            print(
                f"\nreachable_set.X ({cplex_reachable_set.X.shape[0]} points)\n{cplex_reachable_set.X}"
            )
            raise AssertionError()
        assert np.isclose(sortrows(scip_reachable_set.X), sortrows(cplex_reachable_set.X)).all()

def test_enumeration_with_multiple_targets():
    pytest.skip()


def test_enumeration_with_overlapping_constraints():
    pytest.skip()


if __name__ == "__main__":
    pytest.main()
