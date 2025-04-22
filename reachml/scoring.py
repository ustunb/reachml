import numpy as np
from cplex import SparsePair
from tqdm import tqdm

from .action_set import ActionSet
from .cplex_utils import has_solution
from .mip import EnumeratorMIP


class ResponsivenessScorer:
    def __init__(self, action_set: ActionSet, reach_db=None, method="auto"):
        self.action_set = action_set
        self.reach_db = reach_db

        assert method in ["auto", "enumerate", "sample"]
        self.method = method

        if method == "auto":
            if action_set.can_enumerate or reach_db is not None:
                self.method = "enumerate"
            else:
                self.method = "sample"

        if self.reach_db is not None:
            assert action_set.can_enumerate, (
                "only pass in reachable set database with action sets that can be enumerated"
            )

        self.generate_call = (
            self._filter_1D
            if self.reach_db is not None
            else getattr(self, f"_{self.method}_1D")
        )
        self.inter = {}

        self.act_feats = sorted(list(self.action_set.actionable_features))
        self.act_part_feats = [
            f for part in self.action_set.actionable_partition for f in part
        ]

    def __call__(self, X, clf, save=True, **kwargs):
        """ """
        out = np.zeros((len(X), len(self.action_set)))

        for i, x in tqdm(enumerate(X)):
            out[i, :] = self.score(x, clf.predict, **kwargs)

        if save:
            self.X = X
            self.clf = clf
            self.scores = out

        return out

    def score(self, x, pred_f, target=1):
        """ """
        key = self._get_inter_key(x)
        if key not in self.inter:
            self.build_inter(x)

        x_inter = self.inter[key]
        score_out = np.zeros(len(self.action_set))

        for j, interv_1D in x_inter.items():
            n_interv = interv_1D.shape[0]

            if n_interv == 0:
                score_out[j] = 0
                continue

            xp = x + interv_1D
            yp = pred_f(xp)

            score_out[j] = np.count_nonzero(yp == target) / n_interv

        return score_out

    def build_inter(self, x):
        key = self._get_inter_key(x)
        if key in self.inter:
            return

        actions = {}
        for j in self.act_feats:
            interv_1D = self.generate_call(x, j)
            actions[j] = interv_1D

        self.inter[key] = actions

    def _filter_1D(self, x, j):
        """
        helper function to calculate responsiveness scores
        """
        rs_x = self.reach_db[x]
        actions = rs_x.actions

        rs_x = self.reach_db[x]
        actions = rs_x.actions

        if j not in self.action_set.actionable_features:
            return np.array([])

        change_j = actions[actions[:, j] != 0]
        if change_j.shape[0] == 0:  # no actions change j
            return np.array([])

        # check if there is 1-D action
        s = change_j[np.abs(change_j).clip(0, 1).sum(axis=1) == 1]
        if s.shape[0] > 0:
            return s

        joint_feats = set(self.action_set.constraints.get_associated_features(j))

        # check if there isn't any potential action either
        mask = np.ones_like(x, dtype=bool)
        mask[list(joint_feats)] = False
        rest_0_idx = (change_j[:, mask] == 0).all(axis=1)
        rem_act = change_j[rest_0_idx]

        if rem_act.shape[0] == 0:
            return np.array([])

        marg_acts = []
        feat_sets = []
        for a in rem_act:
            non_zero = set(np.where(a != 0)[0])
            if len(non_zero) > 0 and j in non_zero and non_zero.issubset(joint_feats):
                feat_sets.append(non_zero)
                marg_acts.append(a)

        set_size = np.array(list(map(len, feat_sets)))
        min_size = np.where(set_size == set_size.min())[0]

        return np.vstack(marg_acts)[min_size]

    def __find_actions(self, x, aj_lst, j):
        j_part = self.action_set.constraints.get_associated_features(j)
        j_idx = np.where(np.array(j_part) == j)[0].item()

        x_part = x[j_part]
        action_set_part = self.action_set[j_part]

        acts_found = []
        for aj in aj_lst:
            R = EnumeratorMIP(action_set=action_set_part, x=x_part)
            cpx, idx = R.mip, R.indices
            cons = cpx.linear_constraints
            # adding constraint to match action
            cons.add(
                names=[f"match_c[{j_idx}]"],
                lin_expr=[SparsePair(ind=[f"c[{j_idx}]"], val=[1.0])],
                senses=["E"],
                rhs=[aj],
            )
            cpx.solve()

            while has_solution(cpx):
                names = idx.names
                acts = cpx.solution.get_values(names["c"])
                acts_found.append(acts)

                if sum(map(bool, acts)) == 1:
                    # found action that only changes j
                    # means it is not necessary for other features to change
                    break

                R.remove_actions([acts])
                cpx.solve()

        if len(acts_found) == 0:
            return np.array([])

        out = np.zeros((len(acts_found), len(x)))
        out[:, j_part] = np.vstack(acts_found)
        return out

    def _sample_1D(self, x, j, n=20, actions=True):
        """ """
        if self.action_set[j].discrete:
            act_grid = self.action_set[j].reachable_grid(x[j], return_actions=True)
            aj_lst = np.random.choice(act_grid[act_grid != 0], n, replace=False)
        else:
            aj_lst = np.random.uniform(
                lb=self.action_set[j].get_action_bound(x[j], bound_type="lb"),
                ub=self.action_set[j].get_action_bound(x[j], bound_type="ub"),
                size=n,
            )

        out = self.__find_actions(x, aj_lst, j)
        return out

    def _enumerate_1D(self, x, j, actions=True):
        """ """
        aj_lst = self.action_set[j].reachable_grid(x[j], return_actions=True)
        aj_lst = aj_lst[aj_lst != 0]

        out = self.__find_actions(x, aj_lst, j)
        return out

    def __find_actions(self, x, aj_lst, j):
        j_part = self.action_set.constraints.get_associated_features(j)
        j_idx = np.where(np.array(j_part) == j)[0].item()

        x_part = x[j_part]
        action_set_part = self.action_set[j_part]

        acts_found = []
        for aj in aj_lst:
            R = EnumeratorMIP(action_set=action_set_part, x=x_part)
            cpx, idx = R.mip, R.indices
            cons = cpx.linear_constraints
            # adding constraint to match action
            cons.add(
                names=[f"match_c[{j_idx}]"],
                lin_expr=[SparsePair(ind=[f"c[{j_idx}]"], val=[1.0])],
                senses=["E"],
                rhs=[aj],
            )
            cpx.solve()

            while has_solution(cpx):
                names = idx.names
                acts = cpx.solution.get_values(names["c"])
                acts_found.append(acts)

                if sum(map(bool, acts)) == 1:
                    # found action that only changes j
                    # means it is not necessary for other features to change
                    break

                R.remove_actions([acts])
                cpx.solve()

        if len(acts_found) == 0:
            return np.array([])

        out = np.zeros((len(acts_found), len(x)))
        out[:, j_part] = np.vstack(acts_found)
        return out

    def _get_inter_key(self, x):
        if self.method == "sample":
            return tuple(x)

        # if enumerate, store by actionable partition (sibling points)
        return tuple(x[self.act_part_feats])
