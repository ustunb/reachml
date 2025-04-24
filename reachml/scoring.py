from abc import ABC, abstractmethod
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cplex import SparsePair
from scipy.sparse import csr_matrix
from tqdm import tqdm

from .action_set import ActionSet
from .cplex_utils import has_solution
from .mip import EnumeratorMIP

RESP_BAR_COLOR = "#FFC000"

# matplotlib font params
plt.rcParams["font.size"] = 15


class ResponsivenessScorer(ABC):
    def __new__(
        cls, action_set, db=None, inter=None, cnts=None, method="auto", **kwargs
    ):
        """
        Factory method to create a ResponsivenessScorer instance
        """
        assert isinstance(action_set, ActionSet)
        if method == "auto":
            if action_set.can_enumerate or db is not None:
                return super().__new__(EnumeratingScorer)
            return super().__new__(SamplingScorer)
        if method == "enumerate":
            return super().__new__(EnumeratingScorer)
        if method == "sample":
            return super().__new__(SamplingScorer)
        raise ValueError(f"method {method} is not valid")

    def __init__(self, action_set, db=None, *args, **kwargs):
        self._action_set = action_set
        self._db = db
        self._inter = {} if args == () else args[0]
        self._act_feats = sorted(list(action_set.actionable_features))
        self._act_part_feats = [
            f for part in action_set.actionable_partition for f in part
        ]

    @property
    def action_set(self):
        return self._action_set

    @property
    def db(self):
        return self._db

    @property
    def inter(self):
        return self._inter

    @property
    def act_feats(self):
        return self._act_feats

    @property
    def act_part_feats(self):
        return self._act_part_feats

    @abstractmethod
    def score(self, x, pred_f, target=1):
        pass

    @abstractmethod
    def build_inter(self, x):
        pass

    @abstractmethod
    def _get_inter_key(self, x):
        pass

    def plot(self, score_lst=None, x_idx=None):
        if score_lst is None and x_idx is None:
            raise ValueError("Either scores or x_idx must be provided")

        if score_lst is None:
            score_lst = self.scores[x_idx]

        # Sort the scores and names together
        names = self.action_set.names
        sorted_data = sorted(
            zip(score_lst, names), reverse=False
        )  # Sort by score descending
        sorted_scores, sorted_names = zip(*sorted_data)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.barh(sorted_names, sorted_scores, color=RESP_BAR_COLOR)
        ax.set_xlabel("Responsiveness Score")
        ax.set_yticks(range(len(sorted_scores)))
        ax.set_xlim(0, 1)

        return fig

    def __call__(self, X, clf, save=True, **kwargs):
        """ """
        if isinstance(X, pd.DataFrame):
            X = X.values

        out = np.zeros((len(X), len(self.action_set)))

        for i, x in tqdm(enumerate(X), total=len(X)):
            out[i, :] = self.score(x, clf.predict, **kwargs)

        if save:
            self.X = X
            self.clf = clf
            self.scores = out

        return out

    def __reduce__(self):
        return (self.__class__, (self.action_set, self.db, self.inter))

    def _find_actions(self, x, aj, j, find_all=True):
        j_part = self.action_set.constraints.get_associated_features(j)

        if len(j_part) == 1:  # no need to find actions if partition size 1
            ej = np.eye(len(x))[j]
            out = aj * ej
            return out

        j_idx = np.where(np.array(j_part) == j)[0].item()
        x_part = x[j_part]
        action_set_part = self.action_set[j_part]

        R = EnumeratorMIP(action_set=action_set_part, x=x_part)
        cpx, idx = R.mip, R.indices
        cons = cpx.linear_constraints
        # adding constraint to match action
        cons.add(
            names=[f"match_c[{j_idx}]"],
            lin_expr=[SparsePair(ind=[f"c[{j_idx}]"], val=[1.0])],
            senses=["E"],
            rhs=[float(aj)],
        )
        cpx.solve()

        aj_acts = []
        while has_solution(cpx):
            names = idx.names
            acts = cpx.solution.get_values(names["c"])
            aj_acts.append(acts)

            # TODO: check if this is what we want
            if sum(map(bool, acts)) == 1 or not find_all:
                # found action that only changes j
                # means it is not necessary for other features to change
                break

            R.remove_actions([acts])
            cpx.solve()

        if len(aj_acts) == 0:
            return np.array([])

        out = np.zeros((len(aj_acts), len(x)))
        out[:, j_part] = np.vstack(aj_acts)

        return out

    def __repr__(self):
        return f"{self.__class__.__name__}"


class EnumeratingScorer(ResponsivenessScorer):
    def __init__(self, action_set, db=None, *args, **kwargs):
        super().__init__(action_set, db, *args, **kwargs)
        self._method = "enumerate" if db is None else "filter"

    @property
    def db(self):
        return self._db

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        assert value in ["enumerate", "filter"]
        if value == "filter":
            assert self.db is not None, "filter method requires reachable set database"
        self._method = value

    @property
    def generate_call(self):
        if self.method == "filter":
            return self._filter_1D
        return self._enumerate_1D

    def score(self, x, pred_f, target=1):
        """ """
        key = self._get_inter_key(x)
        if key not in self.inter:
            self.build_inter(x)

        x_inter = self.inter[key]
        score_out = np.zeros(len(self.action_set))

        for j, interv_1D in x_inter.items():
            n_interv = interv_1D.shape[0]

            if interv_1D.size == 0:  # no actions
                score_out[j] = 0
                continue

            xp = x + interv_1D
            yp = pred_f(xp)
            eq_target = (yp == target).astype(int)

            score_out[j] = eq_target.sum() / n_interv

        return score_out

    def build_inter(self, x):
        key = self._get_inter_key(x)
        if key in self.inter:
            return

        actions = {}
        for j in self.act_feats:
            actions[j] = self.generate_call(x, j)

        self._inter[key] = actions

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

    def _enumerate_1D(self, x, j, actions=True):
        """ """
        aj_lst = self.action_set[j].reachable_grid(x[j], return_actions=True)
        aj_lst = aj_lst[aj_lst != 0]

        if len(aj_lst) == 0:
            return np.array([])
        else:
            out = np.vstack([self._find_actions(x, aj, j) for aj in aj_lst])

        return out

    def _get_inter_key(self, x):
        return tuple(x[self.act_part_feats])

    def __reduce__(self):
        return (self.__class__, (self.action_set, self.db, self.inter))


class SamplingScorer(ResponsivenessScorer):
    def __init__(self, action_set, db=None, *args, **kwargs):
        super().__init__(action_set, db, *args, **kwargs)
        self._samp_cnts = {} if args == () else args[1]

    def score(self, x, pred_f, target=1, n=500):
        key = self._get_inter_key(x)
        if key not in self.inter:
            self.build_inter(x, n=n)

        x_inter = self.inter[key]
        score_out = np.zeros(len(self.action_set))

        for j, interv_1D in x_inter.items():
            if interv_1D.shape[0] == 0:
                score_out[j] = 0
                continue

            all_inter = interv_1D.toarray()
            all_inter_cnt = self._samp_cnts[key][j]

            if isinstance(all_inter_cnt, np.int32):
                all_inter_cnt = (
                    np.ones(all_inter.shape[0], dtype=np.int32) * all_inter_cnt
                )

            xp = x + all_inter
            yp = pred_f(xp)
            eq_target = (yp == target).astype(int)

            score_out[j] = eq_target.dot(all_inter_cnt) / all_inter_cnt.sum()

        return score_out

    def build_inter(self, x, n):
        key = self._get_inter_key(x)
        if key in self.inter:
            return

        actions = {}
        cnts = {}
        for j in self.act_feats:
            act_out, cnt_out = self._sample_1D(x, j, n=n)

            if act_out.shape[0] > 0:
                act_out = csr_matrix(act_out)
                if (cnt_out == cnt_out[0]).all():
                    cnt_out = cnt_out[0]

            actions[j], cnts[j] = act_out, cnt_out

        self._inter[key] = actions
        self._samp_cnts[key] = cnts

    def _sample_1D(self, x, j, n=500, actions=True):
        """ """
        if self.action_set[j].discrete:
            return self._sample_discrete(x, j, n, actions)
        return self._sample_continuous(x, j, n, actions)

    def _sample_continuous(self, x, j, n=500, actions=True):
        """ """
        low = self.action_set[j].get_action_bound(x[j], bound_type="lb")
        high = self.action_set[j].get_action_bound(x[j], bound_type="ub")

        if low == high:
            return np.array([]), []

        samp_remain = n
        act_dict, cnt_dict = {}, defaultdict(int)
        while samp_remain > 0:
            aj_lst = np.random.uniform(low=low, high=high, size=n)
            aj_uni, cnt = np.unique(aj_lst, return_counts=True)

            for aj, n_aj in zip(aj_uni, cnt):
                aj_acts = self._find_actions(x, aj, j, find_all=False)

                if aj_acts.shape[0] > 0:
                    act_dict[aj] = aj_acts
                    cnt_dict[aj] += n_aj
                    samp_remain -= n_aj

        out_act = np.vstack(list(act_dict.values()), dtype=np.float32)
        out_cnt = np.array(list(cnt_dict.values()), dtype=np.int32)

        return out_act, out_cnt

    def _sample_discrete(self, x, j, n=500, actions=True):
        """ """
        act_grid = self.action_set[j].reachable_grid(x[j], return_actions=True)
        non_zero_acts = act_grid[act_grid != 0]

        samp_remain = n
        remove = []
        act_dict, cnt_dict = {}, defaultdict(int)
        while samp_remain > 0:  # TODO: change to sample from all possible actions
            samp_acts = list(set(non_zero_acts) - set(remove))
            if len(samp_acts) == 0:
                return np.array([]), []

            aj_lst = np.random.choice(samp_acts, samp_remain, replace=True)
            aj_uni, cnt = np.unique(aj_lst, return_counts=True)

            for aj, n_aj in zip(aj_uni, cnt):
                aj_acts = self._find_actions(x, aj, j, find_all=False)

                if aj_acts.shape[0] == 0:
                    remove.append(aj)
                else:
                    act_dict[aj] = aj_acts
                    cnt_dict[aj] += n_aj
                    samp_remain -= n_aj

        out_act = np.vstack(list(act_dict.values()), dtype=np.float32)
        out_cnt = np.array(list(cnt_dict.values()), dtype=np.int32)

        return out_act, out_cnt

    def _get_inter_key(self, x):
        return tuple(x)

    def __reduce__(self):
        return (self.__class__, (self.action_set, self.db, self.inter, self._samp_cnts))
