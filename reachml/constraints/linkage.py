"""Linkage constraint relating source and target action magnitudes."""

import numpy as np

from .abstract import ActionabilityConstraint


class LinkActions(ActionabilityConstraint):
    """Constraint to link actions along features.

    Given a set of features `names`:
    - names[0] is the "source feature"
    - names[1:] are the "target features"
    - actions in the source induce an S[k]-unit change in each target.
    """

    def __init__(self, names, parent=None, scales=None, keep_bounds=True):
        """Initialize linkage with names and optional scales.

        Args:
            names: [source, target1, ...].
            parent: Optional ActionSet.
            scales: Non-zero scale per feature; normalized by source.
            keep_bounds: Whether to preserve target bounds.
        """
        assert len(names) >= 2
        scales = np.ones(len(names)) if scales is None else np.array(scales).flatten()
        assert len(scales) == len(names)
        assert np.count_nonzero(scales) == len(names) - 1
        super().__init__(names=names, parent=parent)
        self._source = self.names[0]
        self._targets = self.names[1:]
        self._scales = np.array(scales[1:]) / np.float(scales[0])
        self._parameters += ("source", "targets", "scales", "keep_bounds")

    def check_compatibility(self, action_set):
        """Return True if the constraint is compatible with the ActionSet.

        Called when attaching this constraint via `ActionSet.constraints.add`.
        """
        # check that scales are compatible
        step_compatability = [
            np.mod(scale, a.step_size) == 0 if a.discrete else True
            for a, scale in zip(action_set[self.targets], self.scales, strict=False)
        ]
        assert all(step_compatability)
        return True

    @property
    def source(self):
        """Source feature name."""
        return self._source

    @property
    def targets(self):
        """List of target feature names."""
        return self._targets

    @property
    def scales(self):
        """Scale factors for targets relative to the source."""
        return self._scales

    def __str__(self):
        """Human-readable description of this linkage."""
        name_list = ", ".join(f"`{n}`" for n in self.names)
        s = f"Link actions on [{name_list}]."
        if self._scaled:
            s += f"Each unit change in {self._source} leads to:" + ", ".join(
                [
                    "f{s:1.2f}-unit change in {n}"
                    for s, n in zip(self._targets, self._scales, strict=False)
                ]
            )
        return s

    def check_feasibility(self, x):
        """Return True if a vector/matrix `x` is viable under constraints."""
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a=x, axis=0)
        v = x[self.indices]
        out = self.check_feature_vector(v)
        return out

    def adapt(self, x):
        """Adapt the constraint to the feature vector `x`."""
        assert self.check_feasibility(x), f"{self.__class__} is infeasible at x = {str(x)}"
        return x[self.indices]

    def add_to_cpx(self, cpx, indices, x):
        """Add linear equalities linking source to targets in CPLEX."""
        from cplex import Cplex, SparsePair
        assert isinstance(cpx, Cplex)
        cons = cpx.linear_constraints
        self.adapt(x)
        pivot_index = self.indices[0]
        pivot_scale = self.scales[1]
        linked_indices = self.indices[1:]
        linked_scales = self.scales[1:] / pivot_scale
        for k, s in zip(linked_indices, linked_scales, strict=False):
            cons.add(
                names=[f"link_{self.id}_{pivot_index}_to_{k}"],
                lin_expr=[SparsePair(ind=[f"a[{pivot_index}]", f"a[{k}]"], val=[1.0, -s])],
                senses="E",
                rhs=[0.0],
            )
        return cpx, indices

    def add_to_scip(self, scip, indices, x):
        """Add linear equalities linking source to targets in SCIP."""
        from pyscipopt import Model
        assert isinstance(scip, Model)
        self.adapt(x)
        pivot_index = self.indices[0]
        pivot_scale = self.scales[1]
        linked_indices = self.indices[1:]
        linked_scales = self.scales[1:] / pivot_scale

        def getA(i):
            return scip.getVarByName(f"a[{i}]")

        a_pivot = getA(pivot_index)

        for k, s in zip(linked_indices, linked_scales, strict=False):
            name = f"link_{self.id}_{pivot_index}_to_{k}"
            a_k = getA(k)
            # Option 1: direct expression (cleanest)
            scip.addCons(a_pivot - s * a_k == 0.0, name=name)
            # Option 2 (equivalent): linear form with explicit lhs/rhs
            # scip.addConsLinear([a_pivot, a_k], [1.0, -s], lhs=0.0, rhs=0.0, name=name)

        return scip, indices
