"""Additive linkage constraint linking source action to target total changes."""

import numpy as np

from .abstract import ActionabilityConstraint


class AdditiveLinkage(ActionabilityConstraint):
    """Link source action to a linear combo of target total changes.

    Given feature list `names`:
    - names[0]: source feature
    - names[1:]: target features
    """

    def __init__(self, names, coeffs, linkage_type="E", parent=None):
        """Initialize linkage with names, coefficients, and type.

        Args:
            names: [source, target1, ...].
            coeffs: Coefficients aligned with `names`.
            linkage_type: One of {"E", "G", "L"}.
            parent: Optional `ActionSet`.
        """
        assert len(names) >= 2, (
            "At least two feature names (source and one target) must be provided."
        )
        coeffs_arr = np.array(coeffs).flatten()
        assert len(coeffs_arr) == len(names), "Length of coeffs must match length of names."
        assert np.count_nonzero(coeffs_arr) == len(coeffs_arr), "All coefficients must be non-zero."
        valid_linkage_types = ["E", "G", "L"]
        assert linkage_type in valid_linkage_types, (
            f"Invalid linkage_type '{linkage_type}'. Must be one of {valid_linkage_types}."
        )

        self._source = names[0]
        self._targets = list(names[1:])
        self._source_coeff_val = coeffs_arr[0]
        self._target_coeffs_list = list(coeffs_arr[1:])
        self._linkage_type_str = linkage_type

        self._parameters = ("source", "targets", "source_coeff", "target_coeffs", "linkage_type")

        # self.names and self.indices are set by super().__init__
        super().__init__(names=names, parent=parent)

    @property
    def source(self):
        """Source feature name."""
        return self._source

    @property
    def targets(self):
        """List of target feature names."""
        return self._targets

    @property
    def source_coeff(self):
        """Coefficient for the source action term."""
        return self._source_coeff_val

    @property
    def target_coeffs(self):
        """Coefficients for target total-change terms."""
        return self._target_coeffs_list

    @property
    def linkage_type(self):
        """Linkage sense: "E" (==), "G" (>=), or "L" (<=)."""
        return self._linkage_type_str

    def check_compatibility(self, action_set):
        """Return True if this constraint is compatible with `action_set`."""
        # TODO
        return True

    # def __str__(self):
    #     target_terms_str_list = []
    #     for coeff, name in zip(self.target_coeffs, self.targets):
    #         target_terms_str_list.append(f"{coeff:+.2f}*c[{name}]")

    #     sum_target_expr_str = "0"
    #     if target_terms_str_list:
    #         sum_target_expr_str = " + ".join(target_terms_str_list).replace("+ -", "- ")

    #     op_map = {"E": "==", "G": ">=", "L": "<="}
    #     op_str = op_map.get(self.linkage_type, self.linkage_type)

    #     source_action_str = f"a[{self.source}]"
    #     source_term_str = f"{self.source_coeff:+.2f}*{source_action_str}"

    #     return f"AdditiveLinkage: {sum_target_expr_str} {op_str} {source_term_str.lstrip('+')}"

    def check_feasibility(self, x):
        """Return True if a vector or matrix `x` is feasible under the constraint."""
        x = np.array(x)
        # if matrix then apply this function again for each point in the matrix
        if x.ndim == 2 and x.shape[0] > 1:
            return np.apply_over_axes(self.check_feasibility, a=x, axis=0)
        v = x[self.indices]
        out = self.check_feature_vector(v)
        return out

    def adapt(self, x):
        """Adapt the constraint to the provided feature vector `x`."""
        assert self.check_feasibility(x), f"{self.__class__} is infeasible at x = {str(x)}"
        return x[self.indices]

    def add_to_cpx(self, cpx, indices, x):
        """Add the additive linkage constraint to a CPLEX model."""
        from cplex import Cplex, SparsePair
        assert isinstance(cpx, Cplex)
        cons = cpx.linear_constraints

        # self.indices contains global problem indices: [source_idx, target1_idx, ...]
        source_idx = self.indices[0]
        target_indices_list = self.indices[1:]

        a_source_var_name = f"a[{source_idx}]"
        c_target_var_names = [f"c[{idx}]" for idx in target_indices_list]

        cplex_expr_vars = []
        cplex_expr_coeffs = []

        cplex_expr_vars.append(a_source_var_name)
        cplex_expr_coeffs.append(float(self.source_coeff))

        for coeff, var_name in zip(self.target_coeffs, c_target_var_names, strict=False):
            cplex_expr_vars.append(var_name)
            cplex_expr_coeffs.append(-float(coeff))

        lin_expr_sparse_pair = [SparsePair(ind=cplex_expr_vars, val=cplex_expr_coeffs)]

        sense_str = self.linkage_type
        rhs_val = [0.0]

        constraint_name = f"additive_linkage_{self.source}_targets"
        # To ensure unique constraint names if multiple instances, consider adding a unique ID.
        # For example, by using hash(tuple(sorted(self.target_names))) or similar.

        cons.add(
            names=[constraint_name], lin_expr=lin_expr_sparse_pair, senses=sense_str, rhs=rhs_val
        )

        return cpx, indices
