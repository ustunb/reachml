"""CPLEX backend implementing the MIPBackend protocol."""

from __future__ import annotations

from functools import reduce
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
from cplex import Cplex, SparsePair

from ...action_set import ActionSet
from ..backend_interface import MIPBackend, MIPSettings
from .cplex_utils import (
    CplexGroupedVariableIndices,
    combine,
    get_cpx_variable_args,
    get_cpx_variable_types,
    get_mip_stats,
    has_solution,
)


class CplexBackend(MIPBackend):
    solver_name = "cplex"

    def build_model(
        self, action_set: ActionSet, x: np.ndarray, actionable_indices: List[int]
    ) -> Tuple[object, object]:
        cpx = Cplex()
        cpx.set_problem_type(cpx.problem_type.MILP)
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        vars = cpx.variables
        cons = cpx.linear_constraints

        a_lb = action_set.get_bounds(x, bound_type="lb")
        a_ub = action_set.get_bounds(x, bound_type="ub")
        a_pos_max = np.abs(a_ub)
        a_neg_max = np.abs(a_lb)
        a_types = get_cpx_variable_types(action_set, actionable_indices)

        variable_args = {
            "a": get_cpx_variable_args(
                obj=0,
                name=[f"a[{j}]" for j in actionable_indices],
                lb=a_lb,
                ub=a_ub,
                vtype=a_types,
            ),
            "a_pos": get_cpx_variable_args(
                obj=1.0,
                name=[f"a[{j}]_pos" for j in actionable_indices],
                lb=0.0,
                ub=a_pos_max,
                vtype=a_types,
            ),
            "a_neg": get_cpx_variable_args(
                obj=1.0,
                name=[f"a[{j}]_neg" for j in actionable_indices],
                lb=0.0,
                ub=a_neg_max,
                vtype=a_types,
            ),
            "a_sign": get_cpx_variable_args(
                obj=0.0,
                name=[f"a[{j}]_sign" for j in actionable_indices],
                lb=0.0,
                ub=1.0,
                vtype="B",
            ),
            "c": get_cpx_variable_args(
                obj=0,
                name=[f"c[{j}]" for j in actionable_indices],
                lb=a_lb,
                ub=a_ub,
                vtype=a_types,
            ),
        }
        vars.add(**reduce(combine, variable_args.values()))

        indices = CplexGroupedVariableIndices()
        indices.append_variables(variable_args)
        names = indices.names

        for j, a_j, a_pos_j, a_neg_j, a_sign_j, c_j in zip(
            actionable_indices,
            names["a"],
            names["a_pos"],
            names["a_neg"],
            names["a_sign"],
            names["c"],
            strict=False,
        ):
            cons.add(
                names=[f"abs_val_pos_{a_j}"],
                lin_expr=[SparsePair(ind=[a_pos_j, a_j], val=[1.0, -1.0])],
                senses="G",
                rhs=[0.0],
            )
            cons.add(
                names=[f"abs_val_neg_{a_j}"],
                lin_expr=[SparsePair(ind=[a_neg_j, a_j], val=[1.0, 1.0])],
                senses="G",
                rhs=[0.0],
            )
            cons.add(
                names=[f"set_{a_j}_sign_pos"],
                lin_expr=[SparsePair(ind=[a_pos_j, a_sign_j], val=[1.0, -a_pos_max[j]])],
                senses="L",
                rhs=[0.0],
            )
            cons.add(
                names=[f"set_{a_j}_sign_neg"],
                lin_expr=[SparsePair(ind=[a_neg_j, a_sign_j], val=[1.0, a_neg_max[j]])],
                senses="L",
                rhs=[a_neg_max[j]],
            )
            cons.add(
                names=[f"set_{a_j}"],
                lin_expr=[SparsePair(ind=[a_j, a_pos_j, a_neg_j], val=[1.0, -1.0, 1.0])],
                senses="E",
                rhs=[0.0],
            )
            cons.add(
                names=[f"set_{c_j}"],
                lin_expr=[SparsePair(ind=[c_j, a_j], val=[1.0, -1.0])],
                senses="E",
                rhs=[0.0],
            )

        return cpx, indices

    def configure(self, model: Cplex, print_flag: bool) -> Cplex:
        p = model.parameters
        p.emphasis.numerical.set(1)
        p.mip.tolerances.integrality.set(1e-7)
        p.mip.tolerances.mipgap.set(0.0)
        p.mip.tolerances.absmipgap.set(0.0)
        p.mip.display.set(print_flag)
        p.simplex.display.set(print_flag)
        p.paramdisplay.set(print_flag)
        if not print_flag:
            model.set_results_stream(None)
            model.set_log_stream(None)
            model.set_error_stream(None)
            model.set_warning_stream(None)
        return model

    def add_constraints(self, model, indices, action_set: ActionSet, x: np.ndarray):
        for con in action_set.constraints:
            model, indices = con.add_to_cpx(cpx=model, indices=indices, x=x)
        return model, indices

    # Solve/inspect
    def solve(self, model: Cplex) -> None:
        model.solve()

    def has_solution(self, model: Cplex) -> bool:
        return has_solution(model)

    def read_vectors(self, model: Cplex, indices, names: List[str]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for nm in names:
            out[nm] = np.array(model.solution.get_values(indices.names[nm]))
        return out

    def add_nogood(
        self,
        model: Cplex,
        indices: CplexGroupedVariableIndices,
        actions: List[np.ndarray],
        actionable_indices: List[int],
        settings: MIPSettings,
    ) -> Tuple[object, object, int]:
        vars = model.variables
        cons = model.linear_constraints

        a_ub = indices.ub["c"]
        a_lb = indices.lb["c"]
        a_types = indices.types["c"]
        d = len(actionable_indices)

        n_points = len(actions)
        A_nogood = np.vstack(actions)
        D_pos = a_ub - A_nogood
        D_neg = A_nogood - a_lb

        start_index = indices.counts.get("nogood", 0)
        point_indices = range(start_index, start_index + n_points)

        variable_args = {
            "delta_pos": get_cpx_variable_args(
                obj=0.0,
                name=[f"delta[{j, k}]_pos" for k in point_indices for j in actionable_indices],
                lb=0.0,
                ub=list(chain.from_iterable(D_pos.tolist())),
                vtype=a_types * n_points,
            ),
            "delta_neg": get_cpx_variable_args(
                obj=0.0,
                name=[f"delta[{j, k}]_neg" for k in point_indices for j in actionable_indices],
                lb=0.0,
                ub=list(chain.from_iterable(D_neg.tolist())),
                vtype=a_types * n_points,
            ),
            "delta_sign": get_cpx_variable_args(
                obj=0.0,
                name=[
                    f"delta[{j, k}]_sign" for k in point_indices for j in actionable_indices
                ],
                lb=0.0,
                ub=1.0,
                vtype="B",
            ),
        }

        vars.add(**reduce(combine, variable_args.values()))
        indices.append_variables(variable_args)
        names = indices.names

        for _idx, (k, ak, Dp_k, Dn_k) in enumerate(
            zip(point_indices, actions, D_pos, D_neg, strict=False)
        ):
            # sum_vars = sum(
            #     names["delta_pos"][j + k * d] for j in actionable_indices
            # ) + sum(names["delta_neg"][j + k * d] for j in actionable_indices)

            cons.add(
                names=[f"sum_abs_dist_val_{k}"],
                lin_expr=[
                    SparsePair(
                        ind=[
                            *[f"delta[{j, k}]_pos" for j in actionable_indices],
                            *[f"delta[{j, k}]_neg" for j in actionable_indices],
                        ],
                        val=[*([1.0] * d), *([1.0] * d)],
                    )
                ],
                senses="G",
                rhs=[float(settings.eps_min)],
            )

            for idx_j, j in enumerate(actionable_indices):
                pos = f"delta[{j, k}]_pos"
                neg = f"delta[{j, k}]_neg"
                sign = f"delta[{j, k}]_sign"
                c_j = f"c[{j}]"

                cons.add(
                    names=[f"nogood_pos_if_{j}_{k}"],
                    lin_expr=[SparsePair(ind=[pos, sign], val=[1.0, -float(Dp_k[idx_j])])],
                    senses="L",
                    rhs=[0.0],
                )
                cons.add(
                    names=[f"nogood_neg_if_{j}_{k}"],
                    lin_expr=[
                        SparsePair(ind=[neg, sign], val=[1.0, float(Dn_k[idx_j])])
                    ],
                    senses="L",
                    rhs=[float(Dn_k[idx_j])],
                )
                cons.add(
                    names=[f"nogood_dist_{j}_{k}"],
                    lin_expr=[
                        SparsePair(ind=[c_j, pos, neg], val=[1.0, -1.0, 1.0])
                    ],
                    senses="E",
                    rhs=[float(ak[idx_j])],
                )

        indices.counts["nogood"] = start_index + len(A_nogood)
        return model, indices, len(A_nogood)

    def stats(self, model: Cplex) -> Dict:
        return get_mip_stats(model)

    # Generic constraint operations
    def add_linear_constraint(
        self,
        model: Cplex,
        indices: CplexGroupedVariableIndices,
        name: str,
        terms: List[tuple[str, int, float]],
        sense: str,
        rhs: float,
    ) -> None:
        cons = model.linear_constraints
        var_names = []
        coeffs = []
        for group, idx, coef in terms:
            var_names.append(indices.names[group][idx])
            coeffs.append(float(coef))
        cons.add(names=[name], lin_expr=[SparsePair(ind=var_names, val=coeffs)], senses=[sense], rhs=[float(rhs)])

    def delete_constraint(self, model: Cplex, indices: CplexGroupedVariableIndices, name: str) -> None:
        model.linear_constraints.delete(name)

    def solution_status(self, model: Cplex) -> str:
        info = get_mip_stats(model)
        return str(info.get("status", "unknown"))
