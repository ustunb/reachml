"""SCIP (PySCIPOpt) backend implementing the MIPBackend protocol."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pyscipopt import Model, quicksum

from ...action_set import ActionSet
from ..backend_interface import MIPBackend, MIPSettings
from .scip_utils import (
    SCIPGroupedVariableIndices,
    get_scip_stats,
    get_scip_variable_args,
    get_scip_variable_types,
)


class ScipBackend(MIPBackend):
    solver_name = "scip"

    def build_model(self, action_set, x, actionable_indices):
        # Set up SCIP object
        model = Model()
        model.setParam("presolving/convertinttobin/maxdomainsize", 0)
        model.setParam("presolving/maxrounds", 0)
        #model.setParam("presolving/linear/maxrounds", 0)

        # variable parameters
        a_lb = action_set.get_bounds(x, bound_type="lb")
        a_ub = action_set.get_bounds(x, bound_type="ub")
        a_pos_max = np.abs(a_ub)
        a_neg_max = np.abs(a_lb)
        
        a_types = get_scip_variable_types(action_set, actionable_indices)
        variable_args = {
            "a": get_scip_variable_args(
                name=[f"a[{j}]" for j in actionable_indices],
                vtype="C",
                lb=a_lb,
                ub=a_ub,
                obj=0.0
            ),
            "a_pos": get_scip_variable_args(
                name=[f"a[{j}]_pos" for j in actionable_indices],
                vtype=a_types,
                lb=0,
                ub=np.abs(a_ub),
                obj=1.0
            ),
            "a_neg" : get_scip_variable_args(
                name=[f"a[{j}]_neg" for j in actionable_indices],
                vtype=a_types,
                lb=0.0,
                ub=np.abs(a_lb),
                obj=1.0
            ),
            "a_sign" : get_scip_variable_args(
                name=[f"a[{j}]_sign" for j in actionable_indices],
                vtype="B",
                lb=0.0,
                ub=1.0,
                obj=0.0
            ),
            "c": get_scip_variable_args(
                name=[f"c[{j}]" for j in actionable_indices],
                vtype="C",
                lb=a_lb,
                ub=a_ub,
                obj=0.0
            )
        }
        
        indices = SCIPGroupedVariableIndices(model, variable_args)
        scip_vars = {} 
        for group, d in variable_args.items():
            names = d.get("names", d.get("name"))
            lb    = d["lb"]
            ub    = d["ub"]
            vtype = d.get("types", d.get("vtype"))
            obj   = d.get("obj", 0.0)

            n = len(names)
            if not isinstance(lb, list):    lb    = [lb]    * n
            if not isinstance(ub, list):    ub    = [ub]    * n
            if not isinstance(vtype, list): vtype = [vtype] * n
            if not isinstance(obj, list):   obj   = [obj]   * n

            scip_vars[group] = [
                model.addVar(name=nm, lb=float(lo), ub=float(hi), vtype=vt, obj=float(oc))
                for nm, lo, hi, vt, oc in zip(names, lb, ub, vtype, obj, strict=False)
            ]

        # add linking + absolute-value constraints
        for j in (actionable_indices):
            a_j = scip_vars["a"][j]
            a_pos_j = scip_vars["a_pos"][j]
            a_neg_j = scip_vars["a_neg"][j]
            c_j = scip_vars["c"][j]

            a_j_name = indices.names["a"][j]
            
            model.addCons(
                a_pos_j - a_j >= 0,
                name=f"abs_val_pos_{a_j_name}"
            )
            model.addCons(
                a_neg_j + a_j >= 0,
                name=f"abs_val_neg_{a_j_name}"
            )
            model.addCons(
                a_j == a_pos_j - a_neg_j,
                name=f"decomp_[{j}]"
            )

            con = model.addCons(
                c_j == a_j,
                name=f"set_c_[{j}]"
            )

            indices.add_constraint(f"set_c_[{j}]", con)
        
        model.hideOutput()

        terms = [scip_vars["a_pos"][j] + scip_vars["a_neg"][j]
            for j in actionable_indices]
        obj_expr = quicksum(terms)

        model.setObjective(obj_expr, "minimize")

        return model, indices

    def configure(self, model: Model, print_flag: bool) -> Model:
        from pyscipopt import SCIP_PARAMEMPHASIS

        model.setEmphasis(SCIP_PARAMEMPHASIS.NUMERICS)
        model.setParam("limits/gap", 0.0)
        model.setParam("limits/absgap", 0.0)
        model.setParam("numerics/feastol", 1e-7)
        if not print_flag:
            model.hideOutput()
        else:
            model.setParam("display/verblevel", 4)
        return model

    def add_constraints(self, model, indices, action_set: ActionSet, x: np.ndarray):
        for con in action_set.constraints:
            model, indices = con.add_to_scip(scip=model, indices=indices, x=x)
        return model, indices

    # Solve/inspect
    def solve(self, model: Model) -> None:
        model.optimize()

    def has_solution(self, model: Model) -> bool:
        # Use presence of a best solution rather than status, to be robust
        # return model.getBestSol() is not None
        return model.getNSols() > 0

    def read_vectors(self, model: Model, indices, names: List[str]) -> Dict[str, np.ndarray]:
        sol = model.getBestSol()
        if sol is None:
            return {nm: None for nm in names}
        out: Dict[str, np.ndarray] = {}
        for nm in names:
            group_names = indices.names[nm]
            vars_group = [indices.get_var(model, v) for v in group_names]
            vals = [model.getSolVal(sol, v) for v in vars_group]
            out[nm] = np.array(vals)
        return out

    def add_nogood(
        self,
        model: Model,
        indices: SCIPGroupedVariableIndices,
        actions: List[np.ndarray],
        actionable_indices: List[int],
        settings: MIPSettings,
    ) -> Tuple[object, object, int]:
        # This mirrors the logic in the original scip path inside mip.py.
        # In SCIP, new variables/constraints must be added on the original problem.
        # free the transform first (add_nogood always called after solving)
        model.freeTransform()

        a_ub = indices.ub["c"]
        a_lb = indices.lb["c"]
        a_types = indices.types["c"]

        n_points = len(actions)
        A_nogood = np.vstack(actions)
        D_pos = a_ub - A_nogood
        D_neg = A_nogood - a_lb

        start_index = indices.counts.get("nogood", 0)
        point_indices = range(start_index, start_index + n_points)

        # Build grouped var args for append_variables bookkeeping
        grouped = {
            "delta_pos": {"names": [], "obj": [], "ub": [], "lb": [], "types": []},
            "delta_neg": {"names": [], "obj": [], "ub": [], "lb": [], "types": []},
            "delta_sign": {"names": [], "obj": [], "ub": [], "lb": [], "types": []},
        }

        # Create variables and add constraints per point
        for _idx, (k, ak, Dp_k, Dn_k) in enumerate(
            zip(point_indices, actions, D_pos, D_neg, strict=False)
        ):
            # create variables for this point
            for j, ub_dp, ub_dn in zip(actionable_indices, Dp_k, Dn_k, strict=False):
                name_pos = f"delta[{j, k}]_pos"
                name_neg = f"delta[{j, k}]_neg"
                name_sgn = f"delta[{j, k}]_sign"

                # bookkeeping for indices
                grouped["delta_pos"]["names"].append(name_pos)
                grouped["delta_pos"]["obj"].append(0.0)
                grouped["delta_pos"]["ub"].append(float(ub_dp))
                grouped["delta_pos"]["lb"].append(0.0)
                grouped["delta_pos"]["types"].append(a_types[j])

                grouped["delta_neg"]["names"].append(name_neg)
                grouped["delta_neg"]["obj"].append(0.0)
                grouped["delta_neg"]["ub"].append(float(ub_dn))
                grouped["delta_neg"]["lb"].append(0.0)
                grouped["delta_neg"]["types"].append(a_types[j])

                grouped["delta_sign"]["names"].append(name_sgn)
                grouped["delta_sign"]["obj"].append(0.0)
                grouped["delta_sign"]["ub"].append(1.0)
                grouped["delta_sign"]["lb"].append(0.0)
                grouped["delta_sign"]["types"].append("B")

                v_pos = model.addVar(lb=0.0, ub=float(ub_dp), vtype=a_types[j], name=name_pos, obj=0.0)
                v_neg = model.addVar(lb=0.0, ub=float(ub_dn), vtype=a_types[j], name=name_neg, obj=0.0)
                v_sgn = model.addVar(lb=0.0, ub=1.0, vtype="B", name=name_sgn, obj=0.0)

            # sum(abs) >= eps_min
            terms = [
                indices.get_var(model, f"delta[{j, k}]_pos") for j in actionable_indices
            ] + [indices.get_var(model, f"delta[{j, k}]_neg") for j in actionable_indices]
            model.addCons(quicksum(terms) >= float(settings.eps_min), name=f"sum_abs_dist_val_{k}")

            # feature-wise linking
            for idx_j, j in enumerate(actionable_indices):
                pos = indices.get_var(model, f"delta[{j, k}]_pos")
                neg = indices.get_var(model, f"delta[{j, k}]_neg")
                sign = indices.get_var(model, f"delta[{j, k}]_sign")
                c_j = indices.get_var(model, f"c[{j}]")

                model.addCons(pos <= float(Dp_k[idx_j]) * sign, name=f"nogood_pos_if_{j}_{k}")
                model.addCons(neg + float(Dn_k[idx_j]) * sign <= float(Dn_k[idx_j]), name=f"nogood_neg_if_{j}_{k}")
                model.addCons(c_j - pos + neg == float(ak[idx_j]), name=f"nogood_dist_{j}_{k}")

        # now pass grouped dict for indices bookkeeping
        indices.append_variables(model, grouped)

        indices.counts["nogood"] = start_index + len(A_nogood)
        return model, indices, len(A_nogood)

    def stats(self, model: Model) -> Dict:
        return get_scip_stats(model)

    # Generic constraint operations
    def add_linear_constraint(
        self,
        model: Model,
        indices: SCIPGroupedVariableIndices,
        name: str,
        terms: List[tuple[str, int, float]],
        sense: str,
        rhs: float,
    ) -> None:
        try:
            model.freeTransform()
        except Exception:
            # If not transformed, ignore.
            pass
        # Build expression from terms
        expr = None
        for group, idx, coef in terms:
            varname = indices.names[group][idx]
            var = indices.get_var(model, varname)
            term = coef * var
            expr = term if expr is None else expr + term

        if sense == "E":
            con = model.addCons(expr == float(rhs), name=name)
        elif sense == "L":
            con = model.addCons(expr <= float(rhs), name=name)
        elif sense == "G":
            con = model.addCons(expr >= float(rhs), name=name)
        else:
            raise ValueError(f"unknown sense '{sense}'")

        # store for deletion
        indices.add_constraint(name, con)

    def delete_constraint(self, model: Model, indices: SCIPGroupedVariableIndices, name: str) -> None:
        try:
            model.freeTransform()
        except Exception:
            # If not transformed, ignore.
            pass
        con = indices.get_constraint(name)
        model.delCons(con)

    def solution_status(self, model: Model) -> str:
        return str(model.getStatus())
