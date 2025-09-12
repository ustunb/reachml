"""SCIP utility helpers for variable tracking, parameters, and stats."""

import operator
from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd

from pyscipopt import Model


def concat(d):
    """Concatenate dict-of-lists values into a single list (preserves order)."""
    return list(chain.from_iterable(d.values()))


def combine(a, b):
    """Combine two dicts of lists by key-wise concatenation."""
    return {key: a.get(key, []) + b.get(key, []) for key in (a.keys() | b.keys())}


# feasibility/solution checking
def is_certifiably_infeasible(model: Model) -> bool:
    """Checks if the last solve on the Model object proved infeasibility."""
    return model.getStatus() == "infeasible"


def has_solution(model: Model) -> bool:
    """Returns true if the model has found at least one feasible solution."""
    return model.getNSols() > 0


# variable stamping
VTYPE_TO_SCIP = {int: "I", bool: "B", float: "C"}


class SCIPGroupedVariableIndices:
    """Stores and tracks groups of variables created in a PySCIPOpt Model."""

    variable_fields = ("names", "obj", "ub", "lb", "types")

    constraints = {}

    def __init__(self, model: Model = None, cpx_indices=None):
        """Initialize tracking; optionally import from CPLEX-grouped indices."""
        # initialize variable fields
        for field in self.variable_fields:
            setattr(self, field, {})
        self.params = {}
        self.counts = {}

        # if provided, convert from CPLEX indices
        if model is not None and cpx_indices is not None:
            from .cplex_utils import CplexGroupedVariableIndices

            # Convert from full CPLEX tracker
            if isinstance(cpx_indices, CplexGroupedVariableIndices):
                type_map = {
                    "C": "CONTINUOUS",
                    "I": "INTEGER",
                    "B": "BINARY",
                    "BINARY": "B",
                    "CONTINUOUS": "C",
                    "INTEGER": "I",
                }
                var_args = {}
                for group in cpx_indices.names:
                    var_args[group] = {
                        "names": list(cpx_indices.names[group]),
                        "obj": list(cpx_indices.obj[group]),
                        "lb": list(cpx_indices.lb[group]),
                        "ub": list(cpx_indices.ub[group]),
                        "types": [type_map[t] for t in cpx_indices.types[group]],
                    }
                self.append_variables(model, var_args)
                self.append_parameters(cpx_indices.params, overwrite=True)
                self.counts.update(getattr(cpx_indices, "counts", {}))

            # Or build from a bare var_args or raw var-object mapping
            elif isinstance(cpx_indices, dict):
                norm = {}
                for group, d in cpx_indices.items():
                    names = d.get("names")
                    lbs = d.get("lb")
                    ubs = d.get("ub")
                    vtype = d.get("types")
                    obj = d.get("obj", [0.0] * len(names))

                    # make sure these are lists
                    if not isinstance(lbs, list):
                        lbs = [lbs] * len(names)
                    if not isinstance(ubs, list):
                        ubs = [ubs] * len(names)
                    if not isinstance(vtype, list):
                        vtype = [vtype] * len(names)

                    norm[group] = {
                        "names": names,
                        "lb": lbs,
                        "ub": ubs,
                        "types": vtype,
                        "obj": obj,
                    }

                var_args = norm

            else:
                # assume already in var_args format
                var_args = cpx_indices

            # create SCIP vars and record metadata
            self.append_variables(model, var_args)
        else:
            raise TypeError("cpx_indices must be a CplexGroupedVariableIndices or a var_args dict")

    def get_var(self, model: Model, name: str):
        """Lookup a variable by name on the given model."""
        for var in model.getVars():
            # var.name or var.getName()
            if getattr(var, "name", None) == name or (
                hasattr(var, "getName") and var.getName() == name
            ):
                return var
        raise KeyError(f"Variable '{name}' not found in model")

    def append_variables(self, model: Model, var_args: dict):
        """Append variable metadata from grouped `var_args`.

        The `var_args` format matches the CPLEX helper: a dict mapping group
        name to a dict with keys `names`, `obj`, `ub`, `lb`, `types`.
        """
        for field in self.variable_fields:
            f = self.__getattribute__(field)
            for name, values in var_args.items():
                if name in f:
                    f[name] = f[name] + list(values[field])
                else:
                    f.update({name: values[field]})

        assert self.__check_rep__()

    def append_parameters(self, parameters: dict, overwrite: bool = True):
        """Append or overwrite solver parameters metadata.

        Always stores each value as a list and returns self so you can chain.
        """
        if overwrite:
            # replace entirely, casting each value to a list
            self.params = {k: list(v) for k, v in parameters.items()}
        else:
            for k, v in parameters.items():
                v_list = list(v) if isinstance(v, (list, tuple)) else [v]
                if k not in self.params:
                    # first time — store as a list
                    self.params[k] = v_list
                else:
                    # already present — extend
                    if not isinstance(self.params[k], list):
                        self.params[k] = [self.params[k]]
                    self.params[k].extend(v_list)
        return self

    def __check_rep__(self):
        """Check internal representation consistency across groups and fields."""
        groups = set(self.names.keys())
        for field in self.variable_fields:
            field_dict = getattr(self, field)
            assert groups == set(field_dict.keys())
            for g, lst in field_dict.items():
                assert len(lst) == len(self.names[g])
        return True

    def add_constraint(self, name: str, constraint):
        """Store a constraint by name."""
        self.constraints[name] = constraint

    def get_constraint(self, name: str):
        """Retrieve a stored constraint by name."""
        if name not in self.constraints:
            raise KeyError(f"Constraint '{name}' not found.")
        return self.constraints[name]

    def replace_constraint(self, name: str, constraint):
        """Replace an existing constraint by name."""
        if name not in self.constraints:
            raise KeyError(f"Constraint '{name}' not found.")
        self.constraints[name] = constraint


def get_scip_variable_types(action_set, indices=None):
    """Mirrors get_cpx_variable_types: returns a string of SCIP var-type codes."""
    if indices is None:
        indices = range(len(action_set.variable_type))
    return "".join(
        VTYPE_TO_SCIP[vt][0]  # first letter: 'I', 'B', or 'C'
        for j, vt in enumerate(action_set.variable_type)
        if j in indices
    )


def get_scip_variable_args(name, obj, ub, lb, vtype):
    """Prepare grouped variable arguments for use with PySCIPOpt.

    Same interface as `get_cpx_variable_args`.
    """
    # normalize to lists
    if isinstance(name, str):
        name = [name]
    elif isinstance(name, np.ndarray):
        name = name.tolist()

    nvars = len(name)
    # convert inputs
    if nvars == 1:
        # convert to list
        name = name if isinstance(name, list) else [name]
        obj = [float(obj[0])] if isinstance(obj, list) else [float(obj)]
        ub = [float(ub[0])] if isinstance(ub, list) else [float(ub)]
        lb = [float(lb[0])] if isinstance(lb, list) else [float(lb)]
        vtype = vtype if isinstance(vtype, list) else [vtype]
    else:
        # convert to list
        if isinstance(vtype, np.ndarray):
            vtype = vtype.tolist()
        elif isinstance(vtype, str):
            if len(vtype) == 1:
                vtype = nvars * [vtype]
            elif len(vtype) == nvars:
                vtype = list(vtype)
            else:
                raise ValueError(
                    "invalid length: len(vtype) = %d. expected either 1 or %d" % (len(vtype), nvars)
                )

        if isinstance(obj, np.ndarray):
            obj = obj.astype(float).tolist()
        elif isinstance(obj, list):
            if len(obj) == nvars:
                obj = [float(v) for v in obj]
            elif len(obj) == 1:
                obj = nvars * [float(obj)]
            else:
                raise ValueError(
                    f"invalid length: len(obj) = {len(obj)}. expected either 1 or {nvars}"
                )
        else:
            obj = nvars * [float(obj)]

        if isinstance(ub, np.ndarray):
            ub = ub.astype(float).tolist()
        elif isinstance(ub, list):
            if len(ub) == nvars:
                ub = [float(v) for v in ub]
            elif len(ub) == 1:
                ub = nvars * [float(ub)]
            else:
                raise ValueError(
                    f"invalid length: len(ub) = {len(ub)}. expected either 1 or {nvars}"
                )
        else:
            ub = nvars * [float(ub)]

        if isinstance(lb, np.ndarray):
            lb = lb.astype(float).tolist()
        elif isinstance(lb, list):
            if len(lb) == nvars:
                lb = [float(v) for v in lb]
            elif len(ub) == 1:
                lb = nvars * [float(lb)]
            else:
                raise ValueError(
                    f"invalid length: len(lb) = {len(lb)}. expected either 1 or {nvars}"
                )
        else:
            lb = nvars * [float(lb)]

    # check that all components are lists
    assert isinstance(name, list)
    assert isinstance(obj, list)
    assert isinstance(ub, list)
    assert isinstance(lb, list)
    assert isinstance(vtype, list)

    # check components
    for n in range(nvars):
        assert isinstance(name[n], str)
        assert isinstance(obj[n], float)
        assert isinstance(ub[n], float)
        assert isinstance(lb[n], float)
        assert isinstance(vtype[n], str)

    out = {
        "names": name,
        "obj": obj,
        "ub": ub,
        "lb": lb,
        "types": vtype,
    }
    return out


def get_scip_stats(model: Model) -> dict:
    """Return basic stats for the latest solve.

    Includes status, node and LP counts, bounds, gap, and if a solution exists.
    """
    info = {
        "status": model.getStatus(),
        "has_solution": model.getNSols() > 0,
        "nodes_processed": model.getNNodes(),
        "lp_iterations": model.getNLPIterations(),
        "primalbound": model.getObjVal() if model.getNSols() > 0 else np.nan,
        "dualbound": model.getDualbound(),
        "gap": model.getGap(),
    }
    return info


def copy_model(model: Model) -> Model:
    """Creates a deep copy of the model by writing and re‐reading to disk."""
    tmp = "_scip_copy.lp"
    model.writeProblem(tmp)
    new_model = Model()
    new_model.readProblem(tmp)
    return new_model


def get_lp_relaxation(model: Model) -> Model:
    """Returns a new Model where all integer/binary vars are relaxed to continuous."""
    rlx = copy_model(model)
    for var in rlx.getVars():
        if rlx.getVarType(var) != "CONTINUOUS":
            rlx.chgVarType(var, "CONTINUOUS")
    return rlx


def add_mip_start(model: Model, solution, name=None):
    """Adds a user‐provided start solution."""
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()
    sols = model.createSol()
    for var, val in zip(model.getVars(), solution, strict=False):
        model.setSolVal(sols, var, val)
    model.addSol(sols)
    return model


# SCIP analogue to CPLEX parameter set
SCIP_MIP_PARAMETERS = {
    "randomseed": 0,  # SCIP: 'randomization/permseed'
    "time_limit": 1e75,  # 'limits/time'
    "node_limit": 2**63 - 1,  # 'limits/nodes'
    "mipgap": 0.0,  # 'limits/gap'
    "absgap": 0.9,  # 'limits/absgap'
}


def set_scip_display_options(model: Model, display: bool = True):
    """Toggle console output."""
    if display:
        model.unhideOutput()
    else:
        model.hideOutput()
    return model


def set_mip_parameters(model: Model, params: dict = SCIP_MIP_PARAMETERS):
    """Map our standard dict into SCIP's parameter names."""
    model.setIntParam("randomization/permseed", int(params["randomseed"]))
    model.setRealParam("limits/time", float(params["time_limit"]))
    model.setIntParam("limits/nodes", int(params["node_limit"]))
    model.setRealParam("limits/gap", float(params["mipgap"]))
    model.setRealParam("limits/absgap", float(params["absgap"]))
    model.setParam("numerics/feastol", 1e-9)
    model.setParam("numerics/epsilon", 1e-12)
    model.setParam("numerics/sumepsilon", 1e-12)
    return model


def get_mip_parameters(model: Model) -> dict:
    """Read back key MIP parameters from SCIP."""
    return {
        "randomseed": model.getIntParam("randomization/permseed"),
        "time_limit": model.getRealParam("limits/time"),
        "node_limit": model.getIntParam("limits/nodes"),
        "mipgap": model.getRealParam("limits/gap"),
        "absgap": model.getRealParam("limits/absgap"),
    }


def toggle_mip_preprocessing(model: Model, toggle: bool = True):
    """Enable/disable SCIP presolve."""
    if toggle:
        model.setIntParam("presolving/maxrounds", 10)
    else:
        model.setIntParam("presolving/maxrounds", 0)
    return model


def set_mip_cutoff_values(model: Model, objval: float, objval_increment: float):
    """Set SCIP's upper cutoff and gap tolerances."""
    assert objval >= 0.0 and objval_increment >= 0.0
    model.setRealParam("limits/uppercutoff", float(objval))
    # shrink absolute gap
    model.setRealParam("limits/absgap", 0.95 * float(objval_increment))
    return model


def set_mip_max_gap_scip(model: Model, max_gap: float = None):
    """Set relative MIP gap tolerance."""
    if max_gap is not None:
        model.setRealParam("limits/gap", float(max_gap))
    else:
        # reset to default
        model.resetParam("limits/gap")
    return model


def set_mip_time_limit(model: Model, time_limit: float = None):
    """Set time limit (in seconds)."""
    if time_limit is not None:
        model.setRealParam("limits/time", float(time_limit))
    else:
        model.resetParam("limits/time")
    return model


def set_mip_node_limit(model: Model, node_limit: int = None):
    """Set node limit."""
    if node_limit is not None:
        model.setIntParam("limits/nodes", int(node_limit))
    else:
        model.resetParam("limits/nodes")
    return model


def solution_df(model: Model, names=None) -> pd.DataFrame:
    """Create a DataFrame with the current best solution."""
    vars_list = model.getVars()
    if names is None:
        names = [v.name for v in vars_list]
        vars_map = {v.name: v for v in vars_list}
    else:
        # names can be dict of groups
        names = reduce(operator.concat, names.values())
        vars_map = {v.name: v for v in vars_list}
    if has_solution(model):
        sol = model.getBestSol()
        all_vals = [model.getSolVal(sol, vars_map[n]) for n in names]
    else:
        all_vals = [np.nan] * len(names)

    df = pd.DataFrame(
        {
            "name": names,
            "value": all_vals,
            "lb": [vars_map[n].lb for n in names],
            "ub": [vars_map[n].ub for n in names],
        }
    )
    return df
