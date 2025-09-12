"""CPLEX utilities: variable tracking, parameters, and basic helpers."""

import operator
from functools import reduce
from itertools import chain

import numpy as np
import pandas as pd
from cplex import Cplex, SparsePair
from cplex.exceptions import CplexError


def concat(d):
    """Concatenate dict-of-lists values into a single list (preserve order)."""
    return list(chain.from_iterable(d.values()))


def combine(a, b):
    """Combine two dicts of lists by key-wise concatenation."""
    return {key: a.get(key, []) + b.get(key, []) for key in (a.keys() | b.keys())}


# feasibility/solution checking
CPX_INFEASIBLE_STATUS_CODES = (103,)


def is_certifiably_infeasible(cpx):
    """Return True if the solution status indicates infeasibility."""
    out = cpx.solution.get_status() in CPX_INFEASIBLE_STATUS_CODES
    return out


def has_solution(cpx):
    """Return True if the model exposes a solution in `cpx.solution`."""
    out = False
    try:
        cpx.solution.get_values()
        out = True
    except CplexError:
        pass
    return out


# variable stamping
VTYPE_TO_CPXTYPE = {int: "I", bool: "I", float: "C"}


class CplexGroupedVariableIndices(object):
    """Track grouped variables and parameters for a CPLEX MIP object."""

    variable_fields = (
        "names",
        "obj",
        "ub",
        "lb",
        "types",
    )
    mip_fields = "params"

    def __init__(self):
        """Initialize tracking maps for variable groups and parameters."""
        # initialize variable fields
        for field in self.variable_fields:
            self.__setattr__(field, {})

        self.params = {}
        self.counts = {}
        return

    def append_variables(self, cpx_variable_args):
        """Append variable metadata from grouped `cpx_variable_args`.

        The `cpx_variable_args` format is a mapping from group name to a dict
        with keys `names`, `obj`, `ub`, `lb`, `types`.
        """
        for field in self.variable_fields:
            f = self.__getattribute__(field)
            for name, values in cpx_variable_args.items():
                if name in f:
                    f[name] = f[name] + list(values[field])
                else:
                    f.update({name: values[field]})
        assert self.__check_rep__()

    def append_parameters(self, parameters, overwrite=False):
        """Append or update stored parameter values."""
        if overwrite:
            self.params.update(parameters)
        else:
            for name, values in parameters.items():
                if name not in self.params:
                    self.params[name] = values
                elif isinstance(self.params[name], list):
                    self.params[name] += list(values)
                elif self.params[name] != values:
                    raise ValueError(f"appending new value for parameter {name}")

    def __check_rep__(self):
        """Check internal consistency across all variable group fields."""
        variable_group_names = self.names.keys()
        for field in self.variable_fields:
            field_dict = self.__getattribute__(field)
            assert variable_group_names == field_dict.keys()
            for k, v in field_dict.items():
                assert len(self.names[k]) == len(v)
        return True

    def check_cpx(self, cpx):
        """Check names, bounds, types, and objectives match those in `cpx`."""
        assert isinstance(cpx, Cplex)
        vars = cpx.variables
        assert self.__check_rep__()
        try:
            indexed_names = concat(self.names)
            assert set(indexed_names) == set(vars.get_names())
            assert concat(self.lb) == vars.get_lower_bounds(indexed_names)
            assert concat(self.ub) == vars.get_upper_bounds(indexed_names)
            assert concat(self.types) == vars.get_types(indexed_names)
            assert concat(self.obj) == cpx.objective.get_linear(indexed_names)
        except AssertionError as e:
            print(e)
            from dev.debug import ipsh

            ipsh()
        return True


def get_cpx_variable_types(action_set, indices=None):
    """Return a string of CPLEX variable-type codes for selected features."""
    if indices is None:
        indices = range(len(action_set))
    out = "".join(
        [VTYPE_TO_CPXTYPE[vt] for j, vt in enumerate(action_set.variable_type) if j in indices]
    )
    return out


def get_cpx_variable_args(name, obj, ub, lb, vtype):
    """Prepare grouped variable arguments to add to a CPLEX model.

    This normalizes scalar inputs to lists and validates lengths, returning a
    dict suitable for `cpx.variables.add(**variable_args)`.
    """
    # name
    if isinstance(name, np.ndarray):
        name = name.tolist()
    elif isinstance(name, str):
        name = [name]

    nvars = len(name)
    # convert inputs
    if nvars == 1:
        # convert to list
        name = name if isinstance(name, list) else [name]
        obj = [float(obj[0])] if isinstance(obj, list) else [float(obj)]
        ub = [float(ub[0])] if isinstance(ub, (list, np.ndarray)) else [float(ub)]
        lb = [float(lb[0])] if isinstance(lb, (list, np.ndarray)) else [float(lb)]
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


###### Generic Helper Functions #####


def get_mip_stats(cpx):
    """Return a dict of high-level MIP statistics and solution info."""
    info = {
        "status": "no solution exists",
        "status_code": float("nan"),
        "has_solution": False,
        "has_mipstats": False,
        "iterations": 0,
        "nodes_processed": 0,
        "nodes_remaining": 0,
        "values": float("nan"),
        "objval": float("nan"),
        "upperbound": float("nan"),
        "lowerbound": float("nan"),
        "gap": float("nan"),
    }

    try:
        sol = cpx.solution
        info.update(
            {
                "status": sol.get_status_string(),
                "status_code": sol.get_status(),
                "iterations": sol.progress.get_num_iterations(),
                "nodes_processed": sol.progress.get_num_nodes_processed(),
                "nodes_remaining": sol.progress.get_num_nodes_remaining(),
            }
        )
        info["has_mipstats"] = True
    except CplexError:
        pass

    try:
        sol = cpx.solution
        info.update(
            {
                "values": np.array(sol.get_values()),
                "objval": sol.get_objective_value(),
                "upperbound": sol.MIP.get_cutoff(),
                "lowerbound": sol.MIP.get_best_objective(),
                "gap": sol.MIP.get_mip_relative_gap(),
            }
        )
        info["has_solution"] = True
    except CplexError:
        pass

    return info


def copy_cplex(cpx):
    """Create a copy of the model and copy changed parameters."""
    cpx_copy = Cplex(cpx)
    cpx_parameters = cpx.parameters.get_changed()
    for pname, pvalue in cpx_parameters:
        phandle = reduce(getattr, str(pname).split("."), cpx_copy)
        phandle.set(pvalue)
    return cpx_copy


def get_lp_relaxation(cpx):
    """Return a copy of the model with integrality relaxed (LP)."""
    rlx = copy_cplex(cpx)
    if rlx.get_problem_type() is rlx.problem_type.MILP:
        rlx.set_problem_type(rlx.problem_type.LP)
    return rlx


def add_mip_start(cpx, solution, effort_level=1, name=None):
    """Add a MIP start vector to the model."""
    if isinstance(solution, np.ndarray):
        solution = solution.tolist()

    mip_start = SparsePair(val=solution, ind=list(range(len(solution))))
    if name is None:
        cpx.MIP_starts.add(mip_start, effort_level)
    else:
        cpx.MIP_starts.add(mip_start, effort_level, name)

    return cpx


# Parameter Manipulation
CPX_MIP_PARAMETERS = {
    "display_cplex_progress": True,  # set to True to show CPLEX progress in console
    "n_cores": 8,  # Number of CPU cores to use in B&B;
    "randomseed": 0,  # Set the random seed differently for diversity of solutions. https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RandomSeed.html
    "time_limit": 1e75,  # Runtime before stopping,
    "node_limit": 9223372036800000000,  # Number of nodes to process before stopping,
    #
    "mipgap": np.finfo("float").eps,
    # Relative tolerance on gap between incumbent and best remaining node.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpGap.html
    #
    "absmipgap": 0.9,  # np.finfo('float').eps,
    # Absolute tolerance on the gap to stop MIP search when satisfied.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpAGap.html
    #
    "objdifference": 0.9,
    # Cutoff update slack to ignore negligible improvements to incumbent.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ObjDif.html#
    #
    "integrality_tolerance": 0.0,
    # Integrality tolerance (0 treated as exact integer feasibility).
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpInt.html
    #
    "mipemphasis": 0,
    # Controls trade-offs between speed, feasibility, optimality, and moving bounds in MIP.
    # 0     =	Balance optimality and feasibility; default
    # 1	    =	Emphasize feasibility over optimality
    # 2	    =	Emphasize optimality over feasibility
    # 3 	=	Emphasize moving best bound
    # 4	    =	Emphasize finding hidden feasible solutions
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIPEmphasis.html
    #
    "bound_strengthening": -1,
    # Decides whether to apply bound strengthening in mixed integer programs (MIPs).
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/BndStrenInd.html
    # -1    = cplex chooses
    # 0     = no bound strengthening
    # 1     = bound strengthening
    #
    "cover_cuts": -1,
    # Decides whether or not cover cuts should be generated for the problem.
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/Covers.html
    # -1    = Do not generate cover cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate cover cuts moderately
    # 2	    = Generate cover cuts aggressively
    # 3     = Generate cover cuts very  aggressively
    #
    "zero_half_cuts": -1,
    # Toggle zero-half cuts (often not effective, disabled here).
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ZeroHalfCuts.html
    # -1    = Do not generate MIR cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate MIR cuts moderately
    # 2	    = Generate MIR cuts aggressively
    #
    "mir_cuts": -1,
    # Toggle mixed-integer rounding cuts (often not effective; disabled here)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIRCuts.html
    # -1    = Do not generate zero-half cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate zero-half cuts moderately
    # 2	    = Generate zero-half cuts aggressively
    #
    "implied_bound_cuts": 0,
    # Decides whether or not to generate valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    "locally_implied_bound_cuts": 3,
    # Decides whether or not to generate locally valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    "scale_parameters": 1,
    # Decides how to scale the problem matrix.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ScaInd.html
    # 0     = equilibration scaling
    # 1     = aggressive scaling
    # -1    = no scaling
    #
    "numerical_emphasis": 0,
    # Emphasizes precision in numerically unstable or difficult problems.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/NumericalEmphasis.html
    # 0     = off
    # 1     = on
    #
    "poolsize": 100,
    # Limits the number of solutions kept in the solution pool
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolCapacity.html
    # number of feasible solutions to keep in solution pool
    #
    "poolrelgap": float("nan"),
    # Sets a relative tolerance on the objective value for the solutions in the solution pool.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolGap.html
    #
    "poolreplace": 2,
    # Strategy for replacing solutions when the solution pool is full
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolReplace.html
    # 0	= Replace oldest (FIFO); default
    # 1	= Replace the solution which has the worst objective
    # 2	= Replace solutions in order to build a set of diverse solutions
    #
    "repairtries": 20,
    # Limits the attempts to repair an infeasible MIP start.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RepairTries.html
    # -1	None: do not try to repair
    #  0	Automatic: let CPLEX choose; default
    #  N	Number of attempts
    #
    "nodefilesize": (120 * 1024) / 1,
    # size of the node file (for large scale problems)
    # if the B & B can no longer fit in memory, then CPLEX stores the B & B in a node file
}


def set_cpx_display_options(cpx, display_mip=True, display_parameters=False, display_lp=False):
    """Toggle CPLEX console output for MIP, simplex, and parameters."""
    cpx.parameters.mip.display.set(display_mip)
    cpx.parameters.simplex.display.set(display_lp)
    cpx.parameters.paramdisplay.set(display_parameters)

    if not (display_mip or display_lp):
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_error_stream(None)
        cpx.set_warning_stream(None)

    return cpx


def set_mip_parameters(cpx, param=CPX_MIP_PARAMETERS):
    """Apply selected MIP parameters and display options to the model."""
    # get parameter handle
    p = cpx.parameters

    # Record calls to C API
    # cpx.parameters.record.set(True)

    if param["display_cplex_progress"] is (None or False):
        cpx = set_cpx_display_options(
            cpx, display_mip=False, display_lp=False, display_parameters=False
        )

    # major parameters
    p.randomseed.set(param["randomseed"])
    p.output.clonelog.set(0)

    # solution strategy
    p.emphasis.mip.set(param["mipemphasis"])
    p.preprocessing.boundstrength.set(param["bound_strengthening"])

    # cuts
    p.mip.cuts.implied.set(param["implied_bound_cuts"])
    p.mip.cuts.localimplied.set(param["locally_implied_bound_cuts"])
    p.mip.cuts.zerohalfcut.set(param["zero_half_cuts"])
    p.mip.cuts.mircut.set(param["mir_cuts"])
    p.mip.cuts.covers.set(param["cover_cuts"])
    #
    # tolerances
    p.emphasis.numerical.set(param["numerical_emphasis"])
    p.mip.tolerances.integrality.set(param["integrality_tolerance"])

    # initialization
    p.mip.limits.repairtries.set(param["repairtries"])

    # solution pool
    p.mip.pool.capacity.set(param["poolsize"])
    p.mip.pool.replace.set(param["poolreplace"])
    #
    # p.preprocessing.aggregator.set(0)
    # p.preprocessing.reduce.set(0)
    # p.preprocessing.presolve.set(0)
    # p.preprocessing.coeffreduce.set(0)
    # p.preprocessing.boundstrength.set(0)

    # stopping
    p.mip.tolerances.mipgap.set(param["mipgap"])
    p.mip.tolerances.absmipgap.set(param["absmipgap"])

    if param["time_limit"] < CPX_MIP_PARAMETERS["time_limit"]:
        cpx = set_mip_time_limit(cpx, param["time_limit"])

    if param["node_limit"] < CPX_MIP_PARAMETERS["node_limit"]:
        cpx = set_mip_node_limit(cpx, param["node_limit"])

    # node file
    # p.workdir.Cur  = exp_workdir;
    # p.workmem.Cur                    = cplex_workingmem;
    # p.old_tests.strategy.file.Cur          = 2; %nodefile uncompressed
    # p.old_tests.limits.treememory.Cur      = cplex_nodefilesize;

    return cpx


def get_mip_parameters(cpx):
    """Read back a subset of key MIP parameters from the model."""
    p = cpx.parameters

    param = {
        # major
        "display_cplex_progress": p.mip.display.get() > 0,
        "randomseed": p.randomseed.get(),
        "n_cores": p.threads.get(),
        #
        # strategy
        "mipemphasis": p.emphasis.mip.get(),
        "scale_parameters": p.read.scale.get(),
        "locally_implied_bound_cuts": p.mip.cuts.localimplied.get(),
        #
        # stopping
        "time_limit": p.timelimit.get(),
        "node_limit": p.mip.limits.nodes.get(),
        "mipgap": p.mip.tolerances.mipgap.get(),
        "absmipgap": p.mip.tolerances.absmipgap.get(),
        #
        # old_tests tolerances
        "integrality_tolerance": p.mip.tolerances.integrality.get(),
        "numerical_emphasis": p.emphasis.numerical.get(),
        #
        # solution pool
        "repairtries": p.mip.limits.repairtries.get(),
        "poolsize": p.mip.pool.capacity.get(),
        "poolreplace": p.mip.pool.replace.get(),
        #
        # node file
        # old_tests.parameters.workdir.Cur  = exp_workdir;
        # old_tests.parameters.workmem.Cur                    = cplex_workingmem;
        # old_tests.parameters.old_tests.strategy.file.Cur          = 2; %nodefile uncompressed
        # old_tests.parameters.old_tests.limits.treememory.Cur      = cplex_nodefilesize;
    }

    return param


def toggle_mip_preprocessing(cpx, toggle=True):
    """Toggles pre-processing on/off for debugging / computational experiments."""
    # presolve
    # old_tests.parameters.preprocessing.presolve.help()
    # 0 = off
    # 1 = on

    # boundstrength
    # type of bound strengthening  :
    # -1 = automatic
    # 0 = off
    # 1 = on

    # reduce
    # old_tests.parameters.preprocessing.reduce.help()
    # type of primal and dual reductions  :
    # 0 = no primal and dual reductions
    # 1 = only primal reductions
    # 2 = only dual reductions
    # 3 = both primal and dual reductions

    # coeffreduce strength
    # level of coefficient reduction  :
    #   -1 = automatic
    #   0 = none
    #   1 = reduce only to integral coefficients
    #   2 = reduce any potential coefficient
    #   3 = aggressive reduction with tilting

    # dependency
    # indicator for preprocessing dependency checker  :
    #   -1 = automatic
    #   0 = off
    #   1 = at beginning
    #   2 = at end
    #   3 = at both beginning and end

    if toggle:
        cpx.parameters.preprocessing.aggregator.reset()
        cpx.parameters.preprocessing.reduce.reset()
        cpx.parameters.preprocessing.presolve.reset()
        cpx.parameters.preprocessing.coeffreduce.reset()
        cpx.parameters.preprocessing.boundstrength.reset()
    else:
        cpx.parameters.preprocessing.aggregator.set(0)
        cpx.parameters.preprocessing.reduce.set(0)
        cpx.parameters.preprocessing.presolve.set(0)
        cpx.parameters.preprocessing.coeffreduce.set(0)
        cpx.parameters.preprocessing.boundstrength.set(0)

    return cpx


def set_mip_cutoff_values(cpx, objval, objval_increment):
    """Set upper cutoff and absolute tolerances based on incumbent progress."""
    assert objval >= 0.0
    assert objval_increment >= 0.0
    p = cpx.parameters
    p.mip.tolerances.uppercutoff.set(float(objval))
    p.mip.tolerances.objdifference.set(0.95 * float(objval_increment))
    p.mip.tolerances.absmipgap.set(0.95 * float(objval_increment))
    return cpx


# Stopping Conditions
def set_mip_max_gap(cpx, max_gap=None):
    """Set the largest value of the relative optimality gap to stop solving a MIP."""
    if max_gap is not None:
        max_gap = float(max_gap)
        max_gap = min(max_gap, cpx.parameters.mip.tolerances.mipgap.max())
    else:
        max_gap = cpx.parameters.mip.tolerances.mipgap.min()

    assert max_gap >= 0.0
    cpx.parameters.mip.tolerances.mipgap.set(max_gap)

    return cpx


def set_mip_time_limit(cpx, time_limit=None):
    """Set the time limit (seconds) for MIP solving."""
    max_time_limit = float(cpx.parameters.timelimit.max())

    if time_limit is None:
        time_limit = max_time_limit
    else:
        time_limit = float(time_limit)
        time_limit = min(time_limit, max_time_limit)

    assert time_limit >= 0.0
    cpx.parameters.timelimit.set(time_limit)
    return cpx


def set_mip_node_limit(cpx, node_limit=None):
    """Set the node limit for MIP branch-and-bound."""
    max_node_limit = cpx.parameters.mip.limits.nodes.max()
    if node_limit is not None:
        node_limit = int(node_limit)
        node_limit = min(node_limit, max_node_limit)
    else:
        node_limit = max_node_limit

    assert node_limit >= 0.0
    cpx.parameters.mip.limits.nodes.set(node_limit)
    return cpx


# Debugging
def solution_df(cpx, names=None):
    """Create a DataFrame with the current solution for a CPLEX object."""
    assert isinstance(cpx, Cplex)
    if names is None:
        names = cpx.variables.get_names()
    else:
        assert isinstance(names, dict)
        names = reduce(operator.concat, names.values())

    if has_solution(cpx):
        all_values = cpx.solution.get_values(names)
    else:
        all_values = np.repeat(np.nan, len(names)).tolist()

    df = pd.DataFrame(
        {
            "name": names,
            "value": all_values,
            "lb": cpx.variables.get_upper_bounds(names),
            "ub": cpx.variables.get_lower_bounds(names),
        }
    )

    return df
