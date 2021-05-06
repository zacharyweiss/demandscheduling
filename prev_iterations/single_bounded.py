#!/usr/bin/env python
"""
Single EV cohort, limited hours of (dis)charge, able to influence price

Presumes EV should be fully charged after final hour connected to the grid, in addition to all assumptions stated in
the earlier singe_unbounded case.
"""

__author__ = "Zachary Weiss"

import pyomo.environ as pyo
import numpy as np

# global settings
N_HOURS = 24
EV_CONFIG = {"S_0": 0, "S_max": 5, "R_max": 1, "H": range(4, 18), "P": 1}
# S -> stored energy
# R -> charge rate (per hour)
# H -> (array of) hours available to charge during
# P -> price influence coefficient, zero makes price independent of EV demand

# price / load signals
base_prices = np.random.rand(N_HOURS) * 10


def main():
    model = pyo.ConcreteModel()

    # check valid hour configuration (no online hours specified beyond N_HOURS)
    if max(EV_CONFIG["H"]) >= N_HOURS or min(EV_CONFIG["H"]) < 0:
        raise SystemExit("Hours specified for EV (dis)charge must be between zero and N_HOURS. Modify the EV config "
                         "and rerun.")

    # index
    hours = range(N_HOURS)

    # decision vars
    model.S = pyo.Var(range(N_HOURS + 1), bounds=(0, EV_CONFIG["S_max"]), within=pyo.NonNegativeReals)
    model.R = pyo.Var(hours, bounds=(-EV_CONFIG["R_max"], EV_CONFIG["R_max"]), within=pyo.Reals)
    model.P = pyo.Var(hours, within=pyo.NonNegativeReals)

    # objective function
    cost = sum(model.P[t] * model.R[t] for t in hours)
    model.cost = pyo.Objective(expr=cost, sense=pyo.minimize)

    # constraints
    model.cons = pyo.ConstraintList()
    # boundary condition: storage begins at initial value
    model.cons.add(model.S[0] == EV_CONFIG["S_0"])
    # boundary condition: after final hour, storage must equal maximum charge
    model.cons.add(model.S[N_HOURS] == EV_CONFIG["S_max"])

    # constraints applied each hour (bounds already handled in pyomo variable declaration)
    for t in hours:
        # if the EV is not able to (dis)charge during the current hour, the rate must be zero
        if t not in EV_CONFIG["H"]:
            model.cons.add(model.R[t] == 0)
        # update rule, storage at next time point is current storage plus amount charged
        model.cons.add(model.S[t + 1] == model.S[t] + model.R[t])
        # price at each hour is sum of base price and amount of price increase from the load scheduled
        model.cons.add(model.P[t] == base_prices[t] + (EV_CONFIG["P"] * model.R[t]))

    # cbc, glpk, gurobi, cplex, pico, scip, xpress: LP/MIP solvers
    # conopt, cyipopt, ipopt: NLP
    # path: MCP
    # more can be found via "pyomo help --solvers"
    results = pyo.SolverFactory('multistart').solve(model, suppress_unbounded_warning=True)

    # display results
    model.pprint()
    print("\n" + "#"*150 + "\n")
    results.write()


if __name__ == '__main__':
    main()
