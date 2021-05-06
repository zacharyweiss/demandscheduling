#!/usr/bin/env python
"""Single EV cohort, able to (dis)charge whenever during the day, able to influence price


Demand scheduling optimization for EV charging.
Possible setups, in increasing difficulty:
1. Easiest: one cohort, able to charge whenever, independent price
2. One cohort, able to charge whenever, demand impacts prices
3. One cohort, limited charging window, independent price
4. One cohort, limited charging window, demand impacts prices
5. Two cohorts, limited charging window, independent price
6. Two cohorts, limited charging window, demand impacts prices
7. caps to total system load?

All assume perfect knowledge ahead of time for demand / clearing price (excluding EV additional demand)
Robust model? Could create price or demand profiles w/ rand noise representing SD from predictions to see how stable
results are

Assumes we wish to charge all to maximum capacity in the N_HOURS window

Does not address possibility of selling back to grid
Uses overly simplistic price variability, can be made arbitrarily complex however
Assumes price of sale back to grid == clearing price
Assumes symmetric maximum charge and discharge rates
"""

__author__ = "Zachary Weiss"

import pyomo.environ as pyo
import numpy as np

# global settings
N_HOURS = 24
EV_CONFIG = {"S_0": 0, "S_max": 5, "R_max": 1, "P": 1}
# S -> stored energy
# R -> charge rate (per hour)
# P -> price influence coefficient, zero makes price independent of EV demand

# price / load signals
base_prices = np.random.rand(N_HOURS) * 10


def main():
    model = pyo.ConcreteModel()

    # index
    hours = range(N_HOURS)

    # decision vars
    model.S = pyo.Var(range(N_HOURS+1), bounds=(0, EV_CONFIG["S_max"]), within=pyo.NonNegativeReals)
    model.R = pyo.Var(hours, bounds=(-EV_CONFIG["R_max"], EV_CONFIG["R_max"]), within=pyo.Reals)
    model.P = pyo.Var(hours, within=pyo.NonNegativeReals)

    # objective function
    cost = sum(model.P[t] * model.R[t] for t in hours)
    model.cost = pyo.Objective(expr=cost, sense=pyo.minimize)

    # constraints
    model.cons = pyo.ConstraintList()
    # boundary condition: after final hour, storage must equal maximum charge
    model.cons.add(model.S[N_HOURS] == EV_CONFIG["S_max"])
    # boundary condition: storage begins at initial value
    model.cons.add(model.S[0] == EV_CONFIG["S_0"])

    # constraints applied each hour (bounds already handled in pyomo variable declaration)
    for t in hours:
        # update rule, storage at next time point is current storage plus amount charged
        model.cons.add(model.S[t+1] == model.S[t] + model.R[t])
        # price at each hour is sum of base price and amount of price increase from the load scheduled
        model.cons.add(model.P[t] == base_prices[t] + (EV_CONFIG["P"] * model.R[t]))

    # cbc, glpk, gurobi, cplex, pico, scip, xpress: LP/MIP solvers
    # conopt, cyipopt, ipopt: NLP
    # path: MCP
    # more can be found via "pyomo help --solvers"
    results = pyo.SolverFactory('multistart').solve(model, suppress_unbounded_warning=True)

    # display results
    model.display()
    print("done")


if __name__ == '__main__':
    main()
