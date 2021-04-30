#!/usr/bin/env python
"""Demand scheduling optimization for EV charging.
Possible setups, in increasing difficulty:
1. Easiest: one cohort, able to charge whenever, independent price
2. One cohort, able to charge whenever, demand impacts prices
3. One cohort, limited charging window, independent price
4. One cohort, limited charging window, demand impacts prices
5. Two cohorts, limited charging window, independent price
6. Two cohorts, limited charging window, demand impacts prices

All assume perfect knowledge ahead of time for demand / clearing price (excluding EV additional demand)
Robust model? Could create price or demand profiles w/ rand noise representing SD from predictions to see how stable
results are

Assumes we wish to charge all to maximum capacity in the N_HOURS window
"""

__author__ = "Zachary Weiss"

import pyomo.environ as pyomo
import numpy as np

# global settings
N_HOURS = 24

# S -> stored energy
# R -> charge rate
# H -> (array of) hours available to charge during
# P -> price influence coefficient, zero makes price independent of demand
EV_CONFIG = [{"S": 0, "S_max": 10, "R_max": 1, "H": range(N_HOURS), "P": 0},
             ]


def main():
    model = pyomo.ConcreteModel()

    #
    hours = range(N_HOURS)
    prices = np.random.rand(N_HOURS)*10
    loads = np.random.rand(N_HOURS)*2

    for t in hours:




if __name__ == '__main__':
    main()
