#!/usr/bin/env python
"""
Multiple EV cohorts, limited hours of (dis)charge, able to influence price

Same assumptions as in single_bounded. Doesn't account for modular math if running overnight (i.e. 6pm to 8am is parsed
as 12am to 8am followed by 6pm to 12am). Can be abated if one chooses to study a 2-day (or longer) period by increasing
N_HOURS.
"""

__author__ = "Zachary Weiss"

import pyomo.environ as pyo
import numpy as np

# global settings
N_HOURS = 24
EV_CONFIG = [{"S_0": 0, "S_max": 5, "R_max": 1, "H": range(7), "P": 0.5},
             {"S_0": 2, "S_max": 10, "R_max": 2, "H": range(5, 10), "P": 0.5},
             {"S_0": 5, "S_max": 20, "R_max": 3, "H": range(4, N_HOURS), "P": 0.5},
             ]


# S -> stored energy [kWh]
# R -> charge rate [kW] (as everywhere referenced the rate is applied over an hour--implied "*1hr" after each instance--
#      it is effectively in units of kWh)
# H -> (array of) hours available to charge during [unitless indexes]
# P -> price influence coefficient, zero makes price independent of EV demand [$/kWh^2]


def main():
    model = pyo.ConcreteModel()

    # price signal: array of prices at each hour [$/kWh], peak value at 6pm
    # base_prices = np.random.rand(N_HOURS) * 10
    base_prices = 5 * gaussian(np.linspace(0, N_HOURS - 1, N_HOURS), 17, 26) + 5

    # check valid hour configuration (no online hours specified beyond N_HOURS)
    for ev in EV_CONFIG:
        if max(ev["H"]) >= N_HOURS or min(ev["H"]) < 0:
            raise SystemExit("Hours specified for EV (dis)charge must be between zero and N_HOURS. Modify the EV "
                             "config and rerun.")

    # index
    hours = range(N_HOURS)
    model.i = pyo.Set(initialize=[i for i, ev in enumerate(EV_CONFIG)])
    model.t = pyo.Set(initialize=hours)
    model.t_1 = pyo.Set(initialize=range(N_HOURS + 1))

    def ij_init(m):
        # key pairs for S and R matrices w/i pyomo
        # i is the EV number, j is the hour (t)
        return ((i, j) for i in m.i for j in m.t)

    # same as above, but initializes for one extra hour to be compatible with the update rule
    def ij_init_1(m):
        return ((i, j) for i in m.i for j in m.t_1)

    # decision vars
    model.S = pyo.Var(pyo.Set(dimen=2, initialize=ij_init_1), within=pyo.NonNegativeReals)
    model.R = pyo.Var(pyo.Set(dimen=2, initialize=ij_init), within=pyo.Reals)
    model.P = pyo.Var(hours, within=pyo.NonNegativeReals)

    # objective function
    cost = sum(sum(model.P[t] * model.R[i, t] for t in hours) for i, ev in enumerate(EV_CONFIG))
    model.cost = pyo.Objective(expr=cost, sense=pyo.minimize)

    # constraints
    model.cons = pyo.ConstraintList()
    for i, ev in enumerate(EV_CONFIG):
        # boundary condition: storage begins at initial value
        model.cons.add(model.S[i, 0] == ev["S_0"])
        # boundary condition: after final schedule-able hour, storage must equal maximum charge
        model.cons.add(model.S[i, max(ev["H"]) + 1] == ev["S_max"])

    # constraints applied each hour (bounds already handled in pyomo variable declaration)
    for t in hours:
        for i, ev in enumerate(EV_CONFIG):
            # if the EV is not able to (dis)charge during the current hour, the rate must be zero. Else, bounded by max
            # and min (now added as constraint as cannot be easily added in variable bounds at time of declaration)
            if t not in ev["H"]:
                model.cons.add(model.R[i, t] == 0)
            else:
                model.cons.add(pyo.inequality(-ev["R_max"], model.R[i, t], ev["R_max"]))
            # stored energy must be between 0 and the maximum for each EV cohort
            model.cons.add(pyo.inequality(0, model.S[i, t], ev["S_max"]))
            # update rule, storage at next time point is current storage plus amount charged for each EV cohort
            model.cons.add(model.S[i, t + 1] == model.S[i, t] + model.R[i, t])
        # price at each hour is sum of base price and amount of price increase from the load scheduled
        model.cons.add(model.P[t] == base_prices[t] + sum([ev["P"] * model.R[i, t] for i, ev in enumerate(EV_CONFIG)]))

    # cbc, glpk, gurobi, cplex, pico, scip, xpress: LP/MIP solvers
    # conopt, cyipopt, ipopt: NLP
    # path: MCP
    # more can be found via "pyomo help --solvers"
    results = pyo.SolverFactory('multistart').solve(model, suppress_unbounded_warning=True)

    readout(model, base_prices)


def lrange(*args):
    """drop in replacement for 'range()', if one wishes to easily concatenate ranges with '+' in the 'EV_CONFIG'"""
    return list(range(*args))


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def readout(model, base_prices):
    S_sol = np.array([v.value for i, v in model.S.items()]).reshape(len(EV_CONFIG), N_HOURS + 1)
    R_sol = np.array([v.value for i, v in model.R.items()]).reshape(len(EV_CONFIG), N_HOURS)
    P_sol = np.array([v.value for i, v in model.P.items()])

    costs = np.array([np.array(ev_R * P_sol) for ev_R in R_sol])
    ev_tot_cost = costs.sum(axis=1)
    ev_avg_price = ev_tot_cost / R_sol.sum(axis=1)

    def pretty_print(arr):
        # zero out floating point errors within tolerance
        tol = 1e-16
        arr[(arr < tol) & (-tol < arr)] = 0

        def trimmer(n):
            return np.format_float_positional(n, 2, trim="-")

        for i, row in enumerate(arr):
            num_str = ''.join(
                f'\033[92m{trimmer(item): >7s}\033[0m' if t in EV_CONFIG[i]["H"] else
                f'{trimmer(item): >7s}' for t, item in enumerate(row))
            print("\033[3m{:<6s}\033[0m{}".format(f"EV#{i}", num_str))

    print("""\n\033[1mKey\033[0m
\033[3mS\033[0m -> stored energy [kWh]
\033[3mR\033[0m -> charge rate [kW] (as everywhere referenced the rate is applied over an hour--implied "*1hr" after each instance--
     it is effectively in units of kWh)
\033[3mH\033[0m -> (array of) hours available to charge during [unitless indexes]
\033[3mP\033[0m -> price influence coefficient, zero makes price independent of EV demand [$/kWh^2]
Values highlighted in green indicate the EV was available for (dis)charge during that hour.""")

    print("\n\033[1mEV Cohort Configuration\033[0m")
    for i, ev in enumerate(EV_CONFIG):
        print("\033[3m{:<12s}\033[0m".format(f"EV#{i}") + "".join("{:<25s}".format(f"\033[3m{k}\033[0m: {ev[k]}") for k in ev))

    print("\n\033[1mStorage values by hour [kWh]\033[0m")
    hour_arr = np.append(np.add(lrange(N_HOURS), 1), "S_final")
    print("\033[3m{:<9s}".format("Hour") + ''.join("{:^7s}".format(hr) for hr in hour_arr) + "\033[0m")
    pretty_print(S_sol)

    print("\n\033[1mCharging rates by hour [kW]\033[0m")
    pretty_print(R_sol)

    print("\n\033[1mBase and adjusted clearing price (after additional EV load) by hour [$/kWh]\033[0m")
    print("\033[3m{:<6s}\033[0m".format("Π") + "".join(f"{item: >7.2f}" for item in base_prices))
    print("\033[3m{:<6s}\033[0m".format("Π_adj") + "".join(f"{item: >7.2f}" for item in P_sol))

    print("\n\033[1mTotal cost by EV cohort\033[0m")
    for i, ev in enumerate(EV_CONFIG):
        print("\033[3m{:<8s}\033[0m{:>8s}    {}".format(f"EV#{i}", f"${ev_tot_cost[i]:.2f}", "(average "
              f"of ${ev_avg_price[i]:.2f} per unit of energy)"))
    print(f"\033[3mOverall\033[0m  ${model.cost():.2f}    (average of ${model.cost() / sum(sum(R_sol)):.2f} per unit "
          f"of energy)")

    print("\n")


"""
def df_from_pyo(model_var):
    # Extremely messy way to extract 2D variable values from pyomo back into pandas
    # poor documentation in pyomo, and solutions online don't generalize well to higher-dim vars.
    # This works passably, and allows for normal tuple indexing, but the df shape is still wrong if inspected
    data = {(i, j): v.value for (i, j), v in model_var.items()}
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index = pd.MultiIndex.from_tuples([k for k, v in df.iterrows()])
    return df[0]
"""

if __name__ == '__main__':
    main()
