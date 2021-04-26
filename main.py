import pyomo.environ as pyomo
import numpy as np


# global settings
N_HOURS = 24


def main():
    model = pyomo.ConcreteModel()

    hours = range(N_HOURS)
    prices = np.random.rand(N_HOURS)*10
    loads = np.random.rand(N_HOURS)*2




if __name__ == '__main__':
    main()
