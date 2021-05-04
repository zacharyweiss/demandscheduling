# demandscheduling
Solving optimization problems relating to demand scheduling on a power grid. The primary file is multi_bounded.py and can be run directly from the terminal, others 
are primitive versions with lesser functionality.

Dependencies: pyomo, numpy, and the multistart NLP solver (may come preinstalled with pyomo)

Key:

S -> stored energy [kWh]

R -> charge rate [kW] (as everywhere referenced the rate is applied over an hour—implied "*1hr" after each instance—it is effectively in units of kWh)

H -> (array of) hours available to charge during [unitless indexes]

P -> price influence coefficient, zero makes price independent of EV demand [$/kWh^2]

Values highlighted in green or red indicate the EV was available for charge or discharge (selling back to the grid) during that hour, whereas the EV cohort is 
offline for all entries in the default text color. To compute for other EV configurations, edit the EV_CONFIG variable at the top of this file and re-run, following 
the example entry format for each new EV cohort, separated by commas. If tables appear weird, expand your window horizontally or disable text wrapping.
