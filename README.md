This code solves the 4-point function as a Schrodinger equation. See https://arxiv.org/abs/2303.12119 for more details.
More specifically it solves F(p) = \int_{u ub} e^{-i(u-ub)p}F(u,ub) as a function of the splitting fraction z and the angle theta.
You run it by running run.py.
run.py uses the functions from functions.py and parameter values from configs.py. 
It requires the packages numpy, pandas, numba and scipy

The values for the physical and numerical parameters are stored in config.py, and can be changed.
The grid size N should be at least 50 for decent results. More is better.

The main function creates a numpy file containing the time, finit Nc 1, finite Nc 2, large-Nc 1, large-Nc 2, factorizable 2, analytic result for large-Nc 1, analytic result for factorizable 2.
These correspond to Q1 and Q2 from the paper, where Q2 is the physical state that we are interested in.
When loading the file (using np.load()) you get a numpy array with [0] being the time, [1] being the full solution 1, etc.

See the notebook plot_Nc.ipynb and plot_eikonal.ipynb as well as plot_functions.py for how to load the files and plot them