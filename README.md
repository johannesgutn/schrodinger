This code solves the 4-point function as a Schrodinger equation. You run it by running run.py.
run.py uses the functions from functions.py and parameter values from configs.py. 
It requires the packages numpy, pandas, numba and scipy

The values for the physical and numerical parameters are stored in config.py, and can be changed.
The grid size N should be at least 50 for okay results. More is better.

The main function creates a numpy file containing the time, full solution 1, full solution 2, Nc solution 1, Nc solution 2 and the analytic result for Nc solution 1.
When loading the file (using np.load()) you get a numpy array with [0] being the time, [1] beeing the full solution 1, etc.
See the notebook plots.ipynb for how to load the files and plot them