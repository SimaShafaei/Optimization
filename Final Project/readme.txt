Compilation
-Not applicable (Python scripting code)

Execution
-Run IDLE
-Open the file main.py in IDLE
-Ensure that dc.py, bc_module.py, and dp_module.py are in the same folder or the system PATH
-Edit parameters in main.py as desired
-Run main.py (F5)

Input files
-Not applicable (model is built in to source code)

Pre-conditions
-Python version 3.8 or newer must be installed
-Python packages numpy and scipy must be installed

Parameters (DC)
-g:  population size
-n:  number of iterations
-gamma:  random numbers in crossover are drawn from [-gamma, 1+gamma]
-mu:  probability that a child will mutate (pm)
-sigma: weight factor for mutation
-cross:  probability that crossover will occur when a pair of parents reproduces (pc)

Outputs (DC)
-Graph of Fmax, Fmin, and Favg as a function of iteration is printed to the screen

Parameters (IDP)
-N:  number of grid points per stage (each grid point being a starting point for integration, with grid points evenly spaced across the current search space)         
-M:  number of u values considered per grid point (the grid point will be integrated with M different random u values, and the u value that yields the best fitness is stored for possible later use) 
-r:  width of the search space in the first iteration 
-γ:  factor by which search space shrinks with each iteration 
-u_initial:  the u value that is used for the preliminary integration that generates the grid; this is a scalar value that is applied for each of the 10 stages 
-iters:  the number of times the DP restarts with a new u policy and smaller r 

Output (IDP)
-Graph of fitness as a function of iteration is printed to the screen



