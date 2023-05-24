import numpy as np
from ypstruct import structure
import bc_module as bc
import dp_module as dp



# Problem Definition
problem= structure()
problem.fitnessfunc = bc.bc
problem.fitnessName="Bifunctional Catalyst"
##defenition of search space
problem.nvar = 10
problem.varmin = 0.6 
problem.varmax = 0.9
#This fitness landscape might be too complicated to measure whether each
##peak was captured like we did before.
#problem.peakposition=[0.1,0.3, 0.5, 0.7,0.9]
#problem.peakintervals=[0, 0.2, 0.4, 0.6,0.8, 1]


params=structure()   
params.maxit = 10 ## maximum iteration
params.npop = 100 ## the size of initial population
params.beta = 1   ## Population Coefficient (if it is 1 then number of children equals to nember of parents) 
params.pc = 1     ## Crossover Rate
params.gamma = 0.1
params.cross=0.9

params.mu = 0.1  ## mutation parameter
params.sigma = 0.1  ## mutation parameter
params.sigma_sharing=0.5  ##Sharing Parameter
params.alpha_sharing=1    ##Sharing Parameter

##u[0]...u[10] are the independent variables
##the optimization problem is to find the values of u that maximize fitness
##u represents the hydrogenation catalyst fraction of the catalyst blend
##There are ten separately controllable values because the reaction
##process is divided into 10 stages.
##In the final code, values of u will be passed in as arguments from main.py
##Values of u are constrained:  0.60 <= u <= 0.90
##The final code will need to search in the 10-dimensional space of u values
##For now, one value of u is provided for testing purposes
u = np.array([0.6661, 0.6734, 0.6764, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

##Evaluate the fitness of u
#output = bc.bc(u)
##Print the fitness of u
#print(output)

##Dynamic programming
u_initial = 0.75
##The initial value of r
##This is the width of the range of u values that will be explored
##during one iteration.
r = 0.15
N = 3
M = 3
gamma = 0.7
iters=4

dp.run(u_initial, r, N, M, gamma,iters)


##
###run_method(ga.run,"Genetic Algorithm",params,problem)
###run_method(sharing.run,"Fittness Sharing Algorithm",params,problem)
###run_method(dc.run,"Deterministic Crowding",params,problem)
