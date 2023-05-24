import numpy as np
from ypstruct import structure
import time 
import bc_module as bc
import dp_module as dp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from tabulate import tabulate
import dc
import ga
import math

###################
# this function detect if a point is local maximum in a function or not
# it uses a delta as input parameter which is a very small value and 
# for a pint u(u[0], ... , u[k]) computes all k neighbors like 
# V=(u[0], ...,u[i]+delta, u[i+1],...,u[k]) if a point has the maximum value among 
# all neighbors it would be a local maximum
def local_max(u, delta,func):
    dim=len(u)
    a=func(u)
    v=[0]*dim
    for i in range(dim):
        for j in range(dim):
            v[j]=u[i]
        v[i]=v[i]+delta
        b=func(v)
        if b>a:
            return False
    return True

###########################
#This function will visualize Fmin, Fmax and Favg by extracting information from out
# out can be the result of ga, sharing or dc methods
def show_result(out,method_name,k,problem):

    plt.semilogy(out.bestfitness, color='green', label="Fmax",linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(method_name)    
    plt.semilogy(out.avgfitness, color='blue', label="Favg",linewidth=2 )    
    plt.semilogy(out.minfitness, color='red' , label="Fmin",linewidth=2)
    plt.legend()
    plt.show()
    
    plt.semilogy(out.bestfitness, color='black',linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Fmax")
    plt.title(method_name) 
    plt.show()
    
    plt.semilogy(out.avgfitness, color='black',linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Favg")
    plt.title(method_name) 
    plt.show()
    
    plt.semilogy(out.minfitness, color='black',linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Fmin")
    plt.title(method_name) 
    plt.show()  
    
    return

#######################
# to select best mutation and cross over rate we run each method for different 
# value of Pm and Pc and choosed best parameters based on the average fitness 
# over 10 iteration 
def select_best_param(method,method_name, params,problem):
    #method Parameters
    iter=10
    crossover_rate=[1,0.75,0.5,0.25]
    mutation_rate=[0.05,0.1,0.2]
    best=-1*math.inf
    avg=np.zeros((len(crossover_rate),len(mutation_rate)))
    
    for c,crate in enumerate(crossover_rate):
        for m,mrate in enumerate(mutation_rate):
            params.mu = mrate
            params.cross = crate            
            for i in range(iter):
                 print(".", end = "")
                 r=method(problem, params)
                 avg[c][m]+=r.bestfitness[params.maxit-1]
            avg[c][m]/=iter
            print("fitness of {} for mutation rate= {} crossover rate={} is: {}".format(method_name,mrate,crate, avg[c][m]))
            if avg[c][m] > best:
                best=avg[c][m]
                bestc=c
                bestmu=m
                result=method(problem, params)
    
                     
    print("best parameters for {}: mutation rate={} , crossover rate={} , fitness={}".format(method_name, mutation_rate[bestmu], crossover_rate[bestc],best))
    show_result(result,method_name,params.npop,problem)
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(crossover_rate,mutation_rate)
    ax.plot_wireframe(X,Y,np.transpose(avg),color='black')
    ax.plot_surface(X,Y,np.transpose(avg),cmap='viridis')
    ax.set_xlabel('crossover rate')
    ax.set_ylabel('mutation rate')
    ax.set_zlabel('average fitness in '+method_name)
    plt.show()
    
    headers=mutation_rate
    table=np.c_[np.transpose([crossover_rate]),avg]
    print(tabulate(table,headers, tablefmt="grid")) 
    
    return mutation_rate[bestmu], crossover_rate[bestc]

#########################
# main program
    
# Problem Definition : defining fitness function, constraints and number of variables 
# of the problem
problem= structure()
problem.fitnessfunc = bc.bc
problem.fitnessName="Bifunctional Catalyst"
##defenition of search space
problem.nvar = 10
problem.varmin = 0.6 
problem.varmax = 0.9

# Parameter Definition: defining parameters in heuristic methods (sharing,DC and GA)
params=structure()   
params.maxit = 20        ## maximum iteration
params.npop = 30         ## the size of initial population
params.beta = 1          ## Population Coefficient (if it is 1 then number of children equals to nember of parents) 
params.pc = 1            ## Crossover Rate
params.gamma = 0.1
params.cross=0.9

params.mu = 0.1           ## mutation parameter
params.sigma = 0.1        ## mutation parameter
params.sigma_sharing=0.5  ##Sharing Parameter
params.alpha_sharing=1    ##Sharing Parameter


##Dynamic programming Parameter Definition
u_initial = 0.75
gamma = 0.7
iters=8                # number of iteration
##The initial value of r
##This is the width of the range of u values that will be explored
##during one iteration.
r = 0.3

# to select best value for M and N we run dynammip programming for different 
# value of M and N and select the parameters that create highest fitness value 
# in 8 
N = [3,10,20,30]
M = [3,5,7]


t=np.zeros((len(M),len(N)))
bestfitness=np.zeros((len(M),len(N)))
best=-1*math.inf
for i,m in enumerate(M):
    for j,n in enumerate(N):
        start = time.time()
        a=dp.run(u_initial, r, n, m, gamma,iters)
        stop = time.time()
        print("Dynamic Programming run time:  {}".format(stop - start))
        t[i][j]=(stop - start)
        bestfitness[i][j]=a
        if a > best:
            best=a
            bestN=n
            bestM=m


print("best parameters for IDP: M={} , N={} , best={}".format( bestM, bestN,best))

# visualizing the result of parameter tuning experiment in a table
headers=N
table=np.c_[np.transpose([M]),bestfitness]
print(tabulate(table,headers, tablefmt="grid"))
# visualizing the time of parameter tuning experiment in a table
headers=N
table=np.c_[np.transpose([M]),t]
print(tabulate(table,headers, tablefmt="grid"))
# visualizing the time of parameter tuning experiment using a 3D plot
fig = plt.figure()
ax = Axes3D(fig)
X, Y = np.meshgrid(M,N)
ax.plot_wireframe(X,Y,np.transpose(t),color='black')
ax.plot_surface(X,Y,np.transpose(t),cmap='viridis')
ax.set_xlabel('M')
ax.set_ylabel('N')
ax.set_zlabel('time of executing Dynamic programming')
plt.show()


# after selecting best value for N and M we run the dp method for 20 iteration
#and with best value selected for N and M 
iters=15
start = time.time()
a=dp.run(u_initial, r, bestN,bestM, gamma,iters)
stop = time.time()
print("Dynamic Programming execution time:  {}".format(stop - start))

#r=ga.run(problem, params)
#print(r.bestfitness)

# Running heuristic method:
# First we choose the best Pm and Pc for dc 
[params.mu,params.cross]=select_best_param(dc.run,"dc", params,problem)
# Then we run DC with best Pm and Pc and record the time for npop =100 and maxit=50
params.maxit = 50 ## maximum iteration
params.npop = 100 ## the size of initial population
start = time.time()
r=dc.run(problem, params)
stop = time.time()
print("Deterministic Crowding execution time:  {}".format(stop - start))

