"""
Created on Sun Nov 22 20:01:49 2020

@author: Sima Shafaei & James Richard Lefevre

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from tabulate import tabulate
from ypstruct import structure
import math
import ga
import sharing
import dc

#Cost Function
def M1(x):
    return (math.sin(5*math.pi*x))** 6

def M4(x):
    return math.exp(-2*math.log(2)* ((x-0.08)/0.854)**2 ) * math.sin(5*math.pi*(x**0.75 - 0.05))**6


def show_result(out,method_name,k,problem):
    fitnessfunc=problem.fitnessfunc
    fitnessName=problem.fitnessName


    plt.semilogy(out.bestfitness, color='green', label="Fmax",linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(method_name +" for "+fitnessName)    
    plt.semilogy(out.avgfitness, color='blue', label="Favg",linewidth=2 )    
    plt.semilogy(out.minfitness, color='red' , label="Fmin",linewidth=2)
    plt.legend()
    plt.show()
    
    plt.semilogy(out.bestfitness, color='black',linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Fmax")
    plt.title(method_name +" for "+fitnessName) 
    plt.show()
    
    plt.semilogy(out.avgfitness, color='black',linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Favg")
    plt.title(method_name +" for "+fitnessName) 
    plt.show()
    
    plt.semilogy(out.minfitness, color='black',linewidth=2)
    plt.xlim(0,params.maxit)
    plt.xlabel("Iteration")
    plt.ylabel("Fmin")
    plt.title(method_name +" for "+fitnessName) 
    plt.show()
    # Plot Cost function and initial points:
    data = np.arange(0., 1.01, 0.01)
    outFitness=[]
    for i in range (len(data)):
        outFitness.append(fitnessfunc(data[i]))
    plt.plot(data,outFitness)
    plt.xlabel("x")
    plt.ylabel(fitnessName)
    plt.title("Initial population in "+method_name)
    plt.grid(True)
    for i in range(k):
        plt.plot(out.initpop[i].position,fitnessfunc(out.initpop[i].position),'bo')
    plt.show()
    # Plot Cost function and final point:
    data = np.arange(0., 1.01, 0.01)
    outFitness=[]
    for i in range (len(data)):
        outFitness.append(fitnessfunc(data[i]))
    plt.plot(data,outFitness)
    plt.xlabel("x")
    plt.ylabel(fitnessName)
    plt.title("Final population in "+method_name)
    plt.grid(True)
    for i in range(k):
        plt.plot(out.pop[i].position,fitnessfunc(out.pop[i].position),'ro')
    plt.show()
    
    print("peak capture="+str(out.peakcapture))
    
    
    return
 
    
def run_method(method,method_name, params,problem):
    #method Parameters
    iter=10
    crossover_rate=[1,0.75,0.5,0.25]
    mutation_rate=[0.05,0.1,0.2]
    best_avg=-1*math.inf
    avg=np.zeros((len(crossover_rate),len(mutation_rate)))
    
    for c,crate in enumerate(crossover_rate):
        for m,mrate in enumerate(mutation_rate):
            params.mu = mrate
            params.cross = crate
            print("running {} for mutation rate= {} crossover rate={} ".format(method_name,mrate,crate))
            for i in range(iter):
                 print(".", end = "")
                 r=method(problem, params)
                 avg[c][m]+=r.avgfitness[params.maxit-1]
            avg[c][m]/=iter
            print("average fitness="+str(avg[c][m]))
            if avg[c][m] > best_avg:
                best_avg=avg[c][m]
                bestc=c
                bestmu=m
                result=method(problem, params)
    
                     
    print("best parameters for {}: mutation rate={} , crossover rate={} , avg={}".format(method_name, mutation_rate[bestmu], crossover_rate[bestc],best_avg))
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
    
    


# Problem Definition
problem= structure()
problem.fitnessfunc = M1
problem.fitnessName="M1"
##defenition of search space
problem.nvar = 1
problem.varmin = 0 
problem.varmax = 1 
problem.peakposition=[0.1,0.3, 0.5, 0.7,0.9]
problem.peakintervals=[0, 0.2, 0.4, 0.6,0.8, 1]


params=structure()   
params.maxit = 15 ## maximum iteration
params.npop = 100 ## the size of initial population
params.beta = 1   ## Population Coefficient (if it is 1 then number of children equals to nember of parents) 
params.pc = 1     ## Crossover Rate
params.gamma = 0.1
params.cross=0.9

params.mu = 0.1  ## mutation parameter
params.sigma = 0.1  ## mutation parameter
params.sigma_sharing=0.5  ##Sharing Parameter
params.alpha_sharing=1    ##Sharing Parameter

#popnum=[100,500,1000]
#maxiteration=[10,50,100]

run_method(ga.run,"Genetic Algorithm",params,problem)
run_method(sharing.run,"Fittness Sharing Algorithm",params,problem)
run_method(dc.run,"Deterministic Crowding",params,problem)

problem.fitnessfunc = M4
problem.fitnessName="M4"

run_method(ga.run,"Genetic Algorithm",params,problem)
run_method(sharing.run,"Fittness Sharing Algorithm",params,problem)
run_method(dc.run,"Deterministic Crowding",params,problem)
