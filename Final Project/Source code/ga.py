# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:09:33 2020

@author: Sima Shafaei & James Richard Lefevre
"""

import numpy as np
from ypstruct import structure


def run(problem, params):
    
    # Problem Informtaion
    fitnessfunc = problem.fitnessfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    
    # Parameters    
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc =params.pc
    
    ## nc = number of children
    nc = int(np.round(pc*npop/2)*2) 
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma
    cross = params.cross
    
    
    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position= None
    empty_individual.fitness = None
    
        
    
    # Initialiaze Population
    pop = empty_individual.repeat(npop)
    initpop = empty_individual.repeat(npop)
    for i in range (npop):
        pop[i].position =np.random.uniform(varmin,varmax,nvar)
        initpop[i].position = pop[i].position
    for i in range (npop):
        pop[i].fitness=fitnessfunc(pop[i].position)
        initpop[i].fitness=pop[i].fitness
        
            
    # Best mean and Worst fitness of iterations    
    bestfitness = np.empty(maxit)
    avgfitness = np.empty(maxit)
    minfitness = np.empty(maxit)
    bestposition = np.empty(maxit)
    
    # Main Loop of GA
    for it in range(maxit):
        fitnesses = np.array([ x.fitness for x in pop])
        avg_fitness= np.mean(fitnesses)
        if avg_fitness != 0:
            fitnesses = fitnesses/avg_fitness
        probs = np.exp(-beta*fitnesses)
        
        popc = []
        for _ in range(nc//2):
            
            #Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
            
            # Perform Crossover
            c1, c2=crossover(p1, p2, cross, gamma)
            
            # Perform Mutation
            c1=mutate(c1, mu, sigma)
            c2=mutate(c2, mu, sigma)
            
            # Apply Bounds
            apply_bounds(c1, varmin, varmax)
            apply_bounds(c2, varmin, varmax)
            
            #Evaluate Offsprings
            c1.fitness = fitnessfunc(c1.position)
            c2.fitness = fitnessfunc(c2.position)         
            
            #Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)
            
        # Merge Sort and Select
        pop += popc 
        pop = sorted(pop, key=lambda x: x.fitness, reverse = True)
        pop = pop[0:npop]
        
        #Store Best Fitness
        bestfitness[it] = pop[0].fitness
        avgfitness[it] = sum([ x.fitness for x in pop])/npop
        minfitness[it] = pop[npop-1].fitness
        #bestposition[it] = pop[0].position
        
        #Show Iteration Information
        print("Iteration {}: Best Fitness = {} ".format(it, bestfitness[it]))
         
    #Output
    out = structure()
    out.initpop=initpop
    out.pop=pop
    out.bestsol = pop[0]
    out.bestfitness = bestfitness
    out.avgfitness = avgfitness
    out.minfitness = minfitness
    return out
              

def crossover(p1, p2, cross, gamma=0.1):
     c1 = p1.deepcopy()
     c2 = p2.deepcopy()
     if np.random.random() <= cross:
         alpha = np.random.uniform(-gamma,1+gamma, *c1.position.shape)
         c1.position = alpha*p1.position + (1-alpha)*p2.position
         c2.position = alpha*p2.position + (1-alpha)*p1.position
     return c1, c2
    
def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = (np.random.rand(*x.position.shape) <= mu)
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    return y

def apply_bounds(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)
    
def roulette_wheel_selection(p):
    c=np.cumsum(p)
    r=sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

