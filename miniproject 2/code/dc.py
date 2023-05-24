# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:09:33 2020

@author: Sima Shafaei & James Richard Lefevre
"""

import numpy as np
from ypstruct import structure
from numpy.random import default_rng
import math


def run(problem, params):
    
    # Problem Informtaion
    fitnessfunc = problem.fitnessfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    ## Added by Sima
    peakposition=problem.peakposition
    peakintervals=problem.peakintervals
    
    # Parameters    
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc =params.pc
    ## nc = number of children
    ## Deterministic crowding always runs npop/2 iterations that
    ## produce the same number of children as the old population
    nc = npop
    ##nc = int(np.round(pc*npop/2)*2) 
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma
    cross = params.cross
    
    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position= None
    empty_individual.fitness = None
    # Eligibility for parenthood
    empty_individual.eligible = None
   
      
    # Initialiaze Population
    pop = empty_individual.repeat(npop)
    initpop = empty_individual.repeat(npop)
    for i in range (npop):
        pop[i].position =np.random.uniform(varmin,varmax,nvar)
        initpop[i].position = pop[i].position
    for i in range (npop):
        pop[i].fitness=fitnessfunc(pop[i].position)
        initpop[i].fitness=pop[i].fitness
        
    #print("Starting positions: {}".format([x.fitness for x in pop]))        

    # Best, average, and worst fitness of iterations
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
        number_eligible = npop
        for i in range (npop):
            pop[i].eligible = True
        
        for _ in range(nc//2):
            rng = default_rng()
            #Randomly pick a parent
            #Pick1 is a pick of *eligible* parents
            pick1 = rng.integers(number_eligible)
            #print("eligible: {} pick1: {}".format(number_eligible, pick1))
            #Translate pick1 into an index of *all* parents
            index1 = find_index(pop,pick1)
            #Copy from the list into parent1
            
            p1 = pop[index1].copy()
            pop[index1].eligible = False
            number_eligible = number_eligible - 1

            pick2 = rng.integers(number_eligible)
            #print("eligible: {} pick2: {}".format(number_eligible, pick2))
            index2 = find_index(pop,pick2)

            #print("size of pop: {}".format(len(pop)))
            p2 = pop[index2].copy()
            pop[index2].eligible = False
            number_eligible = number_eligible - 1
            
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
            
            #Selection tournament occurs every time 2 offspring are generated
            if (dist(p1,c1) + dist(p2, c2)) <= (dist(p1, c2) + dist(p2, c1)):
                if c1.fitness > p1.fitness:
                    del pop[index1]
                    if (index2 > index1):
                        index2 -= 1
                    popc.append(c1)
                if c2.fitness > p2.fitness:
                    del pop[index2]
                    popc.append(c2)
            else:
                if c2.fitness > p1.fitness:
                    del pop[index1]
                    if (index2 > index1):
                        index2 -= 1
                    popc.append(c2)
                if c1.fitness > p2.fitness:
                    del pop[index2]
                    popc.append(c1)
            
            
        #At this point some number x (between 0 and population size) of children have been added to popc
        #The same number x of parents have been deleted from pop
        #print("Number of replacements: {}".format(len(popc)))
        #print("Size of parent population: {}".format(len(pop)))
        pop += popc
        #print("New population size: {}".format(len(pop)))
        pop = sorted(pop, key=lambda x: x.fitness, reverse = True)
        #Should not be necessary to truncate pop
        #pop = pop[0:npop]
        
        #Store Best, average, and worst Fitness
        bestfitness[it] = pop[0].fitness
        avgfitness[it] = sum([ x.fitness for x in pop])/npop
        minfitness[it] = pop[npop-1].fitness
        bestposition[it] = pop[0].position
        
        #Show Iteration Information
        #Added by Sima
        best_points=best_points_per_peak(pop,peakposition,peakintervals)
        report="Iteration {}: ".format(it)
        peakcapture=[]
        for i in range(len(peakposition)):
            if best_points[i] != None:
                t=fitnessfunc(best_points[i].position)/fitnessfunc(peakposition[i])
                peakcapture.append(t)
                report=report+ "Peak{}: {:.2f}   ".format(i, t ) 
            else:
                report=report+ "Peak{}: Not Found  ".format(i)
                peakcapture.append(0)
        #print(report)
        #print("Iteration {}: Best Fitness = {} for x in position {}".format(it, bestfitness[it],bestposition[it]))
        #print("Iteration {}: Avg Fitness = {}".format(it, avgfitness[it]))

    
            
    #Output
    out = structure()
    out.peakcapture = peakcapture
    out.initpop=initpop
    out.pop=pop
    out.bestsol = pop[0]
    #Next line commented out by Jim
    #out.bestfitness = pop[0].fitness
    #Next line added by Jim; this sets out.bestfitness to equal an array that holds the best fitness from each iteration
    out.bestfitness = bestfitness
    out.avgfitness = avgfitness
    out.minfitness = minfitness
    return out
                

def crossover(p1, p2, cross,  gamma=0.1):
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

#Returns the index of an eligible parent, given a random number
def find_index(pop, pick):
    
    counter = pick
    index = 0
    while counter > 1:
        if pop[index].eligible == True:
            counter = counter - 1
        index = index + 1
    #print("returning index {}".format(index))
    return index

#Returns the genotypic distance between 2 individuals
def dist(i1, i2):
    distance = abs(i1.position - i2.position)
    return distance

def distance(x, y):
    #Euclidean Distance
    return math.sqrt((x- y)**2)

#Added by Sima
def best_points_per_peak(pop,peaks,intervals):
    min_dist=[math.inf]*len(peaks)
    best_points=[None]*len(peaks)
    for i,p in enumerate(peaks):        
        for x in pop:
            d=distance(x.position,p)
            if d<min_dist[i] and x.position>intervals[i] and x.position<intervals[i+1]:
                min_dist[i]=d
                best_points[i]=x.deepcopy()                
    return best_points
