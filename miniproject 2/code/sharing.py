# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:55:14 2020

@author: Sima Shafaei & James Richard Lefevre

implemetation of sharing method which is one of the niching methods

parameters of sharing : 
 
alpha: is a constant (typically set to 1) used to regulate the shape of the sharing function
sigma: The threshold of dissimilarity is specified by a constant sigma_share 
(if the distance between two population elements is greater than or equal to 
sigma_share they do not affect each other's shared fitness)
"""

import numpy as np
from ypstruct import structure
import math

def run(problem, params):
    
    # Problem Informtaion
    fitnessfunc = problem.fitnessfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    peakposition=problem.peakposition
    peakintervals=problem.peakintervals
    
    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc =params.pc       #crossover rate
    ## nc = number of children
    nc = int(np.round(pc*npop/2)*2) 
    gamma = params.gamma  
    mu = params.mu        # mutation rate
    sigma = params.sigma
    cross = params.cross
    
    ##Sharing Parameters
    sigma_sharing=params.sigma_sharing
    alpha_sharing=params.alpha_sharing
    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position= None
    empty_individual.fitness = None
    
    
    # Initialiaze Population
    pop = empty_individual.repeat(npop)
    initpop = empty_individual.repeat(npop)
    for i in range (npop):
        pop[i].position = np.random.uniform(varmin,varmax,nvar)
        initpop[i].position = pop[i].position
    for i in range (npop):
        pop[i].fitness=sharing_fittness(pop[i].position,pop,sigma_sharing, alpha_sharing,fitnessfunc)
        initpop[i].fitness=pop[i].fitness
    
    # Best fitness of iterations
    bestfitness = np.empty(maxit)
    bestposition = np.empty(maxit)
    avgfitness = np.empty(maxit)
    minfitness = np.empty(maxit)
    
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
            c1.fitness = sharing_fittness(c1.position,pop,sigma_sharing, alpha_sharing,fitnessfunc)
            c2.fitness = sharing_fittness(c2.position,pop,sigma_sharing, alpha_sharing,fitnessfunc)
            
            #Add Offsprings to popc
            popc.append(c1)
            popc.append(c2)
            
        # Merge Sort and Select
        pop += popc 
        pop = sorted(pop, key=lambda x: x.fitness, reverse = True)
        pop = pop[0:npop]
        
        #Store Best Fitness
        bestfitness[it] = pop[0].fitness
        bestposition[it] = pop[0].position
        avgfitness[it] = sum([ x.fitness for x in pop])/npop
        minfitness[it] = pop[npop-1].fitness
        
        
        #Show Iteration Information
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
        
    
            
    #Output
    
    out = structure()
    out.peakcapture = peakcapture
    out.initpop=initpop
    out.pop=pop
    out.bestsol = pop[0]
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
    
def roulette_wheel_selection(p):
    c=np.cumsum(p)
    r=sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
    


def distance(x, y):
    #Euclidean Distance
    return math.sqrt((x- y)**2)


def sharing(dist, sigma, alpha):
    if dist<sigma:
        return 1-(dist/sigma)**alpha
    else: 
        return 0    
    

def sharing_fittness(position, pop, sigma_sharing, alpha_sharing,fitnessfunc):
    s=0
    for j in range(len(pop)):
        dist=distance(position,pop[j].position)
        s+= sharing(dist,sigma_sharing,alpha_sharing)
        
    sharingfitt=fitnessfunc(position)/s
    return sharingfitt

def best_points_per_peak(pop,peaks,intervals):
    min_dist=[math.inf]*len(peaks)
    best_points=[None]*len(peaks)
    for i,p in enumerate(peaks):        
        for x in pop:
            d=distance (x.position,p)
            if d<min_dist[i] and x.position>intervals[i] and x.position<intervals[i+1]:
                min_dist[i]=d
                best_points[i]=x.deepcopy()                
    return best_points
                



 

    