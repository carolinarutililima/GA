#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:09:58 2021

@author: carolinarutilidelima
"""
import numpy as np
from ypstruct import structure


def run(problem, params):


    #Problem definition 
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    
    
    
    # Parameters 
    maxit = params.maxit
    accep = params.accep
    npop = params.npop
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma
    beta = params.beta
    

    
    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None
    
    # BestSolution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf 
    
    
    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(0, npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = costfunc(pop[i].position)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()
    
    # Best Cost of Interactions
    bestcost = []
    
    it = 0 
    
    # Main Loop
    while bestsol.cost > accep and maxit > it:
    #for it in range(maxit):
        
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0 :
            costs = costs/avg_cost
        
        probs = np.exp(-beta*costs)
        
        popc = []
        for _ in range(nc//2): # nc = number of children 



            # select parents RANDOM SELECTION
            #q = np.random.permutation(npop)
            #p1 = pop[q[0]]
            #p2 = pop[q[1]]        

            # Perform Roulette Wheel Selection
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]
        
            
            # Perform Crossover           
            c1, c2 = crossover(p1,p2,gamma)
            
            # Perform Mutation 
            c1 = mutate(c1, mu, sigma)
            c2 = mutate(c2, mu, sigma)
            
            # Check limits
            apply_bound(c1, varmin, varmax)
            apply_bound(c2, varmin, varmax)
            
            # Evaluate First Offspring
            c1.cost = costfunc(c1.position)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()
                
            # Evaluate Second Offspring
            c2.cost = costfunc(c2.position)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy() 
                
            # Add Offsprings to population
            popc.append(c1) 
            popc.append(c2) 
            
        # Merge, sort and select
        pop = pop + popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]
        
        
        # Store best cost
        bestcost.append(bestsol.cost)
    
        
        # Show info each int
        print( "Interation {}: best output/cost = {}".format(it,bestcost[it]))
       
        it = it + 1
            
        
    # Output     
    output = structure()
    output.pop = pop
    output.bestsol = bestsol
    output.bestcost = bestcost
    output.it = it
    return output 

# Uniform crossover
def crossover(p1, p2, gamma):   
    
    c1 = p1.deepcopy()
    c2 = p2.deepcopy()
    alpha = np.random.uniform(-gamma,1+gamma, c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)* p2.position
    c2.position = alpha*p2.position + (1-alpha)* p1.position
    
    return c1,c2

# Mutation
def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <=  mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    
    return y


# Check limits    
def apply_bound(x, varmin, varmax): # shuffle the result 
    x.postion = np.maximum(x.position, varmin)
    x.postion = np.minimum(x.position, varmax)
    
    return x
    
    
# Roulette wheel selection    
def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)

    return ind[0][0]    

    

    
    