#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:48:28 2021

@author: carolinarutilidelima
"""
import numpy as np 
import matplotlib.pyplot as plt
from ypstruct import structure
import ga 



# def function to optimize quadract fun
def function(x):
    summ = 0
    lst_summ = []
    for i in x:
        summ = summ + i
        quad = summ ** 2
        lst_summ.append(quad)
    
    fun = sum(lst_summ)
    
    return fun

# Variables according to problem definition
problem = structure()
problem.costfunc = function
problem.nvar = 30 # 30 var
problem.varmin = - 100 # 1-100
problem.varmax = 100 # 100



# GA parameters
params = structure()
params.maxit = 100000 # max iterations number
params.accep = 1 # acceptance for stop the alg
params.npop = 20 # population number
params.pc = 1 # variation initial population
params.gamma = 0.1  # crossover parameter
params.sigma = 0.1 # mutation parameter 
params.mu = 0.01  # mutation % genes
params.beta = 3 # selection for the rolette wheel


#Run GA
 
output = ga.run(problem, params)



# Results  
plt.plot(output.bestcost, 'g', linewidth = 3 )
plt.semilogy(output.bestcost)
plt.xlim(0,output.it)
plt.xlabel('Iterations')
plt.ylabel('Output')
plt.title('Genetic Algorithm (GA)')
plt.show()


    