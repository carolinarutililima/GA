#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 16:48:28 2021

@author: carolinarutilidelima
"""

import math
import numpy as np 
import matplotlib.pyplot as plt
from ypstruct import structure # encapsulates variables 
import ga 




# def function to optimize ackley fun
def function(x):
    D = 30
    cosx = 0
    listcos = []
    for i in x:
        cosx = math.cos(2*math.pi*i) 
        listcos.append(cosx)
    sumcos = sum(listcos)     
    
    sumx = sum(x**2)    
    p1 = -0.2* math.sqrt(1/D * sumx)
    p2 = 1/D *  sumcos  
    fun = 20 + math.exp(1) - 20*math.exp(p1) - math.exp(p2)
    return fun


# Variables according to problem definition
problem = structure()
problem.costfunc = function
problem.nvar = 30 #30 var
problem.varmin = - 32 # -32
problem.varmax = 32 # 32



# GA parameters
params = structure()
params.maxit = 100000 # max iterations number
params.accep = 0.01# acceptance for stop the alg
params.npop = 20 # population number
params.pc = 1# variation initial population
params.gamma = 0.1 # crossover parameter
params.sigma = 0.3 # mutation parameter 
params.mu = 0.03 # mutation % genes
params.beta = 1 # selection for the rolette wheel


#Run GA
output = ga.run(problem, params)



# Results  
plt.plot(output.bestcost, 'b', linewidth = 3)
plt.semilogy(output.bestcost)
plt.xlim(0,output.it)
plt.xlabel('Iterations')
plt.ylabel('Output')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()



    