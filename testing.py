import pandas as pd
import numpy as np
import evo
import random as rnd
from assignta import undersupport, unavailable, unpreferred, conflicts, overallocation

#Agents
def swapper(solutions):
    L = solutions[0]
    i = rnd.randrange(0, len(L))
    j = rnd.randrange(0, len(L))
    L[i], L[j] = L[j], L[i]
    return L

def main():

    # Create the framework object
    E = evo.Evo()
    E.add_agent("swapper", swapper)
    E.add_objective("overallocation", overallocation)

    # Initialize with one random solution
    L = np.random.randint(0, 2, size=(40,16))
    E.add_solution(L)
    print(E)

    E.evolve(n=10000, dom=100, status=1000)
    print(E)

main()

    
