"""

File: sorting.py

Description:  Demonstrate how we can sort a list of numbers
using evolutionary computing WITHOUT implementing
a sorting algorithm.

"""

import random as rnd
import evo_new


# An objective
def stepdowns(L):
    return sum([ x - y for x, y in zip(L,L[1:]) if y < x])


def sumratio(L):
    sz = len(L)
    return round(sum(L[:sz//2]) / sum(L[sz//2+1:]), 5)


# An agent that works on ONE input solution - COPIED from the population
def swapper(solutions):
    L = solutions[0]
    i = rnd.randrange(0, len(L))
    j = rnd.randrange(0, len(L))
    L[i], L[j] = L[j], L[i]
    return L

def main():

    # Create the framework object
    E = evo_new.Evo()
    E.add_agent("swapper", swapper)
    E.add_objective("stepdowns", stepdowns)
    E.add_objective("sumratio", sumratio)



    # Initialize with one random solution
    L = [rnd.randrange(1,99) for _ in range(20)]
    E.add_solution(L)
    print(E)

    E.evolve(n=10000, dom=100, status=1000)
    print(E)

main()







