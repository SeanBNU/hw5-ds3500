import copy
import numpy as np
import time
import random as rnd

class Evo:
    def __init__(self, random_state=None):
        # Population is maintained as a list of (evaluation, solution) pairs.
        self.pop = {}
        self.fitness = {}   # objectives: name -> function
        self.agents = {}    # agents: name -> (operator function, num_solutions_input)
        self.rng = np.random.default_rng(random_state)
        
    def add_objective(self, name, f):
        """Register a new objective function."""
        self.fitness[name] = f
        
    def add_agent(self, name, op, k=1):
        """Register a new agent with its operator function and input count."""
        self.agents[name] = (op, k)
        
    def get_random_solutions(self, k=1):
        """Randomly select k solutions from the current population."""
        if not self.pop:
            return []
        solutions = list(self.pop.values())  # Get values from dictionary
        # Use the RNG to select random solutions (deep copy them)
        return [copy.deepcopy(self.rng.choice(solutions)) for _ in range(k)]
    
    def add_solution(self, sol):
        """Evaluate and add a solution to the population."""
        evaluation = tuple((name, f(sol)) for name, f in self.fitness.items())
        self.pop[evaluation] = sol  # Use as dictionary key -> value
        
    @staticmethod
    def _dominates(p, q):
        """
        Determine if evaluation p dominates q.
        p and q are tuples of (objective_name, score).
        Lower scores are better.
        """
        pscores = [score for _, score in p]
        qscores = [score for _, score in q]
        score_diffs = [q - p for p, q in zip(pscores, qscores)]
        return min(score_diffs) >= 0.0 and max(score_diffs) > 0.0
    
    def remove_dominated(self):
        """Remove dominated solutions from the population."""
        new_pop = {}
        evaluations = list(self.pop.keys())
        
        for i, eval_i in enumerate(evaluations):
            dominated = False
            for j, eval_j in enumerate(evaluations):
                if i != j and Evo._dominates(eval_j, eval_i):
                    dominated = True
                    break
            if not dominated:
                new_pop[eval_i] = self.pop[eval_i]
                
        self.pop = new_pop
        
    def run_agent(self, name):
        """Invoke the named agent on randomly selected solution(s)."""
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)
    
    def evolve(self, n=1, dom=100, status=1000, runtime = 60):
        """ Run the framework (start evolving solutions)
        n = # of random agent invocations (# of generations) """

        agent_names = list(self.agents.keys())
        start = time.time()
        for i in range(n):
            pick = rnd.choice(agent_names)  # pick an agent to run
            self.run_agent(pick)
            if i % dom == 0:
                self.remove_dominated()
            if i % status == 0:
                self.remove_dominated()
                print("Iteration: ", i)
                print("Population size: ", len(self.pop))
                print('time:',time.time() - start)
            if time.time() - start >= runtime:
                    break
        self.remove_dominated()
        
    def __str__(self):
        """ Output the solutions in the population """
        rslt = ""
        for eval,sol in self.pop.items():
            rslt += str(dict(eval))+":\t"+str(sol)+"\n"
        return rslt