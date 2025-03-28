import copy
import numpy as np

class Evo:
    def __init__(self, random_state=None):
        # Population is maintained as a list of (evaluation, solution) pairs.
        self.pop = []
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
        solutions = [s for (_, s) in self.pop]
        # Use the RNG to select random solutions (deep copy them)
        return [copy.deepcopy(self.rng.choice(solutions)) for _ in range(k)]
    
    def add_solution(self, sol):
        """Evaluate and add a solution to the population."""
        evaluation = tuple((name, f(sol)) for name, f in self.fitness.items())
        self.pop.append((evaluation, sol))
    
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
        new_pop = []
        for i, (eval_i, sol_i) in enumerate(self.pop):
            dominated = False
            for j, (eval_j, sol_j) in enumerate(self.pop):
                if i != j and Evo._dominates(eval_j, eval_i):
                    dominated = True
                    break
            if not dominated:
                new_pop.append((eval_i, sol_i))
        self.pop = new_pop
    
    def run_agent(self, name):
        """Invoke the named agent on randomly selected solution(s)."""
        op, k = self.agents[name]
        picks = self.get_random_solutions(k)
        new_solution = op(picks)
        self.add_solution(new_solution)
    
    def evolve(self, n=1, dom_interval=100, status_interval=1000):
        """
        Run the evolutionary process for n iterations.
        Periodically remove dominated solutions and print status.
        """
        agent_names = list(self.agents.keys())
        for i in range(n):
            pick = self.rng.choice(agent_names)
            self.run_agent(pick)
            if i % dom_interval == 0:
                self.remove_dominated()
            if i % status_interval == 0:
                self.remove_dominated()
                print("Iteration:", i)
                print("Population size:", len(self.pop))
                # Report the best evaluation (using total penalty as a simple aggregate)
                best_eval = min(self.pop, key=lambda x: sum(score for _, score in x[0]))
                print("Best evaluation:", best_eval[0])
        self.remove_dominated()