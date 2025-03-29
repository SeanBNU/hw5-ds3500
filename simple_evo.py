import time
import numpy as np
from assignta import TAAssignment, overallocation
from evo_new import Evo
from profiler import profile

def bit_flip_agent(solutions):
    """
    Simple agent that flips a single random bit in the candidate solution.
    Assumes 'solutions' is a list with one 2D numpy array.
    """
    sol = solutions[0].copy()
    num_rows, num_cols = sol.shape
    # Choose a random TA and lab section
    i = np.random.randint(0, num_rows)
    j = np.random.randint(0, num_cols)
    sol[i, j] = 1 - sol[i, j]  # Flip the bit
    return sol

@profile
def run_evolution():
    # Initialize the TA assignment problem.
    ta_solver = TAAssignment("data/tas.csv", "data/sections.csv")
    
    # Create an instance of the evolutionary framework.
    evo_instance = Evo()
    
    # Register the objective using the TA assignment's objective method.
    evo_instance.add_objective("total_penalty", ta_solver.objective)
    evo_instance.add_objective("overallocation", overallocation)

    
    # Register the bit-flip agent.
    evo_instance.add_agent("bit_flip", bit_flip_agent, k=1)
    
    # Seed the population with an initial random solution.
    initial_solution = ta_solver.random_solution()
    evo_instance.add_solution(initial_solution)
    
    # Evolution parameters:
    time_limit = 10  # seconds
    remove_dom_interval = 100  # remove dominated solutions every 100 iterations
    status_interval = 500      # print status every 500 iterations
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < time_limit:
        # Randomly select an agent to run (only one in this case, but structure supports more)
        agent_names = list(evo_instance.agents.keys())
        pick = np.random.choice(agent_names)
        evo_instance.run_agent(pick)
        
        iteration += 1
        
        # Periodically remove dominated solutions
        if iteration % remove_dom_interval == 0:
            evo_instance.remove_dominated()
        
        # Periodically print status
        if iteration % status_interval == 0:
            evo_instance.remove_dominated()
            print("Iteration:", iteration, "Population size:", len(evo_instance.pop))
    
    # Final pruning
    evo_instance.remove_dominated()
    
    print("\nFinal Population after", iteration, "iterations:")
    print(evo_instance)
    
    # Evaluate best performance for each objective in the final population
    best_scores = {}
    for evaluation, sol in evo_instance.pop.items():
        # evaluation is a tuple of (objective_name, score) pairs.
        for obj_name, score in evaluation:
            if obj_name not in best_scores or score < best_scores[obj_name]:
                best_scores[obj_name] = score
                
    print("\nBest performance for each objective:")
    for obj_name, best in best_scores.items():
        print(f"{obj_name}: {best}")
        
run_evolution()
