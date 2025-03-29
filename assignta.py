import numpy as np
import pandas as pd
import evo  # our evolutionary framework from evo.py
import profiler  # for profiling our agent functions

# --- Objective Functions (unchanged) ---

def overallocation(solution, tas_df):
    assignments = np.sum(solution, axis=1)
    max_assigned = tas_df["max_assigned"].to_numpy()
    penalty = np.maximum(assignments - max_assigned, 0)
    return int(np.sum(penalty))

def _has_conflicts(ta_row, section_times):
    assigned_indices = np.where(ta_row == 1)[0]
    assigned_times = section_times[assigned_indices]
    return len(assigned_times) > len(set(assigned_times))

def conflicts(solution, sections_df):
    section_times = sections_df["daytime"].to_numpy()
    conflict_map = map(lambda ta_row: _has_conflicts(ta_row, section_times), solution)
    return sum(conflict_map)

def undersupport(solution, min_ta):
    lab_assignments = solution.sum(axis=0)
    differences = min_ta - lab_assignments
    differences[differences < 0] = 0
    return int(differences.sum())

def unavailable(solution, ta_availability):
    penalty = 0
    num_TAs, num_labs = solution.shape
    for i in range(num_TAs):
        for j in range(num_labs):
            if solution[i, j] == 1 and ta_availability[i, j] == "U":
                penalty += 1
    return penalty

def unpreferred(test, tas):
    return int(((test == 1) & (tas == 'W')).sum().sum())

# --- End of Objective Functions ---


# --- Super Simple Agent Function ---
@profiler.profile
def swapper(candidates):
    """
    Super simple agent: flips one random bit in the candidate solution.
    Expects 'candidates' to be a list containing one 2D NumPy array.
    """
    sol = candidates[0].copy()  # make a copy to avoid modifying the original
    i = np.random.randint(0, sol.shape[0])
    j = np.random.randint(0, sol.shape[1])
    sol[i, j] = 1 - sol[i, j]  # flip the bit (0 becomes 1 and vice versa)
    return sol

@profiler.profile
def destroy_unavailable(candidates):
    """
    Agent that eliminates one random assignment from a candidate solution
    where the TA is unavailable.
    
    Expects:
      - candidates: a list containing one 2D NumPy array (solution)
      - ta_availability: a 2D NumPy array of strings matching the solution's shape,
                         where 'U' indicates an unavailable assignment.
    
    Operation:
      - Find all indices (i, j) where the candidate solution has a 1 (assignment)
        and the corresponding ta_availability is 'U'.
      - Randomly choose one such index and set that assignment to 0.
    """
    sol = candidates[0].copy()  # work on a copy to preserve the original
    
    # Find all indices where there is an assignment and the TA is unavailable.
    # This uses vectorized boolean indexing.
    mask = (sol == 1) & (ta_availability == 'U')
    indices = np.argwhere(mask)
    
    if indices.size > 0:
        # Randomly choose one index among the unavailable assignments
        random_index = indices[np.random.choice(len(indices))]
        sol[random_index[0], random_index[1]] = 0
    
    return sol

# --- Main Function to Run Evolution ---
def main():
    # Set seed for reproducibility (affects np.random.randint used in swapper)
    np.random.seed(42)
    
    # Load data for evaluation functions
    global tas_df
    tas_df = pd.read_csv("data/tas.csv")
    global sections_df
    sections_df = pd.read_csv("data/sections.csv")
    global min_ta
    min_ta = sections_df["min_ta"].to_numpy()
    # Assume the lab section columns start at column 3 and there are 17 sections
    global ta_availability
    ta_availability = np.array(tas_df.iloc[:, 3:3+17])
    
    # Create the Evo framework instance with a reproducible random state
    E = evo.Evo(random_state=42)
    
    # Register the swapper agent with Evo
    E.add_agent("swapper", swapper)
    E.add_agent("unavailable", destroy_unavailable)

    
    # Register our evaluation functions using the real objective functions
    E.add_objective("overallocation", lambda sol: overallocation(sol, tas_df))
    E.add_objective("conflicts", lambda sol: conflicts(sol, sections_df))
    E.add_objective("undersupport", lambda sol: undersupport(sol, min_ta))
    E.add_objective("unavailable", lambda sol: unavailable(sol, ta_availability))
    E.add_objective("unpreferred", lambda sol: unpreferred(sol, ta_availability))
    
    # Initialize population with one random solution (40 TAs x 17 labs)
    initial_solution = np.random.randint(0, 2, size=(40, 17))
    E.add_solution(initial_solution)
    
    print("Initial population:")
    for eval_tuple, sol in E.pop:
        print("Evaluation:", eval_tuple)
    
    # Run evolution for a given number of iterations
    E.evolve(n=10000, dom_interval=100, status_interval=20)
    
    print("\nFinal population:")
    for eval_tuple, sol in E.pop:
        print("Evaluation:", eval_tuple)
    
    # Report profiling results
    profiler.Profiler.report()

if __name__ == "__main__":
    main()
