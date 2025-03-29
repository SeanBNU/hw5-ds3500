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
def repair_overallocation_agent(candidates):
    """
    Repair agent: removes one assignment from a randomly selected overallocated TA.
    Expects 'candidates' to be a list containing one 2D NumPy array.
    """
    sol = candidates[0].copy()  # make a copy to avoid modifying the original
    
    # Compute number of assignments per TA and get their maximum allowed assignments.
    assignments = np.sum(sol, axis=1)
    max_assigned = tas_df["max_assigned"].to_numpy()
    
    # Find indices of TAs that are overallocated.
    overallocated = np.where(assignments > max_assigned)[0]
    if overallocated.size > 0:
        # Choose one overallocated TA at random.
        ta_idx = np.random.choice(overallocated)
        # Find lab sections that this TA is currently assigned to.
        assigned_labs = np.where(sol[ta_idx] == 1)[0]
        if assigned_labs.size > 0:
            # Remove one random assignment for this TA.
            lab_idx = np.random.choice(assigned_labs)
            sol[ta_idx, lab_idx] = 0
    return sol

@profiler.profile
def repair_conflicts_agent(candidates):
    """
    Repair agent: resolves time conflicts in the candidate solution.
    For each TA, if they are assigned multiple sections with the same time slot,
    remove extra assignments until there is at most one assignment per time slot.
    Expects 'candidates' to be a list containing one 2D NumPy array.
    """
    sol = candidates[0].copy()  # make a copy (as in swapper)
    
    section_times = sections_df["daytime"].to_numpy()
    
    # For each TA (each row in the solution)
    for i in range(sol.shape[0]):
        # Get indices of sections assigned to TA i
        assigned_indices = np.where(sol[i] == 1)[0]
        time_mapping = {}
        # Build a mapping from time slot to the sections assigned in that slot.
        for idx in assigned_indices:
            time_slot = section_times[idx]
            if time_slot in time_mapping:
                time_mapping[time_slot].append(idx)
            else:
                time_mapping[time_slot] = [idx]
        # For time slots with conflict (more than one section) remove extras.
        for time_slot, indices in time_mapping.items():
            while len(indices) > 1:
                # Remove one randomly chosen extra assignment
                remove_idx = np.random.choice(indices[1:])
                sol[i, remove_idx] = 0
                indices.remove(remove_idx)
    return sol

@profiler.profile
def repair_unpreferred_agent(candidates):
    """
    Repair agent: removes unpreferred assignments from the candidate solution.
    Expects 'candidates' to be a list containing one 2D NumPy array.
    """
    sol = candidates[0].copy()
    # Assume the TA availability is stored in a global NumPy array, e.g., loaded in main() as "ta_availability"
    global ta_availability
    # For each TA and each lab section, remove assignment if unpreferred.
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            if sol[i, j] == 1 and ta_availability[i, j] == 'W':
                sol[i, j] = 0
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

    min_ta = sections_df["min_ta"].to_numpy()

    # Assume the lab section columns start at column 3 and there are 17 sections
    global ta_availability
    ta_availability = np.array(tas_df.iloc[:, 3:3+17])
    
    # Create the Evo framework instance with a reproducible random state
    E = evo.Evo(random_state=42)
    
    # Register the swapper agent with Evo
    E.add_agent("swapper", swapper)

    # Register the repair overallocation agent
    E.add_agent("repair overallocation", repair_overallocation_agent) 

    # Register the repair conflicts agent
    E.add_agent("repair conflicts", repair_conflicts_agent)  

    # Register the repair unpreferred agent
    E.add_agent("repair unpreferred", repair_unpreferred_agent)

    # Register the destroy unavailable agent
    E.add_agent("destroy unavailable", destroy_unavailable)
    E.add_agent("destroy unavailable v2", destroy_unavailable)
    
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
    E.evolve(n=1000, dom_interval=30, status_interval=50)
    
    print("\nFinal population:")
    for eval_tuple, sol in E.pop:
        print("Evaluation:", eval_tuple)
    
    # Report profiling results
    profiler.Profiler.report()

if __name__ == "__main__":
    main()
