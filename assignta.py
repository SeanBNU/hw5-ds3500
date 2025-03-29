import numpy as np
import pandas as pd
import evo  # assuming evo.py is available in the same package

# --- Objective Functions (kept unchanged) ---

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
    return ((test == 1) & (tas == 'W')).sum().sum()

# --- End of Objective Functions ---

# --- Super Simple Agent Function ---
def swapper(candidates):
    """
    Super simple agent: flips one random bit in the candidate solution.
    Expects 'candidates' to be a list with one solution (a 2D NumPy array).
    """
    sol = candidates[0].copy()  # make a copy to avoid modifying the original
    i = np.random.randint(0, sol.shape[0])
    j = np.random.randint(0, sol.shape[1])
    sol[i, j] = 1 - sol[i, j]  # flip the bit (0->1, 1->0)
    return sol

# --- Main Function to Instantiate Evo and Run the Agent ---
def main():
    # Create the Evo framework instance (with reproducibility)
    E = evo.Evo(random_state=42)
    
    # Register the super simple swapper agent
    E.add_agent("swapper", swapper)
    
    # Register a dummy objective function: here we simply sum the bits of the matrix.
    # (In the future, you might register a composite objective based on the penalty functions above.)
    E.add_objective("sum", lambda sol: np.sum(sol))
    
    # Initialize the population with one random solution (40 TAs x 17 lab sections)
    random_solution = np.random.randint(0, 2, size=(40, 17))
    E.add_solution(random_solution)
    
    print("Initial population:")
    print(E)
    
    # Run evolution for a small number of iterations to test the swapper agent
    E.evolve(n=1000, dom_interval=100, status_interval=500)
    
    print("Final population:")
    print(E)

if __name__ == "__main__":
    main()