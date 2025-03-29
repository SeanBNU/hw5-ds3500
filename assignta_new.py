import pandas as pd
import numpy as np
import evo_new
import random as rnd
_DATA_CACHE = {}
def set_global_data(data_dict):
    """Set global data that can be accessed by objective functions."""
    global _DATA_CACHE
    _DATA_CACHE = data_dict

def get_global_data(key=None):
    """Get global data, either a specific key or the entire dictionary."""
    global _DATA_CACHE
    if key is not None:
        return _DATA_CACHE.get(key)
    return _DATA_CACHE

def load_data(sections_path='data/sections.csv', tas_path='data/tas.csv'):
    """
    Load and preprocess all required data for objective functions.
    """
    # Load sections data
    sections = pd.read_csv(sections_path, usecols=["min_ta", "daytime"])
    
    # Load TA data (header assumed)
    ta_cols = ["max_assigned"] + [str(i) for i in range(17)]
    tas = pd.read_csv(tas_path, usecols=ta_cols)
    
    # Set the global data
    set_global_data({
        'min_ta': sections["min_ta"].to_numpy(),
        'section_times': sections["daytime"].to_numpy(),
        'max_assigned': tas["max_assigned"].to_numpy(),
        'ta_availability': tas.iloc[:,1:].to_numpy()  # Columns 1-17 are availability
    })
    
    print("Data loaded successfully.")

def overallocation(solution):
    """Compute overallocation penalty."""
    max_assigned = get_global_data('max_assigned')
    assignments = np.sum(solution, axis=1)
    penalty = np.maximum(assignments - max_assigned, 0)
    return int(sum(penalty))

def conflicts(solution):
    """ 
    Calculate time conflicts in a solution.
    A time conflict occurs when a TA is assigned to multiple sections with the same time slot.
    """
    section_times = get_global_data('section_times')
    
    def _has_conflicts(ta_row):
        assigned_indices = np.where(ta_row == 1)[0]
        # Get times of assigned sections
        assigned_times = section_times[assigned_indices]
        return len(assigned_times) > len(set(assigned_times))
    
    conflict_map = map(_has_conflicts, solution)
    return sum(conflict_map)

def undersupport(solution):
    """Compute undersupport penalty."""
    min_ta = get_global_data('min_ta')
    # Sum columns (labs)
    lab_assignments = solution.sum(axis=0)
    differences = min_ta - lab_assignments
    differences[differences < 0] = 0
    return int(differences.sum())

def unavailable(solution):
    """Compute unavailable penalty."""
    ta_availability = get_global_data('ta_availability')
    
    return int(((solution == 1) & (ta_availability == "U")).sum())

def unpreferred(solution):
    """
    Compute the unpreferred penalty.
    Penalty for assigning TAs to labs they would prefer not to teach.
    """
    ta_availability = get_global_data('ta_availability')
    
    # Count matching 1s where TAs marked 'W' (weak preference)
    return int(((solution == 1) & (ta_availability == 'W')).sum())
#Agents
def swapper(solutions):
    """
    Randomly modify a few assignments in the solution.
    """
    L = solutions[0].copy()  
    for _ in range(3):  
        ta = rnd.randrange(0, L.shape[0])
        section = rnd.randrange(0, L.shape[1])
        L[ta, section] = 1 - L[ta, section]
    return L

def main():
    # Load the data first
    load_data()

    # Create the framework object
    E = evo_new.Evo(random_state=42)
    E.add_agent("swapper", swapper)
    E.add_objective("overallocation", overallocation)
    E.add_objective("conflicts", conflicts)
    E.add_objective("undersupport", undersupport)
    E.add_objective('unavailable',unavailable)
    E.add_objective('unpreferred',unpreferred)

    L = np.random.randint(0, 2, size=(40,17))
    E.add_solution(L)

    E.evolve(n=1000000, dom=10, status=100, runtime=5)

    # Print final results
    print("\nFinal population:")
    best_eval = list(E.pop.keys())[0]
    best_score = dict(best_eval)["overallocation"]
    print(f"Best overallocation score: {best_score}")    
    print(E)

main()
