
import copy
import numpy as np
import pandas as pd
from evo import Evo
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
    Stores data in the global data cache so that objective functions can access it.
    """
    # Directly extract only what's needed from CSVs
    # For sections, only read the required columns
    sections = pd.read_csv(sections_path, usecols=["min_ta", "daytime"])
    
    # For TAs, only read the required columns
    ta_cols = ["max_assigned"] + [str(i) for i in range(17)]
    tas = pd.read_csv(tas_path, usecols=ta_cols)
    
    # Set the global data with just what's needed
    set_global_data({
        'min_ta': sections["min_ta"].to_numpy(),
        'section_times': sections["daytime"].to_numpy(),
        'max_assigned': tas["max_assigned"].to_numpy(),
        'ta_availability': tas[[str(i) for i in range(17)]].values
    })
    
    print("Data loaded successfully.")

# Objective functions declared outside the Evo class
def overallocation(solution):
    """
    Compute overallocation penalty.
    For each TA, if the number of labs assigned exceeds their max_assigned,
    the penalty is the excess. Sum over all TAs.
    """
    max_assigned = get_global_data('max_assigned')
    assignments = np.sum(solution, axis=1)
    penalty = np.maximum(assignments - max_assigned, 0)
    return int(np.sum(penalty))

def conflicts(solution):
    """ 
    Calculate time conflicts in a solution.
    A time conflict occurs when a TA is assigned to multiple sections with the same time slot.
    """
    section_times = get_global_data('section_times')
    
    def _has_conflicts(ta_row):
        # Get indices of assigned sections
        assigned_indices = np.where(ta_row == 1)[0]
        # Get times of assigned sections
        assigned_times = section_times[assigned_indices]
        # Check if there are duplicate times (conflicts)
        return len(assigned_times) > len(set(assigned_times))
    
    # Use the helper function with section_times
    conflict_map = map(lambda ta_row: _has_conflicts(ta_row), solution)
    
    # Count total conflicts
    return sum(conflict_map)

def undersupport(solution):
    """
    Compute the undersupport penalty.
    Penalty for having fewer TAs assigned to a lab than the minimum required.
    """
    min_ta = get_global_data('min_ta')
    
    # Sum the assignments for each lab (column)
    lab_assignments = solution.sum(axis=0)
    # Calculate penalty: only if assignments are less than the required minimum
    differences = min_ta - lab_assignments
    # Only count positive differences (i.e., when undersupport exists)
    differences[differences < 0] = 0
    return int(differences.sum())

def unavailable(solution):
    """
    Compute the unavailable penalty.
    Penalty for assigning TAs to labs they marked as unavailable.
    """
    ta_availability = get_global_data('ta_availability')
    
    # Vectorized operation to count assignments where TA is unavailable
    return np.sum((solution == 1) & (ta_availability == "U"))

def unpreferred(solution):
    """
    Compute the unpreferred penalty.
    Penalty for assigning TAs to labs they would prefer not to teach.
    """
    ta_availability = get_global_data('ta_availability')
    
    # Count matching 1s where TAs marked 'W' (weak preference)
    return np.sum((solution == 1) & (ta_availability == 'W'))

def swapper(solutions):
    L = solutions[0]
    i = rnd.randrange(0, len(L))
    j = rnd.randrange(0, len(L))
    L[i], L[j] = L[j], L[i]
    return L


'''def main():
    load_data(sections_path='data/sections.csv', tas_path='data/tas.csv')
    # Create the framework object
    E = Evo()
    E.add_agent("swapper", swapper)
    E.add_objective("overallocation", overallocation)

    # Initialize with one random solution
    L = np.random.randint(0, 2, size=(40,17))
    E.add_solution(L)
    print(E)

    E.evolve(n=10000)
    print(E)

main()'''