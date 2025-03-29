import numpy as np
import pandas as pd
import pytest

from objectives import *

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
    tas = (pd.read_csv(tas_path)).iloc[:,2:]    
    # Set the global data with just what's needed
    set_global_data({
        'min_ta': sections["min_ta"].to_numpy(),
        'section_times': sections["daytime"].to_numpy(),
        'max_assigned': tas["max_assigned"].to_numpy(),
        'ta_availability': tas.iloc[1:]
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

# ----------------------------
# PyTest unit tests
# ----------------------------

def load_solution(test_file):
    """Load a test solution from a CSV file."""
    return pd.read_csv(test_file).values

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 34),
    ("data/test2.csv", 37),
    ("data/test3.csv", 19)
])
def test_overallocation(test_file, expected_penalty):
    # Load data first
    load_data()
    
    # Load the test solution
    solution = load_solution(test_file)
    
    # Call the function with just the solution
    penalty = overallocation(solution)
    
    assert penalty == expected_penalty, f"For {test_file}, expected overallocation penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 7),
    ("data/test2.csv", 5),
    ("data/test3.csv", 2)
])
def test_conflicts(test_file, expected_penalty):
    # Load data first
    load_data()
    
    # Load the test solution
    solution = load_solution(test_file)
    
    # Call the function with just the solution
    penalty = conflicts(solution)
    
    assert penalty == expected_penalty, f"For {test_file}, expected time conflicts penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 1),
    ("data/test2.csv", 0),
    ("data/test3.csv", 11)
])
def test_undersupport(test_file, expected_penalty):
    # Load data first
    load_data()
    
    # Load the test solution
    solution = load_solution(test_file)
    
    # Call the function with just the solution
    penalty = undersupport(solution)
    
    assert penalty == expected_penalty, f"For {test_file}, expected undersupport penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 59),
    ("data/test2.csv", 57),
    ("data/test3.csv", 34)
])
def test_unavailable(test_file, expected_penalty):
    # Load data first
    load_data()
    
    # Load the test solution
    solution = load_solution(test_file)
    
    # Call the function with just the solution
    penalty = unavailable(solution)
    
    assert penalty == expected_penalty, f"For {test_file}, expected unavailable penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 10),
    ("data/test2.csv", 16),
    ("data/test3.csv", 17)
])
def test_unpreferred(test_file, expected_penalty):
    # Load data first
    load_data()
    
    # Load the test solution
    solution = load_solution(test_file)
    
    # Call the function with just the solution
    penalty = unpreferred(solution)
    
    assert penalty == expected_penalty, f"For {test_file}, expected unpreferred penalty {expected_penalty} but got {penalty}"