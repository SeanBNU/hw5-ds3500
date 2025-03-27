import numpy as np
import pandas as pd
import pytest
import os

# Import the objective functions from your implementation
from assignta import undersupport, unavailable, unpreferred

def load_solution(file_path):
    """
    Loads a candidate solution from a CSV file.
    Expects a CSV file with 40 rows (TAs) and 17 columns (labs).
    """
    # The CSV is assumed to contain only binary integers, with comma as delimiter.
    return np.loadtxt(file_path, delimiter=',', dtype=int)

def load_sections(file_path='data/sections.csv'):
    """
    Loads section information from a CSV file.
    Expects columns including 'min_ta'.
    Returns a numpy array of the min_ta values for each lab section (length should be 17).
    """
    df = pd.read_csv(file_path)
    # Assuming the column name is exactly "min_ta"
    return df['min_ta'].values

def load_tas(file_path='data/tas.csv'):
    """
    Loads TA availability/preferences from a CSV file.
    Expects a CSV file with 40 rows and at least 17 columns labeled "0" through "16" 
    representing each lab section's availability.
    Returns a numpy array of shape (40, 17) with string entries.
    """
    df = pd.read_csv(file_path)
    # Extract only the columns representing lab sections (assuming they are labeled as strings "0", "1", ... "16")
    cols = [str(i) for i in range(17)]
    return df[cols].values

# ----------------------------
# PyTest unit tests
# ----------------------------

# Expected penalty values from the professor's answer key:
# For undersupport: Test1 => 1, Test2 => 0, Test3 => 11.
@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 1),
    ("data/test2.csv", 0),
    ("data/test3.csv", 11)
])
def test_undersupport(test_file, expected_penalty):
    """
    Test the undersupport objective.
    """
    solution = load_solution(test_file)
    min_ta = load_sections('data/sections.csv')
    penalty = undersupport(solution, min_ta)
    assert penalty == expected_penalty, f"For {test_file}, expected undersupport penalty {expected_penalty} but got {penalty}"

# For unavailable: Test1 => 59, Test2 => 57, Test3 => 34.
@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 59),
    ("data/test2.csv", 57),
    ("data/test3.csv", 34)
])
def test_unavailable(test_file, expected_penalty):
    """
    Test the unavailable objective.
    """
    solution = load_solution(test_file)
    ta_availability = load_tas('data/tas.csv')
    penalty = unavailable(solution, ta_availability)
    assert penalty == expected_penalty, f"For {test_file}, expected unavailable penalty {expected_penalty} but got {penalty}"

# For unpreferred: Test1 => 10, Test2 => 16, Test3 => 17.
@pytest.mark.parametrize("test_file, expected_penalty",[
    ("data/test1.csv", 10),
    ("data/test2.csv", 16),
    ("data/test3.csv", 17)
])
def test_unpreferred(test_file, expected_penalty):
    solution = load_solution(test_file)
    ta_availability = load_tas('data/tas.csv')
    penalty = unpreferred(solution, ta_availability)
    assert penalty == expected_penalty, f"For {test_file}, expected unpreferred penalty {expected_penalty} but got {penalty}"