import numpy as np
import pandas as pd
import pytest

from assignta import undersupport, unavailable, unpreferred, conflicts, overallocation

def load_solution(file_path):
    """
    Loads a candidate solution from a CSV file.
    Expects a CSV file with 40 rows (TAs) and 17 columns (labs).
    """
    return np.loadtxt(file_path, delimiter=',', dtype=int)

def load_sections(file_path='data/sections.csv'):
    """
    Loads section information from a CSV file.
    Returns the 'min_ta' column as a NumPy array.
    """
    df = pd.read_csv(file_path)
    return df['min_ta'].values

def load_tas(file_path='data/tas.csv'):
    """
    Loads TA availability/preferences from a CSV file.
    Expects at least 17 lab section columns labeled "0" through "16".
    """
    df = pd.read_csv(file_path)
    cols = [str(i) for i in range(17)]
    return df[cols].values

# ----------------------------
# PyTest unit tests
# ----------------------------

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 34),
    ("data/test2.csv", 37),
    ("data/test3.csv", 19)
])
def test_overallocation(test_file, expected_penalty):
    solution = load_solution(test_file)
    tas_df = pd.read_csv('data/tas.csv')
    penalty = overallocation(solution, tas_df)
    assert penalty == expected_penalty, f"For {test_file}, expected overallocation penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 7),
    ("data/test2.csv", 5),
    ("data/test3.csv", 2)
])
def test_conflicts(test_file, expected_penalty):
    solution = load_solution(test_file)
    sections_df = pd.read_csv('data/sections.csv')
    penalty = conflicts(solution, sections_df)
    assert penalty == expected_penalty, f"For {test_file}, expected time conflicts penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 1),
    ("data/test2.csv", 0),
    ("data/test3.csv", 11)
])
def test_undersupport(test_file, expected_penalty):
    solution = load_solution(test_file)
    min_ta = load_sections('data/sections.csv')
    penalty = undersupport(solution, min_ta)
    assert penalty == expected_penalty, f"For {test_file}, expected undersupport penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 59),
    ("data/test2.csv", 57),
    ("data/test3.csv", 34)
])
def test_unavailable(test_file, expected_penalty):
    solution = load_solution(test_file)
    ta_availability = load_tas('data/tas.csv')
    penalty = unavailable(solution, ta_availability)
    assert penalty == expected_penalty, f"For {test_file}, expected unavailable penalty {expected_penalty} but got {penalty}"

@pytest.mark.parametrize("test_file, expected_penalty", [
    ("data/test1.csv", 10),
    ("data/test2.csv", 16),
    ("data/test3.csv", 17)
])
def test_unpreferred(test_file, expected_penalty):
    solution = load_solution(test_file)
    ta_availability = load_tas('data/tas.csv')
    penalty = unpreferred(solution, ta_availability)
    assert penalty == expected_penalty, f"For {test_file}, expected unpreferred penalty {expected_penalty} but got {penalty}"