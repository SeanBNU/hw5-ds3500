import numpy as np
import pandas as pd
import pytest
from assignta_new import *

def load_solution(test_file):
    """Load a test solution from a CSV file."""
    return pd.read_csv(test_file, header=None).values

@pytest.fixture(scope="module", autouse=True)
def setup():
    """Load data once before all tests"""
    load_data()

@pytest.mark.parametrize("test_file, expected", [
    ("data/test1.csv", 34),
    ("data/test2.csv", 37),
    ("data/test3.csv", 19)
])
def test_overallocation(test_file, expected):
    solution = load_solution(test_file)
    assert overallocation(solution) == expected

@pytest.mark.parametrize("test_file, expected", [
    ("data/test1.csv", 7),
    ("data/test2.csv", 5),
    ("data/test3.csv", 2)
])
def test_conflicts(test_file, expected):
    solution = load_solution(test_file)
    assert conflicts(solution) == expected

@pytest.mark.parametrize("test_file, expected", [
    ("data/test1.csv", 1),
    ("data/test2.csv", 0),
    ("data/test3.csv", 11)
])
def test_undersupport(test_file, expected):
    solution = load_solution(test_file)
    assert undersupport(solution) == expected

@pytest.mark.parametrize("test_file, expected", [
    ("data/test1.csv", 59),
    ("data/test2.csv", 57),
    ("data/test3.csv", 34)
])
def test_unavailable(test_file, expected):
    solution = load_solution(test_file)
    assert unavailable(solution) == expected