import numpy as np
import pandas as pd
import os
from assignta import undersupport, unavailable

def load_solution(file_path):
    """
    Loads a candidate solution from a CSV file.
    Expects a CSV file with 40 rows (TAs) and 17 columns (labs).
    """
    return np.loadtxt(file_path, delimiter=',', dtype=int)

def load_sections(file_path='data/sections.csv'):
    """
    Loads section information from a CSV file.
    Expects a column 'min_ta'.
    Returns a numpy array of the min_ta values for each lab section.
    """
    df = pd.read_csv(file_path)
    return df['min_ta'].values

def load_tas(file_path='data/tas.csv'):
    """
    Loads TA availability/preferences from a CSV file.
    Expects columns labeled "0" through "16" representing each lab section.
    Returns a numpy array of shape (40, 17) with string entries.
    """
    df = pd.read_csv(file_path)
    cols = [str(i) for i in range(17)]
    return df[cols].values

def main():
    test_files = ["data/test1.csv", "data/test2.csv", "data/test3.csv"]
    
    # Load common data for the objectives
    min_ta = load_sections('data/sections.csv')
    ta_availability = load_tas('data/tas.csv')
    
    results = []
    
    for test_file in test_files:
        solution = load_solution(test_file)
        us_penalty = undersupport(solution, min_ta)
        ua_penalty = unavailable(solution, ta_availability)
        
        results.append({
            "test_file": os.path.basename(test_file),
            "undersupport": us_penalty,
            "unavailable": ua_penalty
        })
    
    # Create a DataFrame from the results and save it to CSV
    df_results = pd.DataFrame(results)
    output_file = "objective_summary.csv"
    df_results.to_csv(output_file, index=False)
    
    print("Objective summary:")
    print(df_results)
    print(f"Summary saved to {output_file}")

if __name__ == '__main__':
    main()
