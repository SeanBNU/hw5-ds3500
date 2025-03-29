import pandas as pd
import numpy as np

sol = pd.read_csv('data/test3.csv', header = None)
sec = (pd.read_csv('data/sections.csv'))

def _has_conflicts(ta_row, section_times):
    # Get indices of assigned sections
    assigned_indices = np.where(ta_row == 1)[0]
    # Get times of assigned sections
    assigned_times = section_times[assigned_indices]
    # Check if there are duplicate times (conflicts)
    return len(assigned_times) > len(set(assigned_times))

def time_conflicts(solution, sections_df):
    """ 
    Calculate time conflicts in a solution using functional programming.
    A time conflict occurs when a TA is assigned to multiple sections with the same time slot.
    
    Parameters:
        solution: 2D numpy array where rows are TAs and columns are sections
        sections_df: DataFrame containing section information with 'daytime' column
    
    Returns:
        Total number of TAs with time conflicts
    """
    # Extract section times
    section_times = sections_df["daytime"].to_numpy()
    
    # Use the helper function with section_times as an additional parameter
    conflict_map = map(lambda ta_row: _has_conflicts(ta_row, section_times), solution)
    
    # Count total conflicts
    return sum(conflict_map)

sol_array = sol.to_numpy()
print(time_conflicts(sol_array, sec))

