import pandas as pd
import numpy as np

def load_sections(file_path='data/sections.csv'):
    """
    Loads section information from a CSV file.
    Expects columns including 'min_ta'.
    """
    df = pd.read_csv(file_path)
    # Assuming the column name is exactly "min_ta"
    return df

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

tas = load_tas()
sections = load_sections()

def overallocation(solution):
    """
    Compute overallocation penalty.
    For each TA, if the number of labs assigned exceeds their max_assigned,
    the penalty is the excess. Sum over all TAs.
    """
    # Load the tas dataframe which contains max_assigned
    tas_df = pd.read_csv('data/tas.csv')
    assignments = np.sum(solution, axis=1)
    max_assigned = tas_df["max_assigned"].to_numpy()
    penalty = np.maximum(assignments - max_assigned, 0)
    return int(np.sum(penalty))

def _has_conflicts(ta_row, section_times):
    # Get indices of assigned sections
    assigned_indices = np.where(ta_row == 1)[0]
    # Get times of assigned sections
    assigned_times = section_times[assigned_indices]
    # Check if there are duplicate times (conflicts)
    return len(assigned_times) > len(set(assigned_times))

def conflicts(solution, sections_df):
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

def undersupport(solution, min_ta):
    """
    Compute the undersupport penalty.
    
    Parameters:
        solution: A 2D numpy array (shape: 40 x 17) representing the assignment matrix.
                  Each row is a TA and each column is a lab section.
                  An element is 1 if the TA is assigned to that lab, and 0 otherwise.
        min_ta: A 1D numpy array of length 17 representing the minimum number of TAs required per lab.
    
    Returns:
        An integer penalty which is the sum over all labs of (min_ta - actual TA assignments),
        counting only where the actual number is less than the minimum.
    """
    # Sum the assignments for each lab (column)
    lab_assignments = solution.sum(axis=0)
    # Calculate penalty: only if assignments are less than the required minimum
    differences = min_ta - lab_assignments
    # Only count positive differences (i.e., when undersupport exists)
    differences[differences < 0] = 0
    return int(differences.sum())

def unavailable(solution, ta_availability):
    """
    Compute the unavailable penalty.
    
    Parameters:
        solution: A 2D numpy array (40 x 17) where each cell is 1 (assigned) or 0 (not assigned).
        ta_availability: A 2D numpy array (40 x 17) of strings representing each TA's availability
                         for each lab section. The value "U" indicates the TA is unavailable.
    
    Returns:
        An integer penalty equal to the number of assignments where the TA is marked as "U".
    """
    penalty = 0
    num_TAs, num_labs = solution.shape
    # Loop through each TA and lab to check for assignments conflicting with unavailability
    for i in range(num_TAs):
        for j in range(num_labs):
            if solution[i, j] == 1 and ta_availability[i, j] == "U":
                penalty += 1
    return penalty

def unpreferred(test, tas):
    # Count matching 1s across all elements
    return ((test == 1) & (tas == 'W')).sum().sum()

print(tas, sections)