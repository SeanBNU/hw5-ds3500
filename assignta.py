import numpy as np

def overallocation(solution, tas_df):
    """
    Compute overallocation penalty.
    For each TA, if the number of labs assigned exceeds their max_assigned,
    the penalty is the excess. Sum over all TAs.
    """
    assignments = np.sum(solution, axis=1)
    max_assigned = tas_df["max_assigned"].to_numpy()
    penalty = np.maximum(assignments - max_assigned, 0)
    return int(np.sum(penalty))

def conflicts(solution, sections_df):
    """
    Compute time conflicts.
    Assumes sections_df contains a 'time' column indicating meeting time.
    For each TA, if they are assigned to more than one section with the same time,
    count 1 conflict for that TA.
    """
    conflict_count = 0
    # Assume the order of sections in sections_df matches the solution's columns.
    section_times = sections_df["daytime"].to_numpy()
    num_tas = solution.shape[0]
    for i in range(num_tas):
        assigned_indices = np.where(solution[i] == 1)[0]
        if len(assigned_indices) > 0:
            times = section_times[assigned_indices]
            if len(np.unique(times)) < len(times):
                conflict_count += 1
    return conflict_count


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