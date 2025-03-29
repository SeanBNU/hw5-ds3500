import pandas as pd
import numpy as np
import evo
import random as rnd
import profiler

_DATA_CACHE = {}
np.random.seed(42)

def set_global_data(data_dict):
    """
    Set global data that can be accessed by objective functions.
    
    Args:
        data_dict (dict): Dictionary containing data to be stored globally
    """
    global _DATA_CACHE
    _DATA_CACHE = data_dict

def get_global_data(key=None):
    """
    Get global data, either a specific key or the entire dictionary.
    
    Args:
        key (str, optional): Specific data key to retrieve. Defaults to None.
        
    Returns:
        The requested data or the entire data dictionary
    """
    global _DATA_CACHE
    if key is not None:
        return _DATA_CACHE.get(key)
    return _DATA_CACHE

@profiler.profile
def load_data(sections_path='data/sections.csv', tas_path='data/tas.csv'):
    """
    Load and preprocess all required data for objective functions.
    
    Args:
        sections_path (str): Path to the sections CSV file
        tas_path (str): Path to the TAs CSV file
    """
    sections = pd.read_csv(sections_path, usecols=["min_ta", "daytime"])
    
    ta_cols = ["max_assigned"] + [str(i) for i in range(17)]
    tas = pd.read_csv(tas_path, usecols=ta_cols)
    
    set_global_data({
        'min_ta': sections["min_ta"].to_numpy(),
        'section_times': sections["daytime"].to_numpy(),
        'max_assigned': tas["max_assigned"].to_numpy(),
        'ta_availability': tas.iloc[:,1:].to_numpy()
    })
    
    print("Data loaded successfully.")
    
@profiler.profile
def overallocation(solution):
    """
    Compute overallocation penalty.
    
    Args:
        solution (numpy.ndarray): Binary matrix of TA assignments
        
    Returns:
        int: Total overallocation penalty
    """
    max_assigned = get_global_data('max_assigned')
    assignments = np.sum(solution, axis=1)
    penalty = np.maximum(assignments - max_assigned, 0)
    return int(sum(penalty))

@profiler.profile
def conflicts(solution):
    """
    Calculate time conflicts in a solution.
    A time conflict occurs when a TA is assigned to multiple sections with the same time slot.
    
    Args:
        solution (numpy.ndarray): Binary matrix of TA assignments
        
    Returns:
        int: Number of time conflicts
    """
    section_times = get_global_data('section_times')
    conflict_count = 0
    row_sums = np.sum(solution, axis=1)
    for i in np.where(row_sums >= 2)[0]:
        assigned_times = section_times[solution[i] == 1]
        if len(set(assigned_times)) < len(assigned_times):
            conflict_count += 1

    return conflict_count

@profiler.profile
def undersupport(solution):
    """
    Compute undersupport penalty.
    
    Args:
        solution (numpy.ndarray): Binary matrix of TA assignments
        
    Returns:
        int: Total undersupport penalty
    """
    min_ta = get_global_data('min_ta')
    lab_assignments = solution.sum(axis=0)
    differences = min_ta - lab_assignments
    differences[differences < 0] = 0
    return int(differences.sum())

@profiler.profile
def unavailable(solution):
    """
    Compute unavailable penalty.
    
    Args:
        solution (numpy.ndarray): Binary matrix of TA assignments
        
    Returns:
        int: Number of assignments where TAs are unavailable
    """
    ta_availability = get_global_data('ta_availability')
    
    return int(((solution == 1) & (ta_availability == "U")).sum())

@profiler.profile
def unpreferred(solution):
    """
    Compute the unpreferred penalty.
    Penalty for assigning TAs to labs they would prefer not to teach.
    
    Args:
        solution (numpy.ndarray): Binary matrix of TA assignments
        
    Returns:
        int: Number of assignments with weak preference
    """
    ta_availability = get_global_data('ta_availability')
    
    return int(((solution == 1) & (ta_availability == 'W')).sum())

@profiler.profile
def swapper(solutions):
    """
    Randomly modify a few assignments in the solution.
    
    Args:
        solutions (list): List containing one solution matrix
        
    Returns:
        numpy.ndarray: Modified solution with random changes
    """
    L = solutions[0].copy()  
    for _ in range(3):  
        ta = rnd.randrange(0, L.shape[0])
        section = rnd.randrange(0, L.shape[1])
        L[ta, section] = 1 - L[ta, section]
    return L

@profiler.profile
def repair_overallocation_agent(candidates):
    """
    Repair agent: removes one assignment from a randomly selected overallocated TA.
    
    Args:
        candidates (list): List containing one solution matrix
        
    Returns:
        numpy.ndarray: Solution with reduced overallocation
    """
    sol = candidates[0].copy()
    
    assignments = np.sum(sol, axis=1)
    max_assigned = get_global_data('max_assigned')
    
    overallocated = np.where(assignments > max_assigned)[0]
    if overallocated.size > 0:
        ta_idx = np.random.choice(overallocated)
        assigned_labs = np.where(sol[ta_idx] == 1)[0]
        if assigned_labs.size > 0:
            lab_idx = np.random.choice(assigned_labs)
            sol[ta_idx, lab_idx] = 0
    return sol

@profiler.profile
def repair_conflicts_agent(candidates):
    """
    Repair agent: resolves time conflicts in the candidate solution.
    For each TA, if they are assigned multiple sections with the same time slot,
    remove extra assignments until there is at most one assignment per time slot.
    
    Args:
        candidates (list): List containing one solution matrix
        
    Returns:
        numpy.ndarray: Solution with reduced time conflicts
    """
    sol = candidates[0].copy()
    
    section_times = get_global_data('section_times')
    
    for i in range(sol.shape[0]):
        assigned_indices = np.where(sol[i] == 1)[0]
        time_mapping = {}
        for idx in assigned_indices:
            time_slot = section_times[idx]
            if time_slot in time_mapping:
                time_mapping[time_slot].append(idx)
            else:
                time_mapping[time_slot] = [idx]
        for time_slot, indices in time_mapping.items():
            while len(indices) > 1:
                remove_idx = np.random.choice(indices[1:])
                sol[i, remove_idx] = 0
                indices.remove(remove_idx)
    return sol

@profiler.profile
def repair_unpreferred_agent(candidates):
    """
    Repair agent: removes unpreferred assignments from the candidate solution.
    
    Args:
        candidates (list): List containing one solution matrix
        
    Returns:
        numpy.ndarray: Solution with reduced unpreferred assignments
    """
    sol = candidates[0].copy()
    ta_availability = get_global_data('ta_availability')
    for i in range(sol.shape[0]):
        for j in range(sol.shape[1]):
            if sol[i, j] == 1 and ta_availability[i, j] == 'W':
                sol[i, j] = 0
    return sol

@profiler.profile
def destroy_unavailable(candidates):
    """
    Agent focused on removing unavailable assignments.
    Identifies and removes assignments where TAs are marked as unavailable.
    Also attempts to reassign those sections to available TAs when possible.
    
    Args:
        candidates (list): List containing one solution matrix
        
    Returns:
        numpy.ndarray: Solution with reduced unavailable assignments
    """
    sol = candidates[0].copy()
    ta_availability = get_global_data('ta_availability')
    
    unavailable_mask = (sol == 1) & (ta_availability == 'U')
    unavailable_positions = np.where(unavailable_mask)
    
    if len(unavailable_positions[0]) > 0:
        idx = np.random.randint(0, len(unavailable_positions[0]))
        ta, section = unavailable_positions[0][idx], unavailable_positions[1][idx]
        
        sol[ta, section] = 0
        
        max_assigned = get_global_data('max_assigned')
        ta_assignments = np.sum(sol, axis=1)
        
        available_mask = (ta_availability[:, section] == 'P') & (ta_assignments < max_assigned)
        available_tas = np.where(available_mask)[0]
        
        if len(available_tas) > 0:
            new_ta = np.random.choice(available_tas)
            sol[new_ta, section] = 1
    
    return sol

def main():
    """
    Main function to run the TA assignment optimization.
    Loads data, sets up the evolutionary framework, and runs the optimization.
    """
    
    load_data()

    E = evo.Evo(random_state=42)
    E.add_agent("swapper", swapper)
    E.add_agent("repair_overallocation", repair_overallocation_agent)
    E.add_agent("repair_conflicts", repair_conflicts_agent)
    E.add_agent("repair_unpreferred", repair_unpreferred_agent)
    E.add_agent("destroy_unavailable", destroy_unavailable)
    E.add_objective("overallocation", overallocation)
    E.add_objective("conflicts", conflicts)
    E.add_objective("undersupport", undersupport)
    E.add_objective('unavailable',unavailable)
    E.add_objective('unpreferred',unpreferred)

    L = np.random.randint(0, 2, size=(40,17))
    E.add_solution(L)

    E.evolve(n=2000000, dom=10, status=1000, runtime=300)

    # Print final results
    print("\nFinal population:")
    best_eval = list(E.pop.keys())[0]
    best_scores = dict(best_eval)
    
    print("\nBest solution scores:")
    for objective, score in best_scores.items():
        print(f"  {objective}: {score}")
    
    # Create summary table of Pareto-optimal solutions
    summary_data = []
    for evaluation in E.pop.keys():
        row = {"groupname": "darwinzz"}
        for obj, val in evaluation:
            row[obj] = val
        summary_data.append(row)
    
    # Convert to DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    # Ensure columns are in the correct order
    summary_df = summary_df[["groupname", "overallocation", "conflicts", 
                            "undersupport", "unavailable", "unpreferred"]]
    summary_df.to_csv("/Users/shouryayadav/Documents/courses/DS3500/hw5-ds3500/darwinzz_summary.csv", index=False)
    
    profiler.Profiler.report()

main()