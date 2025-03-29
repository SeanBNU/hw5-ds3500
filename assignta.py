import numpy as np
import pandas as pd

def overallocation(solution, tas_df):
    """
     Compute overallocation penalty.
     For each TA, if the number of labs assigned exceeds their max_assigned,
     the penalty is the excess. Sum over all TAs.
     """
     # Load the tas dataframe which contains max_assigned
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

class TAAssignment:
    """
    Class for solving the TA assignment problem using evolutionary algorithms.
    """
    def __init__(self, tas_file, sections_file, random_state=None):
        self.tas_df = pd.read_csv(tas_file)
        self.sections_df = pd.read_csv(sections_file)
        self.num_tas = len(self.tas_df)
        self.num_sections = len(self.sections_df)
        # Extract availability data (assuming lab section columns immediately follow 'max_assigned')
        self.ta_availability = np.array(self.tas_df.iloc[:, 2:2+self.num_sections])
        self.min_ta = self.sections_df["min_ta"].to_numpy()
        self.rng = np.random.default_rng(random_state)
        
    def objective(self, solution, weights=None):
        if weights is None:
            weights = {
                'overallocation': 1.0,
                'conflicts': 1.0,
                'undersupport': 1.0,
                'unavailable': 1.0
            }
        penalties = {
            'overallocation': overallocation(solution, self.tas_df),
            'conflicts': conflicts(solution, self.sections_df),
            'undersupport': undersupport(solution, self.min_ta),
            'unavailable': unavailable(solution, self.ta_availability)
        }
        total_penalty = sum(weights[k] * penalties[k] for k in weights)
        return total_penalty

    def random_solution(self):
        """Generate a random binary assignment matrix."""
        return self.rng.integers(0, 2, size=(self.num_tas, self.num_sections))
    
    def mutation_agent(self, solution, mutation_rate=0.05):
        """Flip bits randomly with the given mutation rate."""
        mutant = solution.copy()
        flip = self.rng.random(solution.shape) < mutation_rate
        mutant[flip] = 1 - mutant[flip]
        return mutant

    def crossover_agent(self, sol1, sol2):
        """Perform single-point crossover between two solutions."""
        point = self.rng.integers(1, self.num_tas)
        child = np.vstack((sol1[:point, :], sol2[point:, :]))
        return child

    def repair_agent(self, solution):
        """
        Repair the solution to reduce overallocation and undersupport.
        For TAs over-assigned, remove assignments until meeting max_assigned.
        For sections undersupported, add assignments from TAs not yet assigned.
        """
        repaired = solution.copy()
        assignments = np.sum(repaired, axis=1)
        max_assigned = self.tas_df["max_assigned"].to_numpy()
        for i in range(self.num_tas):
            while assignments[i] > max_assigned[i]:
                ones = np.where(repaired[i] == 1)[0]
                if len(ones) == 0:
                    break
                j = self.rng.choice(ones)
                repaired[i, j] = 0
                assignments[i] -= 1
        assigned_per_section = np.sum(repaired, axis=0)
        min_ta = self.sections_df["min_ta"].to_numpy()
        for j in range(self.num_sections):
            while assigned_per_section[j] < min_ta[j]:
                available_tas = np.where(repaired[:, j] == 0)[0]
                if len(available_tas) == 0:
                    break
                i = self.rng.choice(available_tas)
                repaired[i, j] = 1
                assigned_per_section[j] += 1
        return repaired

    def random_agent(self):
        """Generate a new random solution."""
        return self.random_solution()
    
    def solve(self, population_size=100, generations=200, weights=None):
        """
        Stub for solving the TA assignment problem.
        This should be replaced by an evolutionary algorithm.
        Currently, it returns a random solution.
        """
        return self.random_solution()

# Removed example code; the main block now only demonstrates the stub usage.
if __name__ == "__main__":
    ta_solver = TAAssignment("data/tas.csv", "data/sections.csv", random_state=42)
    solution = ta_solver.solve()
    print("Stub solution generated:\n", solution)