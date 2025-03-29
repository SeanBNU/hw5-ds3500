
# TA Assignment Optimizer 🧠

Welcome to the TA Assignment Optimizer! This project uses evolutionary algorithms to solve the complex problem of assigning TAs to course sections while balancing multiple competing objectives.

## 🚀 What's This All About?

Ever tried to create the perfect schedule for teaching assistants? It's like solving a Rubik's cube blindfolded! This tool helps you:

- Match TAs to course sections based on their preferences
- Avoid scheduling conflicts (nobody can be in two places at once!)
- Ensure each section has enough support
- Respect TA workload limits
- Optimize for everyone's happiness!

## 🛠️ Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- Pandas
- A sense of adventure!

### Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/SeanBNU/hw5-ds3500.git
   cd hw5-ds3500
   ```

2. Make sure you have all dependencies:
   ```bash
   pip install numpy pandas
   ```

## 🎮 How to Use

1. Prepare your data:
   - `sections.csv`: Contains info about each section (time slots, minimum TA requirements)
   - `tas.csv`: Contains TA availability and preferences

2. Run the optimizer:
   ```bash
   python assignta.py
   ```

3. Check out the results in `darwinzz_summary.csv`

## 🧪 How It Works

Our evolutionary algorithm uses these objective functions:
- **Overallocation**: Prevents TAs from being assigned too many sections
- **Conflicts**: Ensures TAs aren't scheduled for multiple sections at the same time
- **Undersupport**: Makes sure each section has enough TAs
- **Unavailable**: Avoids assigning TAs when they're unavailable
- **Unpreferred**: Minimizes assignments to sections TAs prefer not to teach

The algorithm uses various "agents" to evolve solutions:
- **Swapper**: Makes random changes to assignments
- **Repair agents**: Fix specific issues in candidate solutions
- **Destroy unavailable**: Removes problematic assignments

## 🏆 Team Darwinzz

This project was created by Team Darwinzz, masters of evolutionary algorithms and scheduling wizards!

## 📊 Performance Profiling

Want to see how efficient our code is? Check out the profiler report after running the optimizer to see which functions are taking the most time.

Happy optimizing! 🎉
