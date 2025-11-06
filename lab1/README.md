# Lab 1 - Optimization and Modelling

This directory contains solutions to the three exercises in Lab 1 for course 2MA918.

## Files

- `e1.py` - Exercise 1: Warm-up LP-problem
- `e2.py` - Exercise 2: Large LP-problems  
- `e3.py` - Exercise 3: Sensitivity analysis
- `lab1.tex` - Complete LaTeX report with all answers and explanations
- `img/` - Directory containing generated figures

## Running the Exercises

### Prerequisites
Install required Python packages:
```bash
pip install numpy scipy matplotlib
```

### Exercise 1: Warm-up LP-problem
```bash
python3 e1.py
```

Solves a TV production optimization problem:
- Defines the LP model with decision variables, objective, and constraints
- Plots the feasible region and identifies vertices
- Adds level curves to visualize the objective function
- Verifies the optimum at all extreme points
- Uses scipy.optimize.linprog to find the maximum
- Solves in standard form with slack variables

**Result:** Optimal production is 800 type A TVs and 300 type B TVs for a profit of $860,000.

### Exercise 2: Large LP-problems
```bash
python3 e2.py
```

Investigates computational performance of optimization methods:
- Explains advantages of matrix/vector notation
- Tests timing for different problem sizes
- Finds simplex method exceeds 1 second at m=n=150
- Finds HiGHS method exceeds 1 second at m=n=1400
- Creates performance comparison plots

**Result:** HiGHS is approximately 9.3× faster than simplex for large problems.

### Exercise 3: Sensitivity analysis
```bash
python3 e3.py
```

Performs sensitivity analysis on the TV production problem:
- Formulates and solves the dual problem
- Computes shadow prices for all constraints
- Analyzes investment scenarios for 100 extra hours
- Determines price increase needed to change optimal solution
- Evaluates whether new TV type C should be produced
- Calculates quality inspection hours needed

**Key Results:**
- Shadow prices: Stage I = 150, Stage II = 0, Stage III = 125
- Invest in Stage I (highest shadow price)
- Type B price must increase by $200 to change solution
- Type C should be produced (reduced cost = 50)
- Need 490 hours for quality inspection

## LaTeX Report

The file `lab1.tex` contains a complete report with:
- All mathematical formulations
- Explanations of methods and results
- References to generated figures
- Detailed answers to all sub-questions

Compile with:
```bash
pdflatex lab1.tex
```

## Generated Figures

- `img/ex1_feasible_region.png` - Feasible region with constraint lines
- `img/ex1_level_curves.png` - Feasible region with objective function level curves
- `img/ex2_performance_comparison.png` - Simplex vs HiGHS performance comparison
