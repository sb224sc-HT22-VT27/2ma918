"""
Exercise 3: Sensitivity analysis
Performing sensitivity analysis on the TV production LP problem from Exercise 1.
"""

import numpy as np
from scipy.optimize import linprog, OptimizeResult
import matplotlib.pyplot as plt

print("=" * 60)
print("Exercise 3: Sensitivity analysis")
print("=" * 60)

# Original problem from Exercise 1
print("\nOriginal Problem Setup:")
c = np.array([700, 1000])  # Objective: profit per TV type A and B
A = np.array([
    [3, 5],   # Stage I
    [1, 3],   # Stage II
    [2, 2]    # Stage III
])
b = np.array([3900, 2100, 2200])

print(f"Objective coefficients: c = {c}")
print(f"Constraint matrix A =\n{A}")
print(f"Right-hand side b = {b}")

# Solve the primal problem
result_primal = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')
print(f"\nPrimal optimal solution: x* = {result_primal.x}")
print(f"Primal optimal value: z* = {-result_primal.fun}")

# i) Formulate and solve the dual problem
print("\n" + "=" * 60)
print("i) Dual Problem:")
print("=" * 60)

print("""
Primal problem:
  maximize    700*x1 + 1000*x2
  subject to  3*x1 + 5*x2 <= 3900  (Stage I)
              1*x1 + 3*x2 <= 2100  (Stage II)
              2*x1 + 2*x2 <= 2200  (Stage III)
              x1, x2 >= 0

Dual problem:
  minimize    3900*y1 + 2100*y2 + 2200*y3
  subject to  3*y1 + 1*y2 + 2*y3 >= 700   (for x1)
              5*y1 + 3*y2 + 2*y3 >= 1000  (for x2)
              y1, y2, y3 >= 0

where y1, y2, y3 are the dual variables (shadow prices).
""")

# Dual problem matrices
c_dual = b  # Dual objective is primal RHS
A_dual = A.T  # Dual constraints are transpose of primal
b_dual = c  # Dual RHS is primal objective

# Solve dual (note: dual is minimization with >= constraints, so we use -A_ub)
result_dual = linprog(c_dual, A_ub=-A_dual, b_ub=-b_dual, bounds=[(0, None)]*3, method='highs')

print(f"Dual optimal solution: y* = {result_dual.x}")
print(f"Dual optimal value: w* = {result_dual.fun}")
print(f"\nVerification:")
print(f"  Primal optimal value: z* = {-result_primal.fun:.2f}")
print(f"  Dual optimal value:   w* = {result_dual.fun:.2f}")
print(f"  Strong duality holds: {np.isclose(-result_primal.fun, result_dual.fun)}")

# ii) Compute shadow prices
print("\n" + "=" * 60)
print("ii) Shadow Prices:")
print("=" * 60)

shadow_prices = result_dual.x
print(f"Shadow prices (dual variables): y* = {shadow_prices}")
print(f"  Stage I   (y1): {shadow_prices[0]:.4f}")
print(f"  Stage II  (y2): {shadow_prices[1]:.4f}")
print(f"  Stage III (y3): {shadow_prices[2]:.4f}")

print("""
Interpretation:
- Shadow price represents the rate of change in the optimal objective
  value per unit increase in the resource availability.
- A shadow price of 0 indicates the constraint is not binding (there's slack).
- Higher shadow prices indicate more valuable resources.
""")

# iii) 100 extra working hours - where to invest?
print("\n" + "=" * 60)
print("iii) 100 Extra Working Hours Analysis:")
print("=" * 60)

print("\nTesting the impact of adding 100 hours to each stage:")

for i, stage_name in enumerate(['Stage I', 'Stage II', 'Stage III']):
    b_modified = b.copy()
    b_modified[i] += 100
    
    result_modified = linprog(-c, A_ub=A, b_ub=b_modified, bounds=[(0, None), (0, None)], method='highs')
    
    profit_increase = -result_modified.fun - (-result_primal.fun)
    predicted_increase = shadow_prices[i] * 100
    
    print(f"\n{stage_name}:")
    print(f"  Shadow price: {shadow_prices[i]:.4f}")
    print(f"  Predicted profit increase: {predicted_increase:.2f}")
    print(f"  Actual profit increase: {profit_increase:.2f}")
    print(f"  New optimal value: {-result_modified.fun:.2f}")

print("""
Recommendation:
- Invest in the stage with the highest shadow price for maximum benefit.
- Avoid investing in stages with shadow price of 0 (non-binding constraints).
""")

if shadow_prices[0] > shadow_prices[1] and shadow_prices[0] > shadow_prices[2]:
    print("*** INVEST in Stage I ***")
elif shadow_prices[1] > shadow_prices[0] and shadow_prices[1] > shadow_prices[2]:
    print("*** INVEST in Stage II ***")
else:
    print("*** INVEST in Stage III ***")

if np.min(shadow_prices) == 0:
    avoid_stage = np.argmin(shadow_prices)
    print(f"*** DO NOT invest in Stage {avoid_stage + 1} (shadow price = 0) ***")

# iv) Price increase for type B to change optimal solution
print("\n" + "=" * 60)
print("iv) Price Increase for Type B TV:")
print("=" * 60)

print(f"\nCurrent optimal solution: x* = {result_primal.x}")
print(f"Current price for type B: {c[1]}")

print("\nTesting different price increases for type B:")

# Test increasing prices
for price_increase in [0, 50, 100, 150, 200, 250, 300]:
    c_modified = c.copy()
    c_modified[1] = c[1] + price_increase
    
    result_test = linprog(-c_modified, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')
    
    print(f"  Price B = {c_modified[1]:4d}: x* = [{result_test.x[0]:6.1f}, {result_test.x[1]:6.1f}], z* = {-result_test.fun:7.1f}")
    
    # Check if solution changed significantly
    if not np.allclose(result_test.x, result_primal.x, atol=1):
        print(f"\n*** Optimal solution changes at price increase of {price_increase} ***")
        print(f"New optimal solution: x* = {result_test.x}")
        price_change_threshold = price_increase
        break

# v) New TV type C
print("\n" + "=" * 60)
print("v) New TV Type C Analysis:")
print("=" * 60)

print("""
Type C TV specifications:
- Profit: 1350
- Production times: (7, 4, 2) hours for stages I, II, III
""")

# Extended problem with type C
c_extended = np.array([700, 1000, 1350])  # Include type C
A_extended = np.array([
    [3, 5, 7],   # Stage I
    [1, 3, 4],   # Stage II
    [2, 2, 2]    # Stage III
])

# Solve extended problem
result_extended = linprog(-c_extended, A_ub=A_extended, b_ub=b, 
                         bounds=[(0, None), (0, None), (0, None)], method='highs')

print(f"\nSolution with type C included:")
print(f"  x* = {result_extended.x}")
print(f"  (x1=Type A, x2=Type B, x3=Type C)")
print(f"  Optimal value: z* = {-result_extended.fun:.2f}")

# Calculate reduced cost for type C
# Reduced cost = c_j - (shadow prices) @ (constraint coefficients for variable j)
c_C = 1350
a_C = np.array([7, 4, 2])
reduced_cost_C = c_C - shadow_prices @ a_C

print(f"\nReduced cost for Type C (from original problem):")
print(f"  c_C - y* @ a_C = {c_C} - {shadow_prices} @ {a_C}")
print(f"  = {c_C} - {shadow_prices @ a_C:.2f}")
print(f"  = {reduced_cost_C:.2f}")

print(f"\nInterpretation:")
if reduced_cost_C > 0:
    print(f"*** Type C should be PRODUCED (positive reduced cost = {reduced_cost_C:.2f}) ***")
    print(f"The reduced cost indicates the net benefit of producing type C.")
else:
    print(f"*** Type C should NOT be produced (reduced cost = {reduced_cost_C:.2f}) ***")
    print(f"The resources would be better used for types A and B.")

print(f"\nVerification from optimization:")
if result_extended.x[2] > 0.1:
    print(f"*** The optimization confirms: Type C is produced ({result_extended.x[2]:.2f} units) ***")
else:
    print(f"*** The optimization confirms: Type C is not produced ***")

# vi) Quality inspection working hours
print("\n" + "=" * 60)
print("vi) Quality Inspection Hours Needed:")
print("=" * 60)

print("""
Quality inspection times:
- Type A: 0.5 hours
- Type B: 0.75 hours
- Type C: 0.1 hours (6 minutes)
""")

# Use the optimal production amounts to calculate inspection hours
optimal_production = result_extended.x
inspection_times = np.array([0.5, 0.75, 0.1])

total_inspection_hours = optimal_production @ inspection_times

print(f"\nOptimal production amounts: {optimal_production}")
print(f"Inspection times: {inspection_times} hours")
print(f"Total inspection hours needed: {total_inspection_hours:.2f} hours")

print(f"""
To maintain the current optimal production without interference:
*** The company must add {total_inspection_hours:.2f} hours to the quality inspection line ***
""")

print("\n" + "=" * 60)
print("Exercise 3 complete!")
print("=" * 60)
