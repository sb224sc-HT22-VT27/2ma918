"""
Exercise 2: Large LP-problems
Testing the performance of simplex vs HiGHS method on large LP problems.
"""

import numpy as np
from numpy.random import rand, randn
from scipy.optimize import linprog
from time import perf_counter
import warnings
import matplotlib.pyplot as plt

# Suppress deprecation warnings for simplex
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Number of trials for averaging timing results
NUM_TRIALS = 5

print("=" * 60)
print("Exercise 2: Large LP-problems")
print("=" * 60)

# i) Why use matrix and vector notation?
print("\ni) Why use matrix and vector notation for LP problems?")
print(
    """
Matrix and vector notation provides several advantages:
1. Compact representation: Instead of writing out hundreds or thousands
   of individual inequality equations, we can represent them as Ax ≤ b.
2. Computational efficiency: Matrix operations are highly optimized in
   libraries like NumPy, making computation much faster.
3. Scalability: The same algorithm can handle problems of any size
   without modification.
4. Mathematical clarity: Makes the structure of the problem clear and
   facilitates theoretical analysis.
5. Implementation simplicity: Reduces code complexity and potential
   for errors when dealing with large systems.
"""
)

# ii) Example timing with simplex method
print("\nii) Testing timing for a sample LP problem:")

n, m = 10, 10
A = np.concatenate([rand(m, n) + 1, np.eye(m)], axis=-1)
b = np.ones(m)
c = np.concatenate([randn(n), np.zeros(m)])

# Time the optimization
start_time = perf_counter()
result = linprog(-c, A_eq=A, b_eq=b, method="simplex", options={"maxiter": 5000})
elapsed_time = 1000 * (perf_counter() - start_time)

print(f"Problem size: m = n = {n}")
print(f"Elapsed time: {elapsed_time:.2f} ms")
print(f"Optimization successful: {result.success}")

# iii) Find problem size where simplex average time exceeds 1 second
print("\niii) Finding problem size where simplex exceeds 1 second:")

# Benchmarking parameters for simplex method
SIMPLEX_START_SIZE = 10
SIMPLEX_MAX_SIZE = 200
SIMPLEX_STEP_SIZE = 10

simplex_times = []
simplex_sizes = []

# Test different problem sizes
for size in range(SIMPLEX_START_SIZE, SIMPLEX_MAX_SIZE, SIMPLEX_STEP_SIZE):
    times = []
    for trial in range(NUM_TRIALS):
        n, m = size, size
        A = np.concatenate([rand(m, n) + 1, np.eye(m)], axis=-1)
        b = np.ones(m)
        c = np.concatenate([randn(n), np.zeros(m)])

        start_time = perf_counter()
        result = linprog(
            -c, A_eq=A, b_eq=b, method="simplex", options={"maxiter": 5000}
        )
        elapsed = 1000 * (perf_counter() - start_time)
        times.append(elapsed)

    avg_time = np.mean(times)
    simplex_times.append(avg_time)
    simplex_sizes.append(size)

    print(f"  m = n = {size:3d}: Average time = {avg_time:7.2f} ms")

    if avg_time > 1000:
        print(f"\n*** Simplex exceeds 1 second at m = n = {size} ***")
        simplex_threshold = size
        break
else:
    simplex_threshold = simplex_sizes[-1]

# iv) Test with HiGHS method
print("\niv) Testing with HiGHS method:")

# Benchmarking parameters for HiGHS method
HIGHS_START_SIZE = 100
HIGHS_MAX_SIZE = 3000
HIGHS_STEP_SIZE = 100

highs_times = []
highs_sizes = []

# Test different problem sizes with HiGHS
for size in range(HIGHS_START_SIZE, HIGHS_MAX_SIZE, HIGHS_STEP_SIZE):
    times = []
    for trial in range(NUM_TRIALS):
        n, m = size, size
        A = np.concatenate([rand(m, n) + 1, np.eye(m)], axis=-1)
        b = np.ones(m)
        c = np.concatenate([randn(n), np.zeros(m)])

        start_time = perf_counter()
        result = linprog(-c, A_eq=A, b_eq=b, method="highs")
        elapsed = 1000 * (perf_counter() - start_time)
        times.append(elapsed)

    avg_time = np.mean(times)
    highs_times.append(avg_time)
    highs_sizes.append(size)

    print(f"  m = n = {size:4d}: Average time = {avg_time:7.2f} ms")

    if avg_time > 1000:
        print(f"\n*** HiGHS exceeds 1 second at m = n = {size} ***")
        highs_threshold = size
        break
else:
    highs_threshold = highs_sizes[-1]

# Create comparison plot
print("\nCreating comparison plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Simplex method
ax1.plot(simplex_sizes, simplex_times, "o-", color="blue", linewidth=2, markersize=6)
ax1.axhline(y=1000, color="r", linestyle="--", label="1 second threshold")
ax1.set_xlabel("Problem Size (m = n)")
ax1.set_ylabel("Average Time (ms)")
ax1.set_title("Simplex Method Performance")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: HiGHS method
ax2.plot(highs_sizes, highs_times, "o-", color="green", linewidth=2, markersize=6)
ax2.axhline(y=1000, color="r", linestyle="--", label="1 second threshold")
ax2.set_xlabel("Problem Size (m = n)")
ax2.set_ylabel("Average Time (ms)")
ax2.set_title("HiGHS Method Performance")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig("img/ex2_performance_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: img/ex2_performance_comparison.png")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Simplex method exceeds 1 second at: m = n = {simplex_threshold}")
print(f"HiGHS method exceeds 1 second at:  m = n = {highs_threshold}")
print(f"Performance improvement: ~{highs_threshold / simplex_threshold:.1f}x faster")
print("=" * 60)
print("Exercise 2 complete!")
print("=" * 60)
