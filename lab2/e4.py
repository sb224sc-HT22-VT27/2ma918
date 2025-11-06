"""
Exercise 4: Constrained Optimization
Solve constrained optimization using penalty function method
"""

import numpy as np
from scipy.optimize import minimize

def f(x):
    """Objective function: f(x,y) = (2-x-y)^2 + y^4"""
    return (2 - x[0] - x[1])**2 + x[1]**4

def g1(x):
    """Constraint 1: x^2 + y^2 - 4 <= 0"""
    return x[0]**2 + x[1]**2 - 4

def g2(x):
    """Constraint 2: 4x + 5y - 25 <= 0"""
    return 4*x[0] + 5*x[1] - 25

def alpha(x):
    """Penalty function: sum of (max(0, gi(x)))^2"""
    return max(0, g1(x))**2 + max(0, g2(x))**2

def F(x, mu):
    """Augmented objective: F(x) = f(x) + mu * alpha(x)"""
    return f(x) + mu * alpha(x)

def grad_f(x):
    """Gradient of f"""
    df_dx = -2*(2 - x[0] - x[1])
    df_dy = -2*(2 - x[0] - x[1]) + 4*x[1]**3
    return np.array([df_dx, df_dy])

def is_feasible(x, tol=1e-6):
    """Check if point is feasible"""
    return g1(x) <= tol and g2(x) <= tol

def find_initial_feasible_point():
    """Find a feasible starting point"""
    # Try several points
    candidates = [
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([1.5, 1.0]),
        np.array([1.0, 1.5]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    
    for x in candidates:
        if is_feasible(x):
            return x
    
    # If no candidate works, use optimization to find feasible point
    # Minimize sum of constraint violations
    def constraint_violation(x):
        return max(0, g1(x))**2 + max(0, g2(x))**2
    
    x0 = np.array([0.0, 0.0])
    result = minimize(constraint_violation, x0, method='BFGS')
    
    if is_feasible(result.x):
        return result.x
    else:
        # Return origin as fallback (even if infeasible)
        return np.array([0.0, 0.0])

def penalty_method(x0, mu0=0.1, beta=10, tol=1e-4, max_iter=100):
    """
    Penalty method for constrained optimization
    
    Parameters:
    - x0: initial point
    - mu0: initial penalty parameter
    - beta: penalty parameter multiplier
    - tol: tolerance for penalty function value
    - max_iter: maximum number of iterations
    """
    x_k = x0.copy()
    mu_k = mu0
    
    iterations = 0
    results = []
    
    print("=" * 80)
    print("Penalty Method Iterations")
    print("=" * 80)
    print(f"{'Iter':<6} {'mu':<12} {'x':<25} {'f(x)':<12} {'alpha(x)':<12} {'Feasible':<10}")
    print("-" * 80)
    
    while iterations < max_iter:
        # Solve unconstrained problem
        result = minimize(F, x_k, args=(mu_k,), method='BFGS')
        x_k = result.x
        
        alpha_k = alpha(x_k)
        f_k = f(x_k)
        feasible = is_feasible(x_k)
        
        print(f"{iterations:<6} {mu_k:<12.2f} ({x_k[0]:6.4f}, {x_k[1]:6.4f})    {f_k:<12.6f} {alpha_k:<12.6e} {str(feasible):<10}")
        
        results.append({
            'iteration': iterations,
            'mu': mu_k,
            'x': x_k.copy(),
            'f': f_k,
            'alpha': alpha_k,
            'g1': g1(x_k),
            'g2': g2(x_k),
            'feasible': feasible
        })
        
        # Check convergence
        if alpha_k < tol:
            print("-" * 80)
            print(f"Converged: alpha(x) = {alpha_k:.6e} < {tol:.6e}")
            break
        
        # Update penalty parameter
        mu_k = beta * mu_k
        iterations += 1
    
    if iterations >= max_iter:
        print("-" * 80)
        print(f"Warning: Maximum iterations ({max_iter}) reached")
    
    return x_k, results

if __name__ == "__main__":
    print("=" * 80)
    print("Exercise 4: Constrained Optimization with Penalty Method")
    print("=" * 80)
    
    # Task 1: Find feasible starting point
    print("\nTask 1: Finding Feasible Starting Point")
    print("-" * 80)
    
    x0 = find_initial_feasible_point()
    print(f"Initial point: x0 = ({x0[0]:.6f}, {x0[1]:.6f})")
    print(f"f(x0) = {f(x0):.6f}")
    print(f"g1(x0) = x^2 + y^2 - 4 = {g1(x0):.6f} {'<= 0 ✓' if g1(x0) <= 0 else '> 0 ✗'}")
    print(f"g2(x0) = 4x + 5y - 25 = {g2(x0):.6f} {'<= 0 ✓' if g2(x0) <= 0 else '> 0 ✗'}")
    print(f"Feasible: {is_feasible(x0)}")
    
    # Task 2: Penalty method
    print("\n" + "=" * 80)
    print("Task 2: Solving with Penalty Method")
    print("=" * 80)
    
    mu0 = 0.1
    beta = 10
    
    x_opt, results = penalty_method(x0, mu0=mu0, beta=beta, tol=1e-4)
    
    # Final results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Optimal point (estimated): x* = ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"Optimal function value: f(x*) = {f(x_opt):.6f}")
    print(f"Number of iterations: {len(results)}")
    print(f"\nConstraint values at optimal:")
    print(f"  g1(x*) = x^2 + y^2 - 4 = {g1(x_opt):.6f}")
    print(f"  g2(x*) = 4x + 5y - 25 = {g2(x_opt):.6f}")
    print(f"  Penalty function: alpha(x*) = {alpha(x_opt):.6e}")
    print(f"  Feasible: {is_feasible(x_opt)}")
    
    # Gradient at optimal
    grad = grad_f(x_opt)
    print(f"\nGradient at optimal:")
    print(f"  grad_f(x*) = ({grad[0]:.6f}, {grad[1]:.6f})")
    print(f"  ||grad_f(x*)|| = {np.linalg.norm(grad):.6e}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF ITERATIONS")
    print("=" * 80)
    for res in results:
        status = "CONVERGED" if res['iteration'] == len(results) - 1 and res['alpha'] < 1e-4 else ""
        print(f"Iteration {res['iteration']}: mu={res['mu']:.2e}, "
              f"f(x)={res['f']:.6f}, alpha(x)={res['alpha']:.6e} {status}")
