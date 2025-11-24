"""
Exercise 2: Unconstrained Optimization I
Implement steepest descent and Newton's method
"""

import numpy as np


# Define the function f(x, y) = (x+1)^2 - xy + 3(y-5)^2
def f(x):
    """Function to minimize: f(x, y) = (x+1)^2 - xy + 3(y-5)^2"""
    return (x[0] + 1) ** 2 - x[0] * x[1] + 3 * (x[1] - 5) ** 2


def grad_f(x):
    """Gradient of f"""
    dx = 2 * (x[0] + 1) - x[1]
    dy = -x[0] + 6 * (x[1] - 5)
    return np.array([dx, dy])


def hessian_f(x):
    """Hessian of f"""
    return np.array([[2, -1], [-1, 6]])


def hessian_inv_f(x):
    """Inverse Hessian of f (constant for this quadratic)"""
    H = hessian_f(x)
    return np.linalg.inv(H)


def find_optimal_analytical():
    """Find optimal point by solving grad_f = 0"""
    # grad_f = [2(x+1) - y, -x + 6(y-5)] = [0, 0]
    # 2x + 2 - y = 0  =>  y = 2x + 2
    # -x + 6y - 30 = 0  =>  6y = x + 30
    # Substitute: 6(2x + 2) = x + 30
    # 12x + 12 = x + 30
    # 11x = 18
    # x = 18/11
    x_opt = 18 / 11
    y_opt = 2 * x_opt + 2

    optimal_point = np.array([x_opt, y_opt])
    optimal_value = f(optimal_point)

    print("=" * 60)
    print("Analytical Solution")
    print("=" * 60)
    print(f"Optimal point: ({x_opt:.6f}, {y_opt:.6f})")
    print(f"Optimal value: {optimal_value:.6f}")
    print(f"Gradient at optimal: {grad_f(optimal_point)}")
    print(f"Hessian:\n{hessian_f(optimal_point)}")

    # Check if Hessian is positive definite
    eigenvalues = np.linalg.eigvals(hessian_f(optimal_point))
    print(f"Hessian eigenvalues: {eigenvalues}")
    print(f"Positive definite: {np.all(eigenvalues > 0)}")

    return optimal_point, optimal_value


def armijo_line_search(x_k, grad_k):
    """
    Armijo's method for step length selection
    """
    t = 1.0
    grad_norm_sq = grad_k @ grad_k

    def w(t_val):
        return x_k - t_val * grad_k

    # Check initial condition
    if f(w(t)) <= f(x_k) - 0.2 * t * grad_norm_sq:
        # Need to increase t
        while f(w(t)) <= f(x_k) - 0.2 * t * grad_norm_sq:
            t_prev = t
            t = 2 * t
            if t > 1e6:  # Prevent infinite loop
                return t_prev
        return t_prev
    else:
        # Need to decrease t
        while f(w(t)) > f(x_k) - 0.2 * t * grad_norm_sq:
            t = t / 2
            if t < 1e-10:  # Prevent infinite loop
                return t
        return t


def steepest_descent(x0, tol=1e-3, max_iter=10000):
    """Steepest descent method with Armijo line search"""
    x_k = x0.copy()
    iterations = 0
    points = [x_k.copy()]

    while True:
        grad_k = grad_f(x_k)
        grad_norm = np.linalg.norm(grad_k)

        if grad_norm < tol:
            break

        if iterations >= max_iter:
            print(f"Warning: Maximum iterations ({max_iter}) reached")
            break

        # Armijo line search
        t_k = armijo_line_search(x_k, grad_k)

        # Update
        x_k = x_k - t_k * grad_k
        points.append(x_k.copy())
        iterations += 1

    return x_k, iterations, points


def newtons_method(x0, tol=1e-3, max_iter=100):
    """Newton's method for optimization"""
    x_k = x0.copy()
    iterations = 0
    points = [x_k.copy()]

    while True:
        grad_k = grad_f(x_k)
        grad_norm = np.linalg.norm(grad_k)

        if grad_norm < tol:
            break

        if iterations >= max_iter:
            print(f"Warning: Maximum iterations ({max_iter}) reached")
            break

        # Newton update
        H_inv = hessian_inv_f(x_k)
        x_k = x_k - H_inv @ grad_k
        points.append(x_k.copy())
        iterations += 1

    return x_k, iterations, points


if __name__ == "__main__":
    # Find analytical solution
    x_optimal, f_optimal = find_optimal_analytical()

    # Starting point
    x0 = np.array([1.0, 1.0])

    print("\n" + "=" * 60)
    print("Steepest Descent Method")
    print("=" * 60)
    x_sd, iter_sd, points_sd = steepest_descent(x0)
    print(f"Starting point: ({x0[0]:.6f}, {x0[1]:.6f})")
    print(f"Final point: ({x_sd[0]:.6f}, {x_sd[1]:.6f})")
    print(f"Number of iterations: {iter_sd}")
    print(f"Final function value: {f(x_sd):.6f}")
    print(f"Optimal function value: {f_optimal:.6f}")
    print(f"Absolute error: {abs(f(x_sd) - f_optimal):.6e}")
    print(f"Final gradient norm: {np.linalg.norm(grad_f(x_sd)):.6e}")

    print("\n" + "=" * 60)
    print("Newton's Method")
    print("=" * 60)
    x_nm, iter_nm, points_nm = newtons_method(x0)
    print(f"Starting point: ({x0[0]:.6f}, {x0[1]:.6f})")
    print(f"Final point: ({x_nm[0]:.6f}, {x_nm[1]:.6f})")
    print(f"Number of iterations: {iter_nm}")
    print(f"Final function value: {f(x_nm):.6f}")
    print(f"Optimal function value: {f_optimal:.6f}")
    print(f"Absolute error: {abs(f(x_nm) - f_optimal):.6e}")
    print(f"Final gradient norm: {np.linalg.norm(grad_f(x_nm)):.6e}")

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"Steepest descent iterations: {iter_sd}")
    print(f"Newton's method iterations: {iter_nm}")
    print("\nNewton's method is much more efficient for this quadratic problem")
    print("because it uses second-order information (Hessian) and converges")
    print("in a single iteration for quadratic functions when starting close enough.")
