import jax
import jax.numpy as jnp
import numpy as np
import time
from pgs_solver.jax_interface import pgs_solve, pgs_solve_jittable
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def create_poisson_system(n):
    """Create a 2D Poisson problem on an n x n grid."""
    # Create the Laplacian operator
    size = n * n
    diagonals = np.ones(size)
    offsets = np.ones(size - 1)

    # Main diagonal is 4
    diagonals *= 4

    # Set up the off-diagonals (horizontal neighbors)
    for i in range(size - 1):
        if (i + 1) % n == 0:
            offsets[i] = 0  # No connection between adjacent grid rows

    # Create sparse matrix in CSR format
    A = sparse.diags(
        [diagonals, -offsets, -offsets, -np.ones(size - n), -np.ones(size - n)],
        [0, 1, -1, n, -n],
        format="csr",
        dtype=np.float32,
    )

    return A


def solve_poisson_equation(n, f_func, jit=False):
    """
    Solve the Poisson equation ∇²u = f on a unit square with zero boundary conditions.

    Parameters:
    -----------
    n : int
        Number of grid points in each dimension
    f_func : callable
        Function to evaluate the right-hand side f(x, y)
    jit : bool
        Whether to use the jittable version of the solver

    Returns:
    --------
    u : array
        Solution on the n x n grid
    """
    # Create system matrix
    A_sparse = create_poisson_system(n)

    # Convert to JAX array for our solver
    A_jax = jnp.array(A_sparse.toarray())

    # Create grid points
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1 - h, n)
    y = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, y)

    # Evaluate the right-hand side function
    f_vals = f_func(X, Y).flatten()

    # We need to scale f by h² for the discretization
    b = h * h * jnp.array(f_vals)

    # Set bounds (unconstrained in this case, but with very large bounds)
    lo = jnp.full_like(b, -1e10)
    hi = jnp.full_like(b, 1e10)

    # Initial guess
    x0 = jnp.zeros_like(b)

    # Timing without JIT
    start_time = time.time()

    if jit:
        # Use the jittable version
        solver_config = {
            "max_iterations": 1000,
            "tolerance": 1e-6,
            "relaxation": 1.3,  # SOR factor
            "gpu_ids": [0],
            "verbose": False,
        }

        # Compile the function (first call will be slower due to compilation)
        solve_fn = jax.jit(
            lambda A, b, lo, hi, x0, cfg: pgs_solve_jittable(A, b, lo, hi, x0, cfg)
        )

        # Solve the system
        u = solve_fn(A_jax, b, lo, hi, x0, solver_config)

    else:
        # Use the standard version
        u, info = pgs_solve(
            A_jax,
            b,
            lo,
            hi,
            x0,
            max_iterations=1000,
            tolerance=1e-6,
            relaxation=1.3,  # SOR factor
            verbose=False,
        )

    end_time = time.time()

    # Reshape the solution to a 2D grid
    u_grid = u.reshape((n, n))

    print(f"Solved {n}x{n} grid in {end_time - start_time:.4f} seconds")
    if not jit:
        print(f"Iterations: {info['iterations']}, Residual: {info['residual']:.6e}")

    return u_grid


def analytical_solution(x, y):
    """Analytical solution for test problem: u(x,y) = sin(πx)sin(πy)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_term(x, y):
    """Source term for the analytical solution"""
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def visualize_solution(n, u_numerical):
    """Visualize the numerical solution and compare with analytical solution"""
    h = 1.0 / (n + 1)
    x = np.linspace(h, 1 - h, n)
    y = np.linspace(h, 1 - h, n)
    X, Y = np.meshgrid(x, y)

    # Compute analytical solution
    u_analytical = analytical_solution(X, Y)

    # Compute error
    error = np.abs(u_numerical - u_analytical)
    max_error = np.max(error)
    l2_error = np.sqrt(np.mean(error**2))

    # Create figure
    fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X, Y, u_numerical, cmap="viridis")
    ax1.set_title("Numerical Solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u(x,y)")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X, Y, u_analytical, cmap="viridis")
    ax2.set_title("Analytical Solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u(x,y)")

    ax3 = fig.add_subplot(133, projection="3d")
    surf = ax3.plot_surface(X, Y, error, cmap="hot")
    ax3.set_title(f"Error (Max: {max_error:.2e}, L2: {l2_error:.2e})")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("Error")
    fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig("poisson_solution.png")
    plt.show()


def compare_jit_performance():
    """Compare performance with and without JIT compilation"""
    grid_sizes = [32, 64, 96, 128, 192, 256]
    times_no_jit = []
    times_jit = []

    for n in grid_sizes:
        print(f"\nTesting grid size {n}x{n}")

        # Without JIT
        print("Without JIT:")
        start_time = time.time()
        solve_poisson_equation(n, source_term, jit=False)
        end_time = time.time()
        times_no_jit.append(end_time - start_time)

        # With JIT
        print("With JIT:")
        start_time = time.time()
        solve_poisson_equation(n, source_term, jit=True)

        # Run again to exclude compilation time
        start_time = time.time()
        solve_poisson_equation(n, source_term, jit=True)
        end_time = time.time()
        times_jit.append(end_time - start_time)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(grid_sizes, times_no_jit, "o-", label="Without JIT")
    plt.plot(grid_sizes, times_jit, "s-", label="With JIT")
    plt.xlabel("Grid Size (n)")
    plt.ylabel("Time (s)")
    plt.title("Performance Comparison: JIT vs No JIT")
    plt.grid(True)
    plt.legend()
    plt.savefig("jit_performance.png")
    plt.show()


def main():
    # Solve a medium-sized problem and visualize
    n = 64  # Grid size
    u_numerical = solve_poisson_equation(n, source_term)
    visualize_solution(n, u_numerical)

    # Compare JIT performance
    compare_jit_performance()


if __name__ == "__main__":
    main()
