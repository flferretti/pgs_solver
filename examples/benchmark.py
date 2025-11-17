import numpy as np
import time
import argparse
import jax.numpy as jnp
from cupgs.jax_interface import pgs_solve
import matplotlib.pyplot as plt


def create_poisson_matrix_2d(n):
    """
    Create the system matrix for the 2D Poisson equation on an nxn grid.
    Returns a CSR sparse matrix representing the discrete Laplacian.
    """
    size = n * n
    h = 1.0 / (n + 1)
    h2_inv = 1.0 / (h * h)

    # Build CSR matrix components
    row_ptr = [0]
    col_indices = []
    values = []

    for i in range(n):
        for j in range(n):
            idx = i * n + j

            # Diagonal element
            col_indices.append(idx)
            values.append(4.0 * h2_inv)

            # Off-diagonal elements (4-point stencil)
            if i > 0:  # North neighbor
                col_indices.append(idx - n)
                values.append(-h2_inv)
            if i < n - 1:  # South neighbor
                col_indices.append(idx + n)
                values.append(-h2_inv)
            if j > 0:  # West neighbor
                col_indices.append(idx - 1)
                values.append(-h2_inv)
            if j < n - 1:  # East neighbor
                col_indices.append(idx + 1)
                values.append(-h2_inv)

            row_ptr.append(len(col_indices))

    # Convert to numpy arrays with proper dtypes
    row_ptr = np.array(row_ptr, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    values = np.array(values, dtype=np.float32)

    return (row_ptr, col_indices, values), size


def solve_poisson_example(n=32, verbose=True):
    """
    Solve a 2D Poisson equation: ∇²u = 1 with zero boundary conditions.

    Returns:
        solution, solve_time, iterations, residual
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Solving 2D Poisson Equation on {n}x{n} grid")
        print(f"{'='*60}")

    # Create system matrix
    A_csr, size = create_poisson_matrix_2d(n)

    # Right-hand side: f = 1 everywhere
    b = np.ones(size, dtype=np.float32)

    # Bounds: unconstrained (use very large bounds)
    lo = np.full(size, -1e10, dtype=np.float32)
    hi = np.full(size, 1e10, dtype=np.float32)

    # Better initial guess based on expected solution magnitude
    x0 = np.full(size, 0.01, dtype=np.float32)

    # Calculate optimal SOR parameter for 2D Poisson equation
    h = 1.0 / (n + 1)
    rho_jacobi = np.cos(np.pi * h)  # Spectral radius of Jacobi iteration
    omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2))
    # Cap at 1.95 for numerical stability
    omega = min(omega_opt, 1.95)

    if verbose:
        print(f"Grid spacing h = {h:.4f}")
        print(f"Optimal relaxation ω = {omega:.4f}")  # Convert to JAX arrays
    A_jax = (
        jnp.array(A_csr[0]),
        jnp.array(A_csr[1]),
        jnp.array(A_csr[2]),
    )
    b_jax = jnp.array(b)
    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)
    x0_jax = jnp.array(x0)

    # Solve with optimal parameters
    start_time = time.time()
    solution, info = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_jax,
        max_iterations=5000,  # Increased for better convergence
        tolerance=1e-6,
        relaxation=omega,  # Use optimal SOR parameter
        verbose=verbose,
    )
    solve_time = time.time() - start_time

    if verbose:
        print(f"\nSolution computed in {solve_time:.4f} seconds")
        print(f"Iterations: {info['iterations']}")
        print(f"Final residual: {info['residual']:.6e}")
        print(f"Status: {info['status']}")

    return np.array(solution), solve_time, info["iterations"], info["residual"]


def benchmark_scaling(grid_sizes=[16, 32, 48, 64, 96], verbose=True):
    """
    Benchmark solver performance scaling with problem size.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Performance Scaling Benchmark")
        print(f"{'='*60}")

    results = []

    for n in grid_sizes:
        _, solve_time, iterations, residual = solve_poisson_example(n, verbose=False)
        size = n * n

        results.append(
            {
                "grid_size": n,
                "dofs": size,
                "time": solve_time,
                "iterations": iterations,
                "residual": residual,
                "dofs_per_sec": size / solve_time if solve_time > 0 else 0,
            }
        )

        if verbose:
            print(
                f"n={n:3d} ({size:5d} DOFs): {solve_time:.4f}s, "
                f"{iterations:4d} iters, {size/solve_time:.0f} DOFs/s"
            )

    return results


def benchmark_convergence(
    n=64, relaxations=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5], verbose=True
):
    """
    Analyze convergence with different relaxation parameters.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Convergence Analysis (n={n})")
        print(f"{'='*60}")

    # Create system matrix once
    A_csr, size = create_poisson_matrix_2d(n)
    b = np.ones(size, dtype=np.float32)
    lo = np.full(size, -1e10, dtype=np.float32)
    hi = np.full(size, 1e10, dtype=np.float32)
    x0 = np.full(size, 0.01, dtype=np.float32)  # Better initial guess

    # Calculate optimal omega
    h = 1.0 / (n + 1)
    rho_jacobi = np.cos(np.pi * h)
    omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2))
    omega_opt = min(omega_opt, 1.95)

    if verbose:
        print(f"Grid: {n}x{n}, h = {h:.4f}")
        print(f"Theoretical optimal ω = {omega_opt:.4f}\n")

    # Convert to JAX arrays
    A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
    b_jax = jnp.array(b)
    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)
    x0_jax = jnp.array(x0)

    results = []

    for omega in relaxations:
        start_time = time.time()
        solution, info = pgs_solve(
            A_jax,
            b_jax,
            lo_jax,
            hi_jax,
            x0_jax,
            max_iterations=5000,  # Increased for better convergence
            tolerance=1e-6,
            relaxation=omega,
            verbose=False,
        )
        solve_time = time.time() - start_time

        results.append(
            {
                "relaxation": omega,
                "iterations": info["iterations"],
                "time": solve_time,
                "residual": info["residual"],
            }
        )

        if verbose:
            marker = " ⭐" if abs(omega - omega_opt) < 0.05 else ""
            print(
                f"ω={omega:.2f}: {info['iterations']:4d} iters, "
                f"{solve_time:.4f}s, residual={info['residual']:.2e}{marker}"
            )

    return results


def plot_scaling_results(results, save_path="pgs_scaling.png"):
    """Plot performance scaling results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dofs = [r["dofs"] for r in results]
    times = [r["time"] for r in results]
    iterations = [r["iterations"] for r in results]

    # Time vs problem size
    ax1.plot(dofs, times, "o-", linewidth=2, markersize=8)
    ax1.set_xlabel("Problem Size (DOFs)", fontsize=12)
    ax1.set_ylabel("Time (s)", fontsize=12)
    ax1.set_title("Solver Time vs Problem Size", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Iterations vs problem size
    ax2.plot(dofs, iterations, "s-", linewidth=2, markersize=8, color="orange")
    ax2.set_xlabel("Problem Size (DOFs)", fontsize=12)
    ax2.set_ylabel("Iterations", fontsize=12)
    ax2.set_title("Iterations vs Problem Size", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nScaling plot saved to {save_path}")
    plt.close()


def plot_convergence_results(results, save_path="pgs_convergence.png"):
    """Plot convergence analysis results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    relaxations = [r["relaxation"] for r in results]
    iterations = [r["iterations"] for r in results]
    times = [r["time"] for r in results]

    # Iterations vs relaxation
    ax1.plot(relaxations, iterations, "o-", linewidth=2, markersize=8, color="green")
    ax1.set_xlabel("Relaxation Parameter (ω)", fontsize=12)
    ax1.set_ylabel("Iterations to Convergence", fontsize=12)
    ax1.set_title("Convergence vs Relaxation", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=1.0, color="r", linestyle="--", alpha=0.5, label="Standard PGS")
    ax1.legend()

    # Time vs relaxation
    ax2.plot(relaxations, times, "s-", linewidth=2, markersize=8, color="purple")
    ax2.set_xlabel("Relaxation Parameter (ω)", fontsize=12)
    ax2.set_ylabel("Time (s)", fontsize=12)
    ax2.set_title("Solve Time vs Relaxation", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=1.0, color="r", linestyle="--", alpha=0.5, label="Standard PGS")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Convergence plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark and demonstrate the CUDA PGS solver"
    )
    parser.add_argument(
        "--example", action="store_true", help="Run a simple Poisson equation example"
    )
    parser.add_argument(
        "--scaling", action="store_true", help="Run performance scaling benchmark"
    )
    parser.add_argument(
        "--convergence", action="store_true", help="Run convergence analysis"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks (default if no flags specified)",
    )

    args = parser.parse_args()

    # Default to running all if no specific flag is set
    run_all = args.all or not (args.example or args.scaling or args.convergence)

    try:
        # Simple example
        if args.example or run_all:
            solve_poisson_example(n=32, verbose=True)

        # Scaling benchmark
        if args.scaling or run_all:
            scaling_results = benchmark_scaling(
                grid_sizes=[16, 24, 32, 48, 64], verbose=True
            )
            plot_scaling_results(scaling_results)

        # Convergence analysis
        if args.convergence or run_all:
            convergence_results = benchmark_convergence(
                n=48, relaxations=[1.0, 1.2, 1.4, 1.6, 1.7, 1.8, 1.9], verbose=True
            )
            plot_convergence_results(convergence_results)

        print(f"\n{'='*60}")
        print("Benchmark completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
