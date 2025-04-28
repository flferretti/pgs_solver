import numpy as np
import scipy.sparse as sparse
import time
import argparse
import jax
import jax.numpy as jnp
from pgs_solver import PGSSolver, PGSSolverConfig
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve


def generate_test_problem(n, density=0.01, condition=100):
    """Generate a random sparse test problem."""
    np.random.seed(42)  # For reproducibility

    # Generate random sparse matrix
    A_rand = sparse.random(n, n, density=density, format="csr", dtype=np.float32)

    # Make diagonally dominant for convergence
    diag = np.abs(A_rand).sum(axis=1).A.flatten() * 1.5
    A = A_rand + sparse.diags(diag, format="csr")

    # Generate random solution
    x_true = np.random.rand(n).astype(np.float32)

    # Generate right-hand side
    b = A @ x_true

    # Generate random solution
    x_true = np.random.rand(n).astype(np.float32)

    # Generate right-hand side
    b = A @ x_true

    # Generate bounds that include the true solution
    margin = 0.2
    lo = x_true - margin - 0.1 * np.random.rand(n).astype(np.float32)
    hi = x_true + margin + 0.1 * np.random.rand(n).astype(np.float32)

    return A, b, lo, hi, x_true


def benchmark_single_gpu(n_values, density=0.01):
    """Benchmark the solver on a single GPU with varying problem sizes."""
    results = []

    for n in n_values:
        print(f"Testing problem size n={n}")

        # Generate test problem
        A, b, lo, hi, x_true = generate_test_problem(n, density)

        # Convert to JAX arrays
        A_jax = jnp.array(A.toarray())  # For small problems, dense is fine
        b_jax = jnp.array(b)
        lo_jax = jnp.array(lo)
        hi_jax = jnp.array(hi)

        # Time the solution
        start_time = time.time()

        # Create config for PGS solver
        config = PGSSolverConfig(max_iterations=1000, tolerance=1e-6)

        # Create the solver instance
        x, info = PGSSolver(config=config).solve(
            A_blocks=A_jax, b=b_jax, x=x_true, lo=lo_jax, hi=hi_jax
        )

        end_time = time.time()

        # Convert solution back to numpy for error calculation
        x_np = np.array(x)

        # Calculate error
        error = np.linalg.norm(x_np - x_true) / np.linalg.norm(x_true)

        # Record results
        results.append(
            {
                "n": n,
                "time": end_time - start_time,
                "iterations": info["iterations"],
                "residual": info["residual"],
                "error": error,
            }
        )

        print(
            f"  Solved in {info['iterations']} iterations, {end_time - start_time:.4f} seconds"
        )
        print(f"  Residual: {info['residual']:.6e}, Relative error: {error:.6e}")

    return results


def benchmark_multi_gpu(n, densities, num_gpus_list):
    """Benchmark the solver with varying numbers of GPUs."""
    results = []

    for density in densities:
        for num_gpus in num_gpus_list:
            if num_gpus > jax.device_count():
                print(f"Skipping {num_gpus} GPUs (only {jax.device_count()} available)")
                continue

            print(f"Testing with {num_gpus} GPUs, density={density}")

            # Generate test problem
            A, b, lo, hi, x_true = generate_test_problem(n, density)

            # For multi-GPU, split the matrix into blocks
            # In a real implementation, you'd use a more sophisticated partitioning
            A_blocks = []
            block_size = n // num_gpus
            for i in range(num_gpus):
                start_row = i * block_size
                end_row = (i + 1) * block_size if i < num_gpus - 1 else n
                A_block = A[start_row:end_row, :]
                A_blocks.append(jnp.array(A_block.toarray()))

            b_jax = jnp.array(b)
            lo_jax = jnp.array(lo)
            hi_jax = jnp.array(hi)

            # Create config for PGS solver
            config = PGSSolverConfig(max_iterations=1000, tolerance=1e-6)

            # Create the solver instance
            solver = PGSSolver(config=config)

            # Time the solution
            start_time = time.time()

            x, info = solver.solve(
                A_blocks,
                b_jax,
                x_true,
                lo_jax,
                hi_jax,
            )

            end_time = time.time()

            # Convert solution back to numpy for error calculation
            x_np = np.array(x)

            # Calculate error
            error = np.linalg.norm(x_np - x_true) / np.linalg.norm(x_true)

            # Record results
            results.append(
                {
                    "density": density,
                    "num_gpus": num_gpus,
                    "time": end_time - start_time,
                    "iterations": info["iterations"],
                    "residual": info["residual"],
                    "error": error,
                }
            )

            print(
                f"  Solved in {info['iterations']} iterations, {end_time - start_time:.4f} seconds"
            )
            print(f"  Residual: {info['residual']:.6e}, Relative error: {error:.6e}")

    return results


def compare_with_scipy(n_values, density=0.01):
    """Compare our solver with SciPy's direct solver."""
    results = []

    for n in n_values:
        print(f"Comparing solvers for problem size n={n}")

        # Generate test problem
        A, b, lo, hi, x_true = generate_test_problem(n, density)

        # Convert to JAX arrays for our solver
        A_jax = jnp.array(A.toarray())
        b_jax = jnp.array(b)
        lo_jax = jnp.array(lo)
        hi_jax = jnp.array(hi)

        # Create config for PGS solver
        config = PGSSolverConfig(
            max_iterations=1000,
            tolerance=1e-6,
            relaxation=1.3,
            verbose=False,  # SOR factor
        )

        # Create the solver instance
        solver = PGSSolver(config=config)

        # Time our PGS solver
        start_time = time.time()

        x_pgs, info_pgs = solver.solve(A_jax, x_true, b_jax, lo_jax, hi_jax)

        end_time = time.time()

        pgs_time = end_time - start_time

        # Time SciPy's direct solver (ignoring bounds)
        start_time = time.time()
        x_scipy = spsolve(A, b)
        end_time = time.time()
        scipy_time = end_time - start_time

        # Project SciPy solution to bounds for fair comparison
        x_scipy = np.minimum(np.maximum(x_scipy, lo), hi)

        # Calculate errors
        pgs_error = np.linalg.norm(np.array(x_pgs) - x_true) / np.linalg.norm(x_true)
        scipy_error = np.linalg.norm(x_scipy - x_true) / np.linalg.norm(x_true)

        results.append(
            {
                "n": n,
                "pgs_time": pgs_time,
                "scipy_time": scipy_time,
                "pgs_iterations": info_pgs["iterations"],
                "pgs_error": pgs_error,
                "scipy_error": scipy_error,
            }
        )

        print(
            f"  PGS: {pgs_time:.4f}s, {info_pgs['iterations']} iterations, error: {pgs_error:.6e}"
        )
        print(f"  SciPy: {scipy_time:.4f}s, error: {scipy_error:.6e}")

    return results


def plot_results(single_gpu_results, multi_gpu_results, comparison_results):
    """Plot the benchmark results."""
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Single GPU scaling
    n_values = [res["n"] for res in single_gpu_results]
    times = [res["time"] for res in single_gpu_results]
    iterations = [res["iterations"] for res in single_gpu_results]

    ax1 = axes[0, 0]
    ax1.plot(n_values, times, "o-", label="Solve time")
    ax1.set_xlabel("Problem size (n)")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Single GPU Performance Scaling")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(n_values, iterations, "s-", color="red", label="Iterations")
    ax1_twin.set_ylabel("Iterations")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Multi-GPU scaling
    if multi_gpu_results:
        # Group by density
        densities = sorted(set(res["density"] for res in multi_gpu_results))

        ax2 = axes[0, 1]

        for density in densities:
            density_results = [
                res for res in multi_gpu_results if res["density"] == density
            ]
            gpu_counts = [res["num_gpus"] for res in density_results]
            times = [res["time"] for res in density_results]

            ax2.plot(gpu_counts, times, "o-", label=f"Density={density}")

        ax2.set_xlabel("Number of GPUs")
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Multi-GPU Scaling")
        ax2.grid(True)
        ax2.legend()

    # Comparison with SciPy
    if comparison_results:
        n_values = [res["n"] for res in comparison_results]
        pgs_times = [res["pgs_time"] for res in comparison_results]
        scipy_times = [res["scipy_time"] for res in comparison_results]

        ax3 = axes[1, 0]
        ax3.plot(n_values, pgs_times, "o-", label="PGS Solver")
        ax3.plot(n_values, scipy_times, "s-", label="SciPy Direct")
        ax3.set_xlabel("Problem size (n)")
        ax3.set_ylabel("Time (s)")
        ax3.set_title("Performance Comparison with SciPy")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.grid(True)
        ax3.legend()

        pgs_errors = [res["pgs_error"] for res in comparison_results]
        scipy_errors = [res["scipy_error"] for res in comparison_results]

        ax4 = axes[1, 1]
        ax4.plot(n_values, pgs_errors, "o-", label="PGS Solver")
        ax4.plot(n_values, scipy_errors, "s-", label="SciPy Direct")
        ax4.set_xlabel("Problem size (n)")
        ax4.set_ylabel("Relative Error")
        ax4.set_title("Accuracy Comparison")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.grid(True)
        ax4.legend()

    plt.tight_layout()
    plt.savefig("pgs_benchmark_results.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark the CUDA PGS solver")
    parser.add_argument(
        "--single-gpu", action="store_true", help="Run single GPU benchmark"
    )
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Run multi-GPU benchmark"
    )
    parser.add_argument("--compare", action="store_true", help="Compare with SciPy")
    args = parser.parse_args()

    # Default to running all benchmarks if none specified
    run_all = not (args.single_gpu or args.multi_gpu or args.compare)

    single_gpu_results = []
    multi_gpu_results = []
    comparison_results = []

    # Single GPU benchmark
    if args.single_gpu or run_all:
        print("Running single GPU benchmark...")
        n_values = [1000, 2000, 5000, 10000, 20000]
        single_gpu_results = benchmark_single_gpu(n_values)

    # Multi-GPU benchmark
    if args.multi_gpu or run_all:
        print("Running multi-GPU benchmark...")
        n = 20000  # Fixed problem size
        densities = [0.001, 0.01, 0.05]
        num_gpus_list = list(range(1, min(9, jax.device_count() + 1)))
        multi_gpu_results = benchmark_multi_gpu(n, densities, num_gpus_list)

    # Comparison with SciPy
    if args.compare or run_all:
        print("Running comparison with SciPy...")
        n_values = [1000, 2000, 5000, 10000]
        comparison_results = compare_with_scipy(n_values)

    # Plot results
    plot_results(single_gpu_results, multi_gpu_results, comparison_results)


if __name__ == "__main__":
    main()
