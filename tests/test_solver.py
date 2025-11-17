import numpy as np
import jax.numpy as jnp
from cupgs.jax_interface import pgs_solve


def test_diagonal_system(small_diagonal_matrix):
    """Test solving a simple diagonal system."""
    (indptr, indices, data), n = small_diagonal_matrix

    # System: diag([1,2,3,...,10]) * x = [1,1,1,...,1]
    # Solution: x = [1, 0.5, 0.333..., ..., 0.1]
    b = np.ones(n, dtype=np.float32)
    expected_solution = 1.0 / np.arange(1, n + 1, dtype=np.float32)

    # Solve with PGS (unconstrained)
    A_jax = (jnp.array(indptr), jnp.array(indices), jnp.array(data))
    b_jax = jnp.array(b)
    lo_jax = jnp.full(n, -1e10, dtype=jnp.float32)
    hi_jax = jnp.full(n, 1e10, dtype=jnp.float32)
    x0_jax = jnp.zeros(n, dtype=jnp.float32)

    solution, info = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_jax,
        max_iterations=100,
        tolerance=1e-6,
        verbose=False,
    )

    # Check convergence
    assert info["status"] == 0, "Solver should converge"
    assert info["iterations"] < 100, "Should converge quickly for diagonal system"

    # Check solution accuracy
    np.testing.assert_allclose(
        np.array(solution),
        expected_solution,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Solution should match expected values",
    )


def test_solver_with_bounds():
    """Test solver with active bound constraints."""
    n = 5
    # Identity matrix
    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    data = np.ones(n, dtype=np.float32)

    # System: I * x = [10, 10, 10, 10, 10]
    # With bounds: 0 <= x <= 5
    # Solution: x = [5, 5, 5, 5, 5] (all at upper bound)
    b = np.full(n, 10.0, dtype=np.float32)
    lo = np.zeros(n, dtype=np.float32)
    hi = np.full(n, 5.0, dtype=np.float32)

    A_jax = (jnp.array(indptr), jnp.array(indices), jnp.array(data))
    b_jax = jnp.array(b)
    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)
    x0_jax = jnp.zeros(n, dtype=jnp.float32)

    solution, info = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_jax,
        max_iterations=100,
        tolerance=1e-6,
        verbose=False,
    )

    # Check that bounds are satisfied
    assert np.all(np.array(solution) >= lo - 1e-5), "Solution should be >= lower bound"
    assert np.all(np.array(solution) <= hi + 1e-5), "Solution should be <= upper bound"

    # Check that solution is at upper bound
    np.testing.assert_allclose(
        np.array(solution),
        hi,
        rtol=1e-4,
        atol=1e-5,
        err_msg="Solution should be at upper bound",
    )


def test_convergence_with_relaxation():
    """Test that SOR relaxation improves convergence."""
    n = 16
    size = n * n

    # Create a simple Poisson-like matrix
    h = 1.0 / (n + 1)
    h2_inv = 1.0 / (h * h)

    row_ptr = [0]
    col_indices = []
    values = []

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            col_indices.append(idx)
            values.append(4.0 * h2_inv)

            if i > 0:
                col_indices.append(idx - n)
                values.append(-h2_inv)
            if i < n - 1:
                col_indices.append(idx + n)
                values.append(-h2_inv)
            if j > 0:
                col_indices.append(idx - 1)
                values.append(-h2_inv)
            if j < n - 1:
                col_indices.append(idx + 1)
                values.append(-h2_inv)

            row_ptr.append(len(col_indices))

    A_csr = (
        np.array(row_ptr, dtype=np.int32),
        np.array(col_indices, dtype=np.int32),
        np.array(values, dtype=np.float32),
    )

    b = np.ones(size, dtype=np.float32)
    lo = np.full(size, -1e10, dtype=np.float32)
    hi = np.full(size, 1e10, dtype=np.float32)
    x0 = np.zeros(size, dtype=np.float32)

    A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
    b_jax = jnp.array(b)
    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)
    x0_jax = jnp.array(x0)

    # Solve without relaxation
    _, info_no_relax = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_jax,
        max_iterations=500,
        tolerance=1e-5,
        relaxation=1.0,
        verbose=False,
    )

    # Solve with relaxation
    _, info_relax = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_jax,
        max_iterations=500,
        tolerance=1e-5,
        relaxation=1.3,
        verbose=False,
    )

    # Relaxation should reduce iteration count
    assert (
        info_relax["iterations"] < info_no_relax["iterations"]
    ), "SOR should converge faster than standard PGS"


def test_initial_guess():
    """Test that a good initial guess reduces iterations."""
    n = 10
    # Diagonal system
    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    data = np.full(n, 2.0, dtype=np.float32)

    b = np.ones(n, dtype=np.float32)
    lo = np.full(n, -1e10, dtype=np.float32)
    hi = np.full(n, 1e10, dtype=np.float32)

    A_jax = (jnp.array(indptr), jnp.array(indices), jnp.array(data))
    b_jax = jnp.array(b)
    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)

    # Solve from zero initial guess
    x0_zero = jnp.zeros(n, dtype=jnp.float32)
    _, info_zero = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_zero,
        max_iterations=100,
        tolerance=1e-6,
        verbose=False,
    )

    # Solve from good initial guess (close to solution)
    x0_good = jnp.full(n, 0.4, dtype=jnp.float32)  # True solution is 0.5
    _, info_good = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_good,
        max_iterations=100,
        tolerance=1e-6,
        verbose=False,
    )

    # Good initial guess should converge faster
    assert (
        info_good["iterations"] < info_zero["iterations"]
    ), "Good initial guess should reduce iterations"


def test_residual_decreases(small_diagonal_matrix):
    """Test that residual actually decreases during iteration."""
    (indptr, indices, data), n = small_diagonal_matrix

    b = np.ones(n, dtype=np.float32)
    lo = np.full(n, -1e10, dtype=np.float32)
    hi = np.full(n, 1e10, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)

    A_jax = (jnp.array(indptr), jnp.array(indices), jnp.array(data))
    b_jax = jnp.array(b)
    lo_jax = jnp.array(lo)
    hi_jax = jnp.array(hi)
    x0_jax = jnp.array(x0)

    solution, info = pgs_solve(
        A_jax,
        b_jax,
        lo_jax,
        hi_jax,
        x0_jax,
        max_iterations=100,
        tolerance=1e-6,
        verbose=False,
    )

    # Final residual should be small
    assert info["residual"] < 1e-5, "Final residual should be small"
    assert info["status"] == 0, "Solver should converge"
