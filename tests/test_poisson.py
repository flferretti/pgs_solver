import numpy as np
import jax.numpy as jnp
from cupgs.jax_interface import pgs_solve


def create_poisson_system(n):
    """Create 2D Poisson equation system."""
    size = n * n
    h = 1.0 / (n + 1)
    h2_inv = 1.0 / (h * h)

    row_ptr = [0]
    col_indices = []
    values = []

    for i in range(n):
        for j in range(n):
            idx = i * n + j

            # Diagonal
            col_indices.append(idx)
            values.append(4.0 * h2_inv)

            # Off-diagonals
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

    return (
        np.array(row_ptr, dtype=np.int32),
        np.array(col_indices, dtype=np.int32),
        np.array(values, dtype=np.float32),
    ), size


def test_poisson_small():
    """Test solving a small Poisson equation."""
    n = 8
    A_csr, size = create_poisson_system(n)

    # RHS: f = 1
    b = np.ones(size, dtype=np.float32)
    lo = np.full(size, -1e10, dtype=np.float32)
    hi = np.full(size, 1e10, dtype=np.float32)
    x0 = np.zeros(size, dtype=np.float32)

    A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
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
        max_iterations=500,
        tolerance=1e-5,
        relaxation=1.3,
        verbose=False,
    )

    assert info["status"] == 0, "Should converge"
    assert info["residual"] < 1e-4, "Residual should be small"

    # Solution should be positive everywhere
    assert np.all(np.array(solution) > 0), "Solution should be positive"

    # Solution should be symmetric (due to symmetric problem)
    sol_2d = np.array(solution).reshape((n, n))
    assert np.allclose(
        sol_2d, sol_2d.T, rtol=0.1
    ), "Solution should be roughly symmetric"


def test_poisson_convergence():
    """Test convergence for different grid sizes."""
    grid_sizes = [4, 8, 12]

    for n in grid_sizes:
        A_csr, size = create_poisson_system(n)

        b = np.ones(size, dtype=np.float32)
        lo = np.full(size, -1e10, dtype=np.float32)
        hi = np.full(size, 1e10, dtype=np.float32)
        x0 = np.zeros(size, dtype=np.float32)

        A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
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
            max_iterations=1000,
            tolerance=1e-5,
            relaxation=1.3,
            verbose=False,
        )

        assert info["status"] == 0, f"Should converge for n={n}"
        assert info["residual"] < 1e-4, f"Residual too large for n={n}"


def test_poisson_with_different_rhs():
    """Test Poisson with different right-hand sides."""
    n = 8
    A_csr, size = create_poisson_system(n)

    # Test with different RHS patterns
    rhs_patterns = [
        np.ones(size, dtype=np.float32),  # Constant
        np.random.rand(size).astype(np.float32),  # Random
        np.linspace(0, 1, size, dtype=np.float32),  # Linear
    ]

    for b in rhs_patterns:
        lo = np.full(size, -1e10, dtype=np.float32)
        hi = np.full(size, 1e10, dtype=np.float32)
        x0 = np.zeros(size, dtype=np.float32)

        A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
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
            max_iterations=1000,
            tolerance=1e-5,
            relaxation=1.3,
            verbose=False,
        )

        assert info["status"] == 0, "Should converge for all RHS patterns"


def test_poisson_symmetry():
    """Test that symmetric problems give symmetric solutions."""
    n = 8
    A_csr, size = create_poisson_system(n)

    # Symmetric RHS
    b = np.ones(size, dtype=np.float32)
    lo = np.full(size, -1e10, dtype=np.float32)
    hi = np.full(size, 1e10, dtype=np.float32)
    x0 = np.zeros(size, dtype=np.float32)

    A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
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
        max_iterations=1000,
        tolerance=1e-5,
        relaxation=1.3,
        verbose=False,
    )

    # Reshape to 2D grid
    sol_2d = np.array(solution).reshape((n, n))

    # Check approximate symmetry (diagonal symmetry)
    for i in range(n):
        for j in range(i):
            assert (
                np.abs(sol_2d[i, j] - sol_2d[j, i]) < 0.1
            ), f"Solution not symmetric at ({i},{j})"


def test_poisson_maximum_principle():
    """Test that Poisson solution satisfies discrete maximum principle."""
    n = 8
    A_csr, size = create_poisson_system(n)

    # Positive RHS
    b = np.ones(size, dtype=np.float32)
    lo = np.full(size, -1e10, dtype=np.float32)
    hi = np.full(size, 1e10, dtype=np.float32)
    x0 = np.zeros(size, dtype=np.float32)

    A_jax = (jnp.array(A_csr[0]), jnp.array(A_csr[1]), jnp.array(A_csr[2]))
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
        max_iterations=1000,
        tolerance=1e-5,
        relaxation=1.3,
        verbose=False,
    )

    sol_2d = np.array(solution).reshape((n, n))

    # Maximum should be in the interior (not on boundary which would be 0)
    # Since we have zero boundary conditions (implicit)
    assert np.all(sol_2d > 0), "All interior values should be positive"

    # Check that maximum is not at the "boundary" (edges of our grid)
    max_val = np.max(sol_2d)

    # The maximum should be somewhere in the middle
    max_indices = np.where(sol_2d == max_val)

    # At least one maximum should not be on the edge
    interior_max = False

    for i, j in zip(max_indices[0], max_indices[1]):
        if 0 < i < n - 1 and 0 < j < n - 1:
            interior_max = True
            break

    assert interior_max, "Maximum should occur in interior"
