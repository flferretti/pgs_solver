import numpy as np
import jax.numpy as jnp
from cupgs.jax_interface import pgs_solve


def test_jax_array_input():
    """Test that JAX arrays are properly converted."""
    n = 5

    # Simple diagonal system
    indptr = jnp.arange(n + 1, dtype=jnp.int32)
    indices = jnp.arange(n, dtype=jnp.int32)
    data = jnp.ones(n, dtype=jnp.float32) * 2.0

    b = jnp.ones(n, dtype=jnp.float32)
    lo = jnp.full(n, -1e10, dtype=jnp.float32)
    hi = jnp.full(n, 1e10, dtype=jnp.float32)
    x0 = jnp.zeros(n, dtype=jnp.float32)

    solution, info = pgs_solve(
        (indptr, indices, data),
        b,
        lo,
        hi,
        x0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
    )

    assert isinstance(solution, (np.ndarray, jnp.ndarray))
    assert solution.shape == (n,)
    assert info["status"] == 0


def test_numpy_array_input():
    """Test that NumPy arrays are properly converted."""

    n = 5

    # Simple diagonal system
    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    data = np.ones(n, dtype=np.float32) * 2.0

    b = np.ones(n, dtype=np.float32)
    lo = np.full(n, -1e10, dtype=np.float32)
    hi = np.full(n, 1e10, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)

    solution, info = pgs_solve(
        (indptr, indices, data),
        b,
        lo,
        hi,
        x0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
    )

    assert isinstance(solution, (np.ndarray, jnp.ndarray))
    assert solution.shape == (n,)
    assert info["status"] == 0


def test_mixed_array_types():
    """Test mixing JAX and NumPy arrays."""
    n = 5

    # Matrix as NumPy
    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    data = np.ones(n, dtype=np.float32) * 2.0

    # Vectors as JAX
    b = jnp.ones(n, dtype=jnp.float32)
    lo = jnp.full(n, -1e10, dtype=jnp.float32)
    hi = jnp.full(n, 1e10, dtype=jnp.float32)
    x0 = jnp.zeros(n, dtype=jnp.float32)

    solution, info = pgs_solve(
        (indptr, indices, data),
        b,
        lo,
        hi,
        x0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
    )

    assert solution.shape == (n,)
    assert info["status"] == 0


def test_dtype_conversion():
    """Test that incorrect dtypes are properly converted."""
    n = 5

    # Wrong dtypes (int64, float64)
    indptr = np.arange(n + 1, dtype=np.int64)  # Should be int32
    indices = np.arange(n, dtype=np.int64)  # Should be int32
    data = np.ones(n, dtype=np.float64) * 2.0  # Should be float32

    b = np.ones(n, dtype=np.float64)
    lo = np.full(n, -1e10, dtype=np.float64)
    hi = np.full(n, 1e10, dtype=np.float64)
    x0 = np.zeros(n, dtype=np.float64)

    # Should automatically convert
    solution, info = pgs_solve(
        (indptr, indices, data),
        b,
        lo,
        hi,
        x0,
        max_iterations=50,
        tolerance=1e-6,
        verbose=False,
    )

    assert solution.shape == (n,)

    # Solution should be float32
    assert solution.dtype == np.float32 or solution.dtype == jnp.float32


def test_return_types():
    """Test that return values have correct types."""

    n = 5
    indptr = jnp.arange(n + 1, dtype=jnp.int32)
    indices = jnp.arange(n, dtype=jnp.int32)
    data = jnp.ones(n, dtype=jnp.float32)

    b = jnp.ones(n, dtype=jnp.float32)
    lo = jnp.full(n, -1e10, dtype=jnp.float32)
    hi = jnp.full(n, 1e10, dtype=jnp.float32)
    x0 = jnp.zeros(n, dtype=jnp.float32)

    solution, info = pgs_solve(
        (indptr, indices, data), b, lo, hi, x0, max_iterations=50, verbose=False
    )

    # Check return types
    assert isinstance(solution, (np.ndarray, jnp.ndarray))
    assert isinstance(info, dict)
    assert "status" in info
    assert "iterations" in info
    assert "residual" in info
    assert isinstance(info["status"], (int, np.integer))
    assert isinstance(info["iterations"], (int, np.integer))
    assert isinstance(info["residual"], (float, np.floating))


def test_csr_format_validation():
    """Test that CSR format is correctly validated."""
    n = 5

    # Valid CSR
    indptr = jnp.arange(n + 1, dtype=jnp.int32)
    indices = jnp.arange(n, dtype=jnp.int32)
    data = jnp.ones(n, dtype=jnp.float32)

    b = jnp.ones(n, dtype=jnp.float32)
    lo = jnp.full(n, -1e10, dtype=jnp.float32)
    hi = jnp.full(n, 1e10, dtype=jnp.float32)
    x0 = jnp.zeros(n, dtype=jnp.float32)

    solution, info = pgs_solve(
        (indptr, indices, data), b, lo, hi, x0, max_iterations=50, verbose=False
    )
    assert solution.shape == (n,)


def test_empty_matrix():
    """Test handling of matrix with no off-diagonal elements."""
    n = 3

    # Diagonal-only matrix
    indptr = np.array([0, 1, 2, 3], dtype=np.int32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    lo = np.full(n, -1e10, dtype=np.float32)
    hi = np.full(n, 1e10, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)

    solution, info = pgs_solve(
        (indptr, indices, data), b, lo, hi, x0, max_iterations=50, verbose=False
    )

    # Should converge immediately for diagonal matrix
    assert info["status"] == 0
    np.testing.assert_allclose(np.array(solution), np.array([1.0, 1.0, 1.0]), rtol=1e-4)
