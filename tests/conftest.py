import pytest
import numpy as np


@pytest.fixture
def small_diagonal_matrix():
    """Create a small diagonal system for testing."""
    n = 10
    # Diagonal matrix: diag([1, 2, 3, ..., 10])
    indptr = np.arange(n + 1, dtype=np.int32)
    indices = np.arange(n, dtype=np.int32)
    data = np.arange(1, n + 1, dtype=np.float32)

    return (indptr, indices, data), n


@pytest.fixture
def poisson_matrix_small():
    """Create a small 2D Poisson system (8x8 grid)."""
    n = 8
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

    row_ptr = np.array(row_ptr, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    values = np.array(values, dtype=np.float32)

    return (row_ptr, col_indices, values), size


@pytest.fixture
def tolerance():
    """Default tolerance for convergence tests."""
    return 1e-5


@pytest.fixture
def max_iterations():
    """Default maximum iterations."""
    return 1000
