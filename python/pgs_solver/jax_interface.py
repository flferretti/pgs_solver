import jax
import jax.numpy as jnp
import jax.dlpack as jdlpack
import numpy as np
from . import _pgs_solver as pgs


def pgs_solve(
    A: (
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        | list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    ),
    b: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    relaxation: float = 1.0,
    check_frequency: int = 10,
    verbose: bool = False,
) -> tuple[jnp.ndarray, dict]:
    """Solve the constrained linear system Ax = b with bounds lo <= x <= hi using PGS.

    Args:
        A: CSR matrix as (indptr, indices, data) tuple, or list of such tuples for multi-GPU
        b: Right-hand side vector
        lo: Lower bounds on solution
        hi: Upper bounds on solution
        x0: Initial guess (optional, defaults to zeros)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        relaxation: SOR relaxation parameter (1.0 = standard PGS, >1.0 = over-relaxation)
        verbose: Print iteration progress

    Returns:
        Tuple of (solution, info_dict) where info_dict contains 'status', 'iterations', 'residual'
    """
    # Create GPU context for the solver
    context = pgs.GPUContext(0)

    # Process matrices
    if isinstance(A, tuple) and len(A) == 3:
        # Single CSR matrix: (indptr, indices, data)
        matrix = _csr_to_pgs_matrix(A[0], A[1], A[2], context)
        matrices = [matrix]
    elif isinstance(A, list):
        # List of matrices for multi-GPU
        matrices = []
        for mat in A:
            if isinstance(mat, tuple) and len(mat) == 3:
                matrix = _csr_to_pgs_matrix(mat[0], mat[1], mat[2], context)
                matrices.append(matrix)
            else:
                raise TypeError(
                    f"Expected tuple of (indptr, indices, data), got {type(mat)}"
                )
    else:
        raise TypeError(f"Expected CSR tuple or list of CSR tuples, got {type(A)}")

    # Initialize solution if not provided
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # Make sure vectors are the right dtype and on device
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)

    # Create device vectors from JAX arrays via DLPack
    x_vec = _jax_array_to_device_vector(x0, context)
    b_vec = _jax_array_to_device_vector(b, context)
    lo_vec = _jax_array_to_device_vector(lo, context)
    hi_vec = _jax_array_to_device_vector(hi, context)

    # Set up solver configuration
    config = pgs.PGSSolverConfig()
    config.max_iterations = max_iterations
    config.tolerance = tolerance
    config.relaxation = relaxation
    config.check_frequency = check_frequency
    config.verbose = verbose

    # Create solver
    solver = pgs.PGSSolver(config)

    # Call the solver
    status = solver.solve(matrices, x_vec, b_vec, lo_vec, hi_vec)

    # Convert solution back to NumPy then JAX array
    solution_np = np.zeros(x0.shape, dtype=np.float32)
    x_vec.copy_to_host(solution_np)
    solution = jnp.array(solution_np)

    # Return solution and status information
    info = {
        "status": int(status.value) if hasattr(status, "value") else int(status),
        "iterations": solver.iterations,
        "residual": solver.residual,
    }

    return solution, info


def _csr_to_pgs_matrix(indptr, indices, data, gpu_context):
    """Convert CSR components to a pgs_solver SparseMatrix."""
    # Ensure arrays are JAX arrays with correct types
    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float32)

    # Convert to DLPack capsules
    indptr_capsule = jdlpack.to_dlpack(indptr)
    indices_capsule = jdlpack.to_dlpack(indices)
    data_capsule = jdlpack.to_dlpack(data)

    # Calculate matrix dimensions
    num_rows = indptr.shape[0] - 1
    nnz = data.shape[0]

    # Infer num_cols from max column index
    num_cols = int(jnp.max(indices).item()) + 1 if nnz > 0 else num_rows

    # Create sparse matrix using the DLPack constructor wrapper
    return pgs.SparseMatrix_from_dlpack(
        gpu_context,
        num_rows,
        num_cols,
        nnz,
        indptr_capsule,
        indices_capsule,
        data_capsule,
    )


def _jax_array_to_device_vector(jax_array, gpu_context):
    """Convert a JAX array to a pgs_solver DeviceVector by copying data."""
    # Convert to numpy on host first
    host_array = np.array(jax_array, dtype=np.float32)

    # Create DeviceVector and copy from host
    vec = pgs.DeviceVector(gpu_context, host_array.size)
    vec.copy_from_host(host_array)

    return vec
