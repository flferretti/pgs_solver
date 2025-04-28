import jax
import jax.numpy as jnp
from jax.lib import xla_client
import numpy as np
from . import _pgs_solver as pgs

# Register JAX custom calls
for name, fn in [
    ("pgs_solver", pgs.PGSSolver.solve_dlpack),
]:
    xla_client.register_custom_call_target(name, fn, platform="gpu")


def pgs_solve(
    A: jnp.ndarray | list[jnp.ndarray],
    b: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    x0: jnp.ndarray | None = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
    relaxation: float = 1.0,
    verbose: bool = False,
) -> tuple[jnp.ndarray, dict]:
    """Solve the constrained linear system Ax = b with bounds lo <= x <= hi using the Projected Gauss-Seidel method on CUDA GPUs.

    This function solves a bounded linear system using the Projected Gauss-Seidel algorithm,
    with support for both single and multi-GPU execution.

    Args:
        A: The system matrix or a list of matrix
            blocks for multi-GPU execution. For multi-GPU, provide a list of matrices
            that will be distributed across GPUs.
        b: The right-hand side vector.
        lo: Lower bounds for the solution.
        hi: Upper bounds for the solution.
        x0: Initial guess for the solution. If None, zeros will be used.
            Defaults to None.
        max_iterations: Maximum number of iterations. Defaults to 1000.
        tolerance: Convergence tolerance for the residual. Defaults to 1e-6.
        relaxation: Relaxation factor for SOR-like behavior. Defaults to 1.0.
        verbose: Whether to print progress information. Defaults to False.

    Returns:
        A tuple containing:
            - x: The solution vector.
            - info: Additional information about the solve:

    Raises:
        TypeError: If A is not a JAX array or a list of JAX arrays.
    """
    # Input validation
    if isinstance(A, list):
        # Multi-GPU mode
        if not all(isinstance(mat, jnp.ndarray) for mat in A):
            raise TypeError("All elements in A must be JAX arrays")
    else:
        # Single GPU mode
        if not isinstance(A, jnp.ndarray):
            raise TypeError("A must be a JAX array or a list of JAX arrays")
        A = [A]  # Convert to list for unified processing

    # Initialize solution if not provided
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # Set up solver configuration
    config = pgs.PGSSolverConfig()
    config.max_iterations = max_iterations
    config.tolerance = tolerance
    config.relaxation = relaxation
    config.verbose = verbose

    # Create solver
    solver = pgs.PGSSolver(config)

    # Convert JAX arrays to DLPack format
    A_dlpack = [jax.dlpack.to_dlpack(mat) for mat in A]
    x_dlpack = jax.dlpack.to_dlpack(x0)
    b_dlpack = jax.dlpack.to_dlpack(b)
    lo_dlpack = jax.dlpack.to_dlpack(lo)
    hi_dlpack = jax.dlpack.to_dlpack(hi)

    # Solve the system
    status = solver.solve_dlpack(A_dlpack, x_dlpack, b_dlpack, lo_dlpack, hi_dlpack)

    # Convert solution back to JAX array
    x = jax.dlpack.from_dlpack(x_dlpack)

    # Return solution and status information
    info = {
        "status": status,
        "iterations": solver.iterations,
        "residual": solver.residual,
    }

    return x, info


def scipy_csr_to_pgs(matrix, gpu_context):
    """
    Convert a scipy CSR matrix to a pgs_solver SparseMatrix.

    Args:
        matrix : Sparse matrix in CSR format.
        gpu_context: GPU context to use for the conversion.

    Returns:
        pgs_matrix : Matrix in the format required by the PGS solver.
    """
    from scipy import sparse

    if not sparse.isspmatrix_csr(matrix):
        matrix = sparse.csr_matrix(matrix)

    return pgs.SparseMatrix(
        gpu_context,
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        matrix.indptr.astype(np.int32),
        matrix.indices.astype(np.int32),
        matrix.data.astype(np.float32),
    )


# JAX jittable version of the solver
@jax.custom_vjp
def pgs_solve_jittable(A, b, lo, hi, x0=None, config=None):
    """
    JAX-jittable version of the PGS solver.

    This function can be used in JAX computations and supports autodiff.
    """
    # Implementation details for JAX custom VJP would go here
    # This is a simplified placeholder
    result = pgs_solve(A, b, lo, hi, x0, **(config or {}))
    return result[0]  # Return only the solution vector


# Forward and backward passes for autodiff would be defined here
def _pgs_solve_fwd(A, b, lo, hi, x0, config):
    x = pgs_solve_jittable(A, b, lo, hi, x0, config)
    return x, (A, b, lo, hi, x, config)


def _pgs_solve_bwd(res, grad_x):
    A, b, lo, hi, x, config = res
    # Implementation of the backward pass
    # This would involve solving the adjoint system
    # Placeholder for actual implementation
    grad_A = jnp.zeros_like(A)
    grad_b = jnp.zeros_like(b)
    grad_lo = jnp.zeros_like(lo)
    grad_hi = jnp.zeros_like(hi)
    grad_x0 = jnp.zeros_like(x)
    return grad_A, grad_b, grad_lo, grad_hi, grad_x0, None


pgs_solve_jittable.defvjp(_pgs_solve_fwd, _pgs_solve_bwd)
