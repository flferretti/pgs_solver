import jax
import jax.numpy as jnp
import jax.dlpack as jdlpack
from jax.lib import xla_client
from . import _pgs_solver as pgs

# Register the custom call using the capsule
xla_client.register_custom_call_target(
    "pgs_solver", pgs.get_pgs_solver_capsule(), platform="gpu"
)


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
    verbose: bool = False,
) -> tuple[jnp.ndarray, dict]:
    """Solve the constrained linear system Ax = b with bounds lo <= x <= hi using PGS."""
    # Create GPU context for the solver
    context = pgs.GPUContext(0)

    # Process matrices
    if isinstance(A, tuple) and len(A) == 3:
        # Single CSR matrix: (indptr, indices, data)
        matrix = csr_to_pgs_dlpack(A[0], A[1], A[2], context)
        matrices = [matrix]
    elif isinstance(A, list):
        # List of matrices for multi-GPU
        matrices = []
        for mat in A:
            if isinstance(mat, tuple) and len(mat) == 3:
                matrix = csr_to_pgs_dlpack(mat[0], mat[1], mat[2], context)
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

    # Make sure vectors are the right dtype
    x0 = jnp.asarray(x0, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)

    # Convert vectors to DLPack tensors
    x_dlpack = jdlpack.to_dlpack(x0)
    b_dlpack = jdlpack.to_dlpack(b)
    lo_dlpack = jdlpack.to_dlpack(lo)
    hi_dlpack = jdlpack.to_dlpack(hi)

    # Set up solver configuration
    config = pgs.PGSSolverConfig()
    config.max_iterations = max_iterations
    config.tolerance = tolerance
    config.relaxation = relaxation
    config.verbose = verbose

    # Create solver
    solver = pgs.PGSSolver(config)

    # Get DLPack tensors for matrices and create an array of them
    matrix_dlpacks = [m.__dlpack__() for m in matrices]

    # Call the solver with DLPack tensors
    status = solver.solve_dlpack(
        matrix_dlpacks, x_dlpack, b_dlpack, lo_dlpack, hi_dlpack
    )

    # Solution is written directly to x0 via the DLPack tensor

    # Return solution and status information
    info = {
        "status": status,
        "iterations": solver.iterations,
        "residual": solver.residual,
    }

    return x0, info


def csr_to_pgs_dlpack(indptr, indices, data, gpu_context=None):
    """Convert CSR components to a pgs_solver SparseMatrix using DLPack."""
    if gpu_context is None:
        gpu_context = pgs.GPUContext(0)

    # Ensure arrays are JAX arrays with correct types
    indptr = jnp.asarray(indptr, dtype=jnp.int32)
    indices = jnp.asarray(indices, dtype=jnp.int32)
    data = jnp.asarray(data, dtype=jnp.float32)

    # Convert to DLPack
    indptr_dlpack = jdlpack.to_dlpack(indptr)
    indices_dlpack = jdlpack.to_dlpack(indices)
    data_dlpack = jdlpack.to_dlpack(data)

    # Create sparse matrix using the existing DLPack constructor wrapper
    return pgs.SparseMatrix_from_dlpack(
        gpu_context,
        indptr.shape[0] - 1,  # num_rows
        int(indices.max().item()) + 1,  # num_cols
        data.shape[0],  # nnz
        indptr_dlpack,
        indices_dlpack,
        data_dlpack,
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
