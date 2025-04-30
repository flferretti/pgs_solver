import jax
import jax.numpy as jnp
import jax.dlpack as jdlpack
from jax.lib import xla_client
from jax.core import Primitive
from jax import core
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir
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
    status, residual = solver.solve_dlpack(
        matrix_dlpacks, x_dlpack, b_dlpack, lo_dlpack, hi_dlpack
    )

    # Solution is written directly to x0 via the DLPack tensor

    # Return solution and status information
    info = {
        "status": status,
        "iterations": solver.iterations,
        "residual": residual,
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


# Create a JAX primitive for the PGS solver
pgs_solver_p = Primitive("pgs_solver")


# JAX implementation of the PGS solver
def _pgs_solver_impl(
    indptr, indices, data, b, lo, hi, x0, max_iterations, tolerance, relaxation
):
    # Reconstruct the tuple for calling the non-JIT version
    A = (indptr, indices, data)
    if x0 is None:
        x0 = jnp.zeros_like(b)
    return pgs_solve(A, b, lo, hi, x0, max_iterations, tolerance, relaxation)[0]


# Register the implementation
pgs_solver_p.def_impl(_pgs_solver_impl)


def _pgs_solver_abstract_eval(
    indptr, indices, data, b, lo, hi, x0, max_iterations, tolerance, relaxation
):
    # The output has the same shape and dtype as b
    return core.ShapedArray(b.shape, b.dtype)


# Register the abstract evaluation
pgs_solver_p.def_abstract_eval(_pgs_solver_abstract_eval)


# XLA compilation rule
def _pgs_solver_xla_translation(
    ctx,
    avals_in,
    avals_out,
    indptr,
    indices,
    data,
    b,
    lo,
    hi,
    x0,
    max_iterations,
    tolerance,
    relaxation,
):
    # Create custom call with flattened inputs
    out = custom_call(
        ctx.builder,
        "pgs_solver",
        operands=[indptr, indices, data, b, lo, hi, x0],
        shape=avals_out[0].shape,
        dtype=avals_out[0].dtype,
        operand_shapes=[x.shape for x in [indptr, indices, data, b, lo, hi, x0]],
        operand_dtypes=[x.dtype for x in [indptr, indices, data, b, lo, hi, x0]],
        opaque=str(
            {
                "max_iterations": max_iterations,
                "tolerance": tolerance,
                "relaxation": relaxation,
            }
        ),
    )
    return [out]


def _pgs_solver_lowering(
    ctx, indptr, indices, data, b, lo, hi, x0, max_iterations, tolerance, relaxation
):
    """Fixed MLIR lowering function that correctly returns the operation results."""
    # Get output type from context
    result_type = ctx.avals_out[0]

    # Create output shape and create IR types
    result_ir_type = mlir.aval_to_ir_type(result_type)

    # Create the custom call with proper API
    op = mlir.custom_call(
        "pgs_solver",
        result_types=[result_ir_type],
        operands=[indptr, indices, data, b, lo, hi, x0],
        # Pass configuration as serialized string instead
        backend_config=str(
            {
                "max_iterations": 1000,  # Use hardcoded defaults
                "tolerance": 1e-6,
                "relaxation": 1.0,
            }
        ),
    )

    return op.results


# Register MLIR lowering
mlir.register_lowering(pgs_solver_p, _pgs_solver_lowering, platform="gpu")


@jax.custom_vjp
def pgs_solve_jittable(A, b, lo, hi, x0=None, config=None):
    """
    JAX-jittable version of the PGS solver.
    """

    if config is None:
        config = {}

    # Extract config parameters with defaults
    max_iterations = config.get("max_iterations", 1000)
    tolerance = config.get("tolerance", 1e-6)
    relaxation = config.get("relaxation", 1.0)

    if x0 is None:
        x0 = jnp.zeros_like(b)

    # Unpack the A tuple to pass individual arrays
    indptr, indices, data = A

    # Call the primitive with flattened arguments
    return pgs_solver_p.bind(
        indptr, indices, data, b, lo, hi, x0, max_iterations, tolerance, relaxation
    )


# Forward and backward passes for custom VJP
def _pgs_solve_fwd(A, b, lo, hi, x0, config):
    result = pgs_solve_jittable(A, b, lo, hi, x0, config)
    return result, (A, b, lo, hi, result, config)


def _pgs_solve_bwd(res, grad_x):
    A, b, lo, hi, x, config = res
    # Simplified backward pass implementation
    grad_A = jax.tree_map(lambda x: jnp.zeros_like(x), A)
    grad_b = grad_x  # Simplified gradient passing
    grad_lo = jnp.zeros_like(lo)
    grad_hi = jnp.zeros_like(hi)
    grad_x0 = jnp.zeros_like(x)
    return grad_A, grad_b, grad_lo, grad_hi, grad_x0, None


pgs_solve_jittable.defvjp(_pgs_solve_fwd, _pgs_solve_bwd)
