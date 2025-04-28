# CUDA PGS Solver

A CUDA implementation of the Projected Gauss-Seidel (PGS) method for solving constrained linear systems with support for sparse matrices and multi-GPU execution.

## Features

- Fast CUDA implementation of the Projected Gauss-Seidel method
- Support for sparse matrices (CSR format)
- Multi-GPU execution for large problems
- DLPack integration for seamless interoperability with deep learning frameworks
- Python bindings with JAX integration
- SOR (Successive Over-Relaxation) support through relaxation parameter

## Installation

### Prerequisites

- CUDA Toolkit 11.0 or later
- CMake 3.10 or later
- A C++14 compatible compiler
- Python 3.7 or later (for Python bindings)
- JAX (for JAX integration)

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/flferretti/pgs_solver.git
cd pgs_solver
```

2. Create a build directory and run CMake:
```bash
mkdir build && cd build
cmake ..
make -j
```

3. Install the Python package:
```bash
pip install -e .
```

## Usage

### C++ API

```cpp
#include "pgs_solver.h"

// Initialize GPUs
cuda_pgs::PGSSolverConfig config;
config.max_iterations = 1000;
config.tolerance = 1e-6;
config.relaxation = 1.0;
config.verbose = true;

cuda_pgs::PGSSolver solver(config);

// Create sparse matrix in CSR format
int rows = 1000;
int cols = 1000;
int nnz = /* number of non-zeros */;
int* row_ptr = /* ... */;
int* col_indices = /* ... */;
float* values = /* ... */;

cuda_pgs::GPUContext context(0);
cuda_pgs::SparseMatrix A(context, rows, cols, nnz, row_ptr, col_indices, values);

// Create vectors
float* h_b = /* right-hand side */;
float* h_lo = /* lower bounds */;
float* h_hi = /* upper bounds */;
float* h_x0 = /* initial guess */;

cuda_pgs::DeviceVector b(context, rows, h_b);
cuda_pgs::DeviceVector lo(context, rows, h_lo);
cuda_pgs::DeviceVector hi(context, rows, h_hi);
cuda_pgs::DeviceVector x(context, rows, h_x0);

// Solve the system
cuda_pgs::SolverStatus status = solver.Solve({&A}, &x, &b, &lo, &hi);

// Get results
float* h_result = new float[rows];
x.CopyToHost(h_result);

// Check solver status
if (status == cuda_pgs::SolverStatus::SUCCESS) {
    std::cout << "Solver converged in " << solver.iterations() << " iterations\n";
    std::cout << "Final residual: " << solver.residual() << "\n";
} else {
    std::cout << "Solver did not converge\n";
}
```

### Python API

```python
import numpy as np
from pgs_solver.jax_interface import pgs_solve
import jax.numpy as jnp

# Create a test problem
n = 1000
A = np.random.rand(n, n).astype(np.float32)
x_true = np.random.rand(n).astype(np.float32)
b = A @ x_true
lo = x_true - 0.5
hi = x_true + 0.5

# Convert to JAX arrays
A_jax = jnp.array(A)
b_jax = jnp.array(b)
lo_jax = jnp.array(lo)
hi_jax = jnp.array(hi)

# Solve the system
x, info = pgs_solve(
    A_jax, b_jax, lo_jax, hi_jax,
    max_iterations=1000,
    tolerance=1e-6,
    relaxation=1.2,  # SOR factor
    verbose=True
)

# Check results
print(f"Solver {'converged' if info['status'] == 0 else 'did not converge'}")
print(f"Iterations: {info['iterations']}")
print(f"Final residual: {info['residual']}")
print(f"Error: {np.linalg.norm(np.array(x) - x_true) / np.linalg.norm(x_true)}")
```

### JAX Integration

```python
import jax
import jax.numpy as jnp
from pgs_solver import pgs_solve_jittable

# Create a JAX function that uses our solver
@jax.jit
def solve_system(A, b, lo, hi):
    config = {
        'max_iterations': 1000,
        'tolerance': 1e-6,
        'relaxation': 1.2,
        'gpu_ids': [0],
        'verbose': False
    }
    return pgs_solve_jittable(A, b, lo, hi, None, config)

# This function can be used in JAX computations
result = solve_system(A_jax, b_jax, lo_jax, hi_jax)
```

## Multi-GPU Usage

The solver supports distributing computation across multiple GPUs for large problems:

```python
# For a large problem, split the matrix into blocks
A_blocks = [
    jnp.array(A[:n//2, :]),  # First half for GPU 0
    jnp.array(A[n//2:, :])   # Second half for GPU 1
]

# Solve using multiple GPUs
x, info = pgs_solve(
    A_blocks, b_jax, lo_jax, hi_jax,
    gpu_ids=[0, 1],  # Use GPUs 0 and 1
    max_iterations=1000,
    tolerance=1e-6
)
```

## Examples

Check the `examples` directory for more detailed usage examples:

- `examples/benchmark.py`: Performance benchmarking script
- `examples/jax_example.py`: Example of using the solver with JAX

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
