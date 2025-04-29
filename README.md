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
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=<install_prefix> ..
make -j
make install
```

3. Install the Python package:
```bash
pip install -e .
```

## Examples

Check the `examples` directory for more detailed usage examples:

- `examples/benchmark.py`: Performance benchmarking script
- `examples/jax_example.py`: Example of using the solver with JAX
- `examples/poisson_cuda_example.cu`: Example solving the 2D Poisson equation with CUDA

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
