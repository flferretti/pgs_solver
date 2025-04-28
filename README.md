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
