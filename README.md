# GPU PGS Solver

A GPU-accelerated implementation of the Projected Gauss-Seidel (PGS) method for solving constrained linear systems with support for sparse matrices and multi-GPU execution.

Supports both **NVIDIA CUDA** and **AMD ROCm/HIP** backends.

## Features

- Fast GPU implementation of the Projected Gauss-Seidel method
- **Dual backend support**: NVIDIA CUDA and AMD ROCm/HIP (experimental)
- Support for sparse matrices (CSR format)
- Multi-GPU execution for large problems
- DLPack integration for seamless interoperability with deep learning frameworks
- Python bindings with JAX integration
- SOR (Successive Over-Relaxation) support through relaxation parameter

## Installation

### Prerequisites

**For NVIDIA GPUs:**
- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- A C++14 compatible compiler
- Python 3.10 or later (for Python bindings)
- JAX (for JAX integration)

**For AMD GPUs (Experimental):**
- ROCm 5.0 or later with HIP
- rocSPARSE library
- CMake 3.18 or later
- A C++14 compatible compiler (hipcc)
- Python 3.10 or later (for Python bindings)
- JAX (for JAX integration)

See [docs/ROCM.md](docs/ROCM.md) for detailed ROCm setup instructions.

### Building from Source

#### CUDA Backend (Default)

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

#### ROCm Backend (Experimental)

1. Build with ROCm support:
```bash
mkdir -p build && cd build
cmake -DPGS_USE_ROCM=ON \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DCMAKE_INSTALL_PREFIX=<install_prefix> \
      ..
make -j
make install
```

2. Install the Python package:
```bash
export PGS_USE_ROCM=1
pip install -e .
```

See [docs/ROCM.md](docs/ROCM.md) for more details on ROCm support.

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
