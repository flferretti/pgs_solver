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

## Running Examples

### Poisson CUDA Example

This example solves the 2D Poisson equation using the CUDA PGS solver.

```bash
cd build
./poisson_cuda_example
```

After running, you can visualize the solution:
```bash
python visualize_solution.py
```

The visualization will show:
- The numerical solution from the solver
- The analytical solution for comparison
- The error between the two solutions
- A convergence analysis plot

### JAX Example

The JAX example demonstrates how to use the solver with JAX and compares performance with and without JIT compilation.

```bash
cd examples
python jax_example.py
```

This will:
1. Solve a 2D Poisson equation using the JAX interface
2. Generate visualizations comparing the numerical and analytical solutions
3. Measure performance metrics

### Benchmark

The benchmark script evaluates the performance of the solver with different problem sizes and configurations.

```bash
cd examples
python benchmark.py --single-gpu --multi-gpu --compare
```

Available options:
- `--single-gpu`: Run the single GPU scaling test
- `--multi-gpu`: Run the multi-GPU scaling test
- `--compare`: Compare with SciPy's direct solver

Results will be saved to `pgs_benchmark_results.png`.

## Examples

Check the `examples` directory for more detailed usage examples:

- `examples/benchmark.py`: Performance benchmarking script
- `examples/jax_example.py`: Example of using the solver with JAX
- `examples/poisson_cuda_example.cu`: Example solving the 2D Poisson equation with CUDA

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.
