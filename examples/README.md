# Running Examples

## Poisson CUDA Example

This example solves the 2D Poisson equation using the CUDA PGS solver.

```sh
./examples/poisson_cuda_example
```

After running, you can visualize the solution:
```sh
python visualize_solution.py
```

The visualization will show:
- The numerical solution from the solver
- The analytical solution for comparison
- The error between the two solutions
- A convergence analysis plot

## JAX Example

The JAX example demonstrates how to use the solver with JAX and compares performance with and without JIT compilation.

```sh
python jax_example.py
```

This will:
1. Solve a 2D Poisson equation using the JAX interface
2. Generate visualizations comparing the numerical and analytical solutions
3. Measure performance metrics

## Benchmark

The benchmark script evaluates the performance of the solver with different problem sizes and configurations.

```sh
python benchmark.py --single-gpu --multi-gpu --compare
```

Available options:
- `--single-gpu`: Run the single GPU scaling test
- `--multi-gpu`: Run the multi-GPU scaling test
- `--compare`: Compare with SciPy's direct solver

Results will be saved to `pgs_benchmark_results.png`.
