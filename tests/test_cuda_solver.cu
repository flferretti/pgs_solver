#include "pgs_solver.cuh"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace cuda_pgs;

// Simple test for basic solver functionality
void test_diagonal_system() {
    std::cout << "Testing diagonal system..." << std::endl;
    
    const int n = 10;
    
    // Create a diagonal system: diag([1,2,3,...,10]) * x = [1,1,1,...,1]
    // Expected solution: x = [1, 0.5, 0.333..., ..., 0.1]
    
    // CSR format for diagonal matrix
    std::vector<int> indptr(n + 1);
    std::vector<int> indices(n);
    std::vector<float> data(n);
    
    for (int i = 0; i < n; i++) {
        indptr[i] = i;
        indices[i] = i;
        data[i] = static_cast<float>(i + 1);
    }
    indptr[n] = n;
    
    // RHS vector
    std::vector<float> b(n, 1.0f);
    
    // Bounds (unconstrained)
    std::vector<float> lo(n, -1e10f);
    std::vector<float> hi(n, 1e10f);
    
    // Initial guess
    std::vector<float> x0(n, 0.0f);
    
    // Create GPU context
    GPUContext context(0);
    
    // Create sparse matrix
    SparseMatrix matrix(&context, n, n, n, indptr.data(), indices.data(), data.data());
    
    // Create device vectors
    DeviceVector x(&context, n);
    DeviceVector b_dev(&context, n);
    DeviceVector lo_dev(&context, n);
    DeviceVector hi_dev(&context, n);
    
    x.copy_from_host(x0.data());
    b_dev.copy_from_host(b.data());
    lo_dev.copy_from_host(lo.data());
    hi_dev.copy_from_host(hi.data());
    
    // Create solver
    PGSSolver solver(&context, &matrix, &b_dev, &lo_dev, &hi_dev, &x);
    
    // Configure solver
    SolverConfig config;
    config.max_iterations = 100;
    config.tolerance = 1e-6f;
    config.relaxation = 1.0f;
    config.check_frequency = 10;
    config.verbose = false;
    
    // Solve
    SolverInfo info = solver.solve(config);
    
    // Get solution
    std::vector<float> solution(n);
    x.copy_to_host(solution.data());
    
    // Check solution
    bool passed = true;
    for (int i = 0; i < n; i++) {
        float expected = 1.0f / static_cast<float>(i + 1);
        float error = std::abs(solution[i] - expected);
        if (error > 1e-4f) {
            std::cerr << "Error at index " << i << ": expected " << expected 
                      << ", got " << solution[i] << " (error: " << error << ")" << std::endl;
            passed = false;
        }
    }
    
    assert(passed && "Diagonal system test failed!");
    assert(info.converged && "Solver did not converge!");
    
    std::cout << "  ✓ Diagonal system test passed" << std::endl;
    std::cout << "    Iterations: " << info.iterations << std::endl;
    std::cout << "    Final residual: " << info.final_residual << std::endl;
}

void test_identity_with_bounds() {
    std::cout << "\nTesting identity system with bounds..." << std::endl;
    
    const int n = 5;
    
    // Identity matrix
    std::vector<int> indptr(n + 1);
    std::vector<int> indices(n);
    std::vector<float> data(n, 1.0f);
    
    for (int i = 0; i <= n; i++) {
        indptr[i] = i;
    }
    for (int i = 0; i < n; i++) {
        indices[i] = i;
    }
    
    // System: I * x = [10, 10, 10, 10, 10]
    // With bounds: 0 <= x <= 5
    // Solution: x = [5, 5, 5, 5, 5] (all at upper bound)
    std::vector<float> b(n, 10.0f);
    std::vector<float> lo(n, 0.0f);
    std::vector<float> hi(n, 5.0f);
    std::vector<float> x0(n, 0.0f);
    
    // Create GPU context
    GPUContext context(0);
    
    // Create sparse matrix
    SparseMatrix matrix(&context, n, n, n, indptr.data(), indices.data(), data.data());
    
    // Create device vectors
    DeviceVector x(&context, n);
    DeviceVector b_dev(&context, n);
    DeviceVector lo_dev(&context, n);
    DeviceVector hi_dev(&context, n);
    
    x.copy_from_host(x0.data());
    b_dev.copy_from_host(b.data());
    lo_dev.copy_from_host(lo.data());
    hi_dev.copy_from_host(hi.data());
    
    // Create solver
    PGSSolver solver(&context, &matrix, &b_dev, &lo_dev, &hi_dev, &x);
    
    // Configure solver
    SolverConfig config;
    config.max_iterations = 100;
    config.tolerance = 1e-6f;
    config.relaxation = 1.0f;
    config.check_frequency = 10;
    config.verbose = false;
    
    // Solve
    SolverInfo info = solver.solve(config);
    
    // Get solution
    std::vector<float> solution(n);
    x.copy_to_host(solution.data());
    
    // Check solution (should be at upper bound)
    bool passed = true;
    for (int i = 0; i < n; i++) {
        float expected = 5.0f;
        float error = std::abs(solution[i] - expected);
        if (error > 1e-4f) {
            std::cerr << "Error at index " << i << ": expected " << expected 
                      << ", got " << solution[i] << " (error: " << error << ")" << std::endl;
            passed = false;
        }
    }
    
    assert(passed && "Identity with bounds test failed!");
    assert(info.converged && "Solver did not converge!");
    
    std::cout << "  ✓ Identity with bounds test passed" << std::endl;
    std::cout << "    Iterations: " << info.iterations << std::endl;
    std::cout << "    Final residual: " << info.final_residual << std::endl;
}

int main() {
    std::cout << "=== Running CUDA Solver Tests ===" << std::endl;
    
    try {
        test_diagonal_system();
        test_identity_with_bounds();
        
        std::cout << "\n=== All CUDA tests passed! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== Test failed with exception: " << e.what() << " ===" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n=== Test failed with unknown exception ===" << std::endl;
        return 1;
    }
}
