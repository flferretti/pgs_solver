#include <iostream>
#include <vector>
#include "pgs_solver.cuh"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void inspectMatrixKernel(const int* row_ptr, const int* col_indices, const float* values,
                                    int rows, int nnz) {
    printf("CSR Matrix Inspection:\n");
    printf("Row pointers: ");
    for (int i = 0; i <= 5 && i <= rows; i++) {
        printf("%d ", row_ptr[i]);
    }
    printf("...\n");

    printf("First 10 columns & values:\n");
    for (int i = 0; i < 10 && i < nnz; i++) {
        printf("[%d] = %.1f, ", col_indices[i], values[i]);
    }
    printf("\n");
}

int main() {
    try {
        // Create a higher resolution grid for more accurate solution
        const int n = 32;  // Grid size
        const int size = n * n;

        const float h = 1.0f / (n - 1);  // Grid spacing for domain [0,1]
        const float h2_inv = 1.0f / (h * h);  // 1/h²

        // Allocate host-side CSR matrix components
        std::vector<int> h_row_ptr(size + 1);
        std::vector<int> h_col_indices;
        std::vector<float> h_values;

        h_row_ptr[0] = 0;
        int nnz = 0;

        // 5-point stencil for 2D Poisson equation
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int row = i * n + j;

                // Center point (diagonal)
                h_col_indices.push_back(row);
                h_values.push_back(4.0f * h2_inv);  // Scaled by 1/h²
                nnz++;

                // Connect to neighbors (if they exist)
                if (i > 0) {    // North
                    h_col_indices.push_back(row - n);
                    h_values.push_back(-1.0f * h2_inv);  // Scaled by 1/h²
                    nnz++;
                }
                if (i < n-1) {  // South
                    h_col_indices.push_back(row + n);
                    h_values.push_back(-1.0f * h2_inv);  // Scaled by 1/h²
                    nnz++;
                }
                if (j > 0) {    // West
                    h_col_indices.push_back(row - 1);
                    h_values.push_back(-1.0f * h2_inv);  // Scaled by 1/h²
                    nnz++;
                }
                if (j < n-1) {  // East
                    h_col_indices.push_back(row + 1);
                    h_values.push_back(-1.0f * h2_inv);  // Scaled by 1/h²
                    nnz++;
                }

                h_row_ptr[row + 1] = nnz;
            }
        }

        // Print matrix statistics
        std::cout << "Matrix size: " << size << "x" << size << ", non-zeros: " << nnz << std::endl;

        // Create a simple RHS vector (all 1's for interior nodes, 0 for boundary)
        std::vector<float> h_b(size, 1.0f);           // RHS (f=1 for Poisson equation)
        std::vector<float> h_lo(size, -HUGE_VALF);    // Lower bounds (-infinity for interior)
        std::vector<float> h_hi(size, HUGE_VALF);     // Upper bounds (+infinity for interior)
        std::vector<float> h_x0(size, 0.0f);          // Initial guess
        std::vector<float> h_result(size);            // Vector to hold solution

        // Set boundary conditions: fixed at zero
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int idx = i * n + j;
                if (i == 0 || i == n-1 || j == 0 || j == n-1) {
                    h_lo[idx] = 0.0f;     // Fix lower bound at zero for boundary
                    h_hi[idx] = 0.0f;     // Fix upper bound at zero for boundary
                    h_b[idx] = 0.0f;      // Set RHS to zero at boundary
                }
            }
        }

        // Create device memory
        int *d_row_ptr, *d_col_indices;
        float *d_values, *d_b, *d_lo, *d_hi, *d_x;

        cudaMalloc(&d_row_ptr, (size + 1) * sizeof(int));
        cudaMalloc(&d_col_indices, nnz * sizeof(int));
        cudaMalloc(&d_values, nnz * sizeof(float));
        cudaMalloc(&d_b, size * sizeof(float));
        cudaMalloc(&d_lo, size * sizeof(float));
        cudaMalloc(&d_hi, size * sizeof(float));
        cudaMalloc(&d_x, size * sizeof(float));

        // Copy data to device
        cudaMemcpy(d_row_ptr, h_row_ptr.data(), (size + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, h_col_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_lo, h_lo.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_hi, h_hi.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, h_x0.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        // Verify matrix on device
        inspectMatrixKernel<<<1, 1>>>(d_row_ptr, d_col_indices, d_values, size, nnz);
        cudaDeviceSynchronize();

        // Setup PGS solver
        cuda_pgs::GPUContext context(0);
        cuda_pgs::PGSSolverConfig config;
        config.max_iterations = 1500;
        config.tolerance = 1e-6;
        config.relaxation = 1.5f;  // Use over-relaxation for faster convergence
        config.verbose = true;

        cuda_pgs::PGSSolver solver(config);

        // Create PGS objects
        cuda_pgs::SparseMatrix A(context, size, size, nnz, d_row_ptr, d_col_indices, d_values);
        cuda_pgs::DeviceVector b_vec(context, size);
        cuda_pgs::DeviceVector lo_vec(context, size);
        cuda_pgs::DeviceVector hi_vec(context, size);
        cuda_pgs::DeviceVector x_vec(context, size);

        // Copy data to PGS vectors
        b_vec.CopyFromHost(h_b.data());
        lo_vec.CopyFromHost(h_lo.data());
        hi_vec.CopyFromHost(h_hi.data());
        x_vec.CopyFromHost(h_x0.data());

        // Solve the system
        std::cout << "Starting PGS solver..." << std::endl;
        cuda_pgs::SolverStatus status = solver.Solve({&A}, &x_vec, &b_vec, &lo_vec, &hi_vec);

        // Copy solution back to host
        x_vec.CopyToHost(h_result.data());

        // Print results
        std::cout << "\nSolver status: " << static_cast<int>(status) << std::endl;
        std::cout << "Iterations: " << solver.iterations() << std::endl;
        std::cout << "Final residual: " << solver.residual() << std::endl;

        // Print part of the solution
        std::cout << "Solution (first 10 elements): ";
        for (int i = 0; i < 10 && i < size; i++) {
            std::cout << h_result[i] << " ";
        }
        std::cout << std::endl;

        // Free device memory
        CHECK_CUDA(cudaFree(d_row_ptr));
        CHECK_CUDA(cudaFree(d_col_indices));
        CHECK_CUDA(cudaFree(d_values));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_lo));
        CHECK_CUDA(cudaFree(d_hi));
        CHECK_CUDA(cudaFree(d_x));

        FILE* fp = fopen("poisson_solution.txt", "w");
        if (fp) {
            // Save grid dimensions
            fprintf(fp, "%d\n", n);

            // Save all solution values
            for (int i = 0; i < size; i++) {
                fprintf(fp, "%.6f\n", h_result[i]);
            }
            fclose(fp);
            std::cout << "Solution saved to poisson_solution.txt\n";

        // Create a Python script to visualize the solution
        fp = fopen("visualize_solution.py", "w");
        if (fp) {
            fprintf(fp, "import numpy as np\n");
            fprintf(fp, "import matplotlib.pyplot as plt\n");
            fprintf(fp, "from mpl_toolkits.mplot3d import Axes3D\n");
            fprintf(fp, "from matplotlib import cm\n\n");

            fprintf(fp, "# Load solution\n");
            fprintf(fp, "with open('poisson_solution.txt', 'r') as f:\n");
            fprintf(fp, "    n = int(f.readline().strip())\n");
            fprintf(fp, "    solution = np.array([float(line.strip()) for line in f])\n\n");

            fprintf(fp, "# Reshape to 2D grid\n");
            fprintf(fp, "grid_solution = solution.reshape((n, n))\n\n");

            fprintf(fp, "# Ensure boundary conditions are correctly enforced\n");
            fprintf(fp, "for i in range(n):\n");
            fprintf(fp, "    for j in range(n):\n");
            fprintf(fp, "        if i == 0 or i == n-1 or j == 0 or j == n-1:\n");
            fprintf(fp, "            grid_solution[i,j] = 0.0\n\n");

            fprintf(fp, "# Create grid coordinates\n");
            fprintf(fp, "x = np.linspace(0, 1, n)\n");
            fprintf(fp, "y = np.linspace(0, 1, n)\n");
            fprintf(fp, "X, Y = np.meshgrid(x, y)\n\n");

            fprintf(fp, "# Analytical solution function\n");
            fprintf(fp, "def analytical_solution(x, y):\n");
            fprintf(fp, "    result = 0.0\n");
            fprintf(fp, "    for i in range(1, 50, 2):  # Sum over odd i values\n");
            fprintf(fp, "        for j in range(1, 50, 2):  # Sum over odd j values\n");
            fprintf(fp, "            term = np.sin(i*np.pi*x) * np.sin(j*np.pi*y)\n");
            fprintf(fp, "            term /= (i*j*(i**2 + j**2))\n");
            fprintf(fp, "            result += term\n");
            fprintf(fp, "    return 16.0/(np.pi**4) * result\n\n");

            fprintf(fp, "# Compute analytical solution on grid\n");
            fprintf(fp, "analytical = np.zeros((n, n))\n");
            fprintf(fp, "for i in range(n):\n");
            fprintf(fp, "    for j in range(n):\n");
            fprintf(fp, "        analytical[i,j] = analytical_solution(x[j], y[i])\n\n");

            fprintf(fp, "# Calculate error\n");
            fprintf(fp, "error = np.abs(grid_solution - analytical)\n");
            fprintf(fp, "max_error = np.max(error)\n");
            fprintf(fp, "l2_error = np.sqrt(np.mean(error**2))\n");
            fprintf(fp, "print(f'Maximum error: {max_error:.6e}')\n");
            fprintf(fp, "print(f'L2 error: {l2_error:.6e}')\n\n");

            fprintf(fp, "# Create comparison plots\n");
            fprintf(fp, "fig = plt.figure(figsize=(18, 6))\n");

            fprintf(fp, "# Plot numerical solution\n");
            fprintf(fp, "ax1 = fig.add_subplot(131, projection='3d')\n");
            fprintf(fp, "surf1 = ax1.plot_surface(X, Y, grid_solution, cmap='viridis')\n");
            fprintf(fp, "ax1.set_xlabel('X')\n");
            fprintf(fp, "ax1.set_ylabel('Y')\n");
            fprintf(fp, "ax1.set_zlabel('Solution')\n");
            fprintf(fp, "ax1.set_title('Numerical Solution')\n");

            fprintf(fp, "# Plot analytical solution\n");
            fprintf(fp, "ax2 = fig.add_subplot(132, projection='3d')\n");
            fprintf(fp, "surf2 = ax2.plot_surface(X, Y, analytical, cmap='viridis')\n");
            fprintf(fp, "ax2.set_xlabel('X')\n");
            fprintf(fp, "ax2.set_ylabel('Y')\n");
            fprintf(fp, "ax2.set_zlabel('Solution')\n");
            fprintf(fp, "ax2.set_title('Analytical Solution')\n");

            fprintf(fp, "# Plot error\n");
            fprintf(fp, "ax3 = fig.add_subplot(133, projection='3d')\n");
            fprintf(fp, "surf3 = ax3.plot_surface(X, Y, error, cmap='hot')\n");
            fprintf(fp, "ax3.set_xlabel('X')\n");
            fprintf(fp, "ax3.set_ylabel('Y')\n");
            fprintf(fp, "ax3.set_zlabel('Error')\n");
            fprintf(fp, "ax3.set_title(f'Error (Max: {max_error:.6e})')\n");

            fprintf(fp, "plt.tight_layout()\n");
            fprintf(fp, "plt.savefig('poisson_comparison.png', dpi=300, bbox_inches='tight')\n");
            fprintf(fp, "plt.show()\n\n");

            fprintf(fp, "# Also create a 2D contour plot for better accuracy visualization\n");
            fprintf(fp, "plt.figure(figsize=(12, 4))\n");
            fprintf(fp, "plt.subplot(131)\n");
            fprintf(fp, "plt.contourf(X, Y, grid_solution, 20, cmap='viridis')\n");
            fprintf(fp, "plt.colorbar()\n");
            fprintf(fp, "plt.title('Numerical')\n");
            fprintf(fp, "plt.subplot(132)\n");
            fprintf(fp, "plt.contourf(X, Y, analytical, 20, cmap='viridis')\n");
            fprintf(fp, "plt.colorbar()\n");
            fprintf(fp, "plt.title('Analytical')\n");
            fprintf(fp, "plt.subplot(133)\n");
            fprintf(fp, "plt.contourf(X, Y, error, 20, cmap='hot')\n");
            fprintf(fp, "plt.colorbar()\n");
            fprintf(fp, "plt.title('Error')\n");
            fprintf(fp, "plt.savefig('poisson_contours.png', dpi=300, bbox_inches='tight')\n");
            fprintf(fp, "plt.show()\n");

            // Add convergence study
            fprintf(fp, "# Log-log plot showing error vs grid size\n");
            fprintf(fp, "# This is for reference - not actual data from this run\n");
            fprintf(fp, "grid_sizes = np.array([8, 16, 32, 64, 128])\n");
            fprintf(fp, "theoretical_errors = 1/(grid_sizes**2)\n");
            fprintf(fp, "plt.figure(figsize=(6, 5))\n");
            fprintf(fp, "plt.loglog(grid_sizes, theoretical_errors, 'r-', marker='o', label='O(h²) convergence')\n");
            fprintf(fp, "plt.scatter([n], [max_error], color='blue', s=100, label='Current run')\n");
            fprintf(fp, "plt.xlabel('Grid size')\n");
            fprintf(fp, "plt.ylabel('Maximum error')\n");
            fprintf(fp, "plt.title('Convergence Analysis')\n");
            fprintf(fp, "plt.grid(True)\n");
            fprintf(fp, "plt.legend()\n");
            fprintf(fp, "plt.savefig('convergence.png', dpi=300)\n");
            fprintf(fp, "plt.show()\n");

            fclose(fp);
            std::cout << "Run 'python visualize_solution.py' to see detailed comparisons\n";
        }
        }
    }
     catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}