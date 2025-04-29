#include "pgs_solver.cuh"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <limits>

namespace cuda_pgs {

// Helpers to enable P2P transfer between GPUs
void enable_peer_access(int from_device, int to_device) {
    int can_access_peer = 0;

    cudaDeviceCanAccessPeer(&can_access_peer, to_device, from_device);

    if (can_access_peer) {
        cudaSetDevice(to_device);
        cudaDeviceEnablePeerAccess(from_device, 0); // ignore error if already enabled
    }
}

// GPU Context implementation
GPUContext::GPUContext(int device_id) : device_id_(device_id) {
    cudaError_t cuda_status = cudaSetDevice(device_id);
    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to set CUDA device: " +
                         std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaStreamCreate(&stream_);
    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to create CUDA stream: " +
                         std::string(cudaGetErrorString(cuda_status)));
    }

    cusparseStatus_t cusparse_status = cusparseCreate(&cusparse_handle_);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaStreamDestroy(stream_);
        throw CudaError("Failed to create cuSPARSE handle");
    }

    cusparse_status = cusparseSetStream(cusparse_handle_, stream_);
    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroy(cusparse_handle_);
        cudaStreamDestroy(stream_);
        throw CudaError("Failed to set cuSPARSE stream");
    }
}

GPUContext::~GPUContext() {
    cusparseDestroy(cusparse_handle_);
    cudaStreamDestroy(stream_);
}

// SparseMatrix implementation
SparseMatrix::SparseMatrix(
    const GPUContext& context, int rows, int cols, int nnz,
    const int* row_ptr, const int* col_indices, const float* values,
    bool dlpack_owned)
    : context_(context), rows_(rows), cols_(cols), nnz_(nnz), dlpack_owned_(dlpack_owned) {

    cudaError_t cuda_status;

    // TODO: Enable multi-GPU support
    // const int dst_device = context_.device_id();
    // int src_device = 0;
    // cudaPointerAttributes attr;

    // // Determine source device of input pointers
    // if (cudaPointerGetAttributes(&attr, row_ptr) == cudaSuccess) {
    //     #if CUDART_VERSION >= 10000
    //             src_device = attr.device;
    //     #else
    //             src_device = attr.device;
    //     #endif
    //     }

    //     // Enable P2P access if needed
    //     if (src_device != dst_device) {
    //         enable_peer_access(src_device, dst_device);
    //     }

    // // Set destination device context for allocation and stream copy
    // cudaSetDevice(dst_device);

    // Store the device pointers
    d_row_ptr_ = const_cast<int*>(row_ptr);
    d_col_indices_ = const_cast<int*>(col_indices);
    d_values_ = const_cast<float*>(values);

    // Copy data to device if in multi-GPUs context
    if (context_.device_id() != 0) {
        cuda_status = cudaMemcpyAsync(d_row_ptr_, row_ptr, (rows + 1) * sizeof(int),
                                    cudaMemcpyDeviceToDevice, context_.stream());
        if (cuda_status != cudaSuccess) {
            cudaFree(d_values_);
            cudaFree(d_col_indices_);
            cudaFree(d_row_ptr_);
            throw CudaError("Failed to copy row_ptr data");
        }

        cuda_status = cudaMemcpyAsync(d_col_indices_, col_indices, nnz * sizeof(int),
                                    cudaMemcpyDeviceToDevice, context_.stream());
        if (cuda_status != cudaSuccess) {
            cudaFree(d_values_);
            cudaFree(d_col_indices_);
            cudaFree(d_row_ptr_);
            throw CudaError("Failed to copy col_indices data");
        }

        cuda_status = cudaMemcpyAsync(d_values_, values, nnz * sizeof(float),
                                    cudaMemcpyDeviceToDevice, context_.stream());
        if (cuda_status != cudaSuccess) {
            cudaFree(d_values_);
            cudaFree(d_col_indices_);
            cudaFree(d_row_ptr_);
            throw CudaError("Failed to copy values data");
        }
    }

    // Create cuSPARSE matrix descriptor
    cusparseStatus_t cusparse_status = cusparseCreateCsr(
        &cusparse_mat_, rows, cols, nnz,
        d_row_ptr_, d_col_indices_, d_values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        // If the DLPack tensor owns the data, we don't free it here
        if (dlpack_owned_)
            return;

        cudaFree(d_values_);
        cudaFree(d_col_indices_);
        cudaFree(d_row_ptr_);
        throw CudaError("Failed to create cuSPARSE CSR matrix");
    }
}

SparseMatrix::~SparseMatrix() {
    cusparseDestroySpMat(cusparse_mat_);

    // If the DLPack tensor owns the data, we don't free it here
    if (dlpack_owned_)
        return;

    // Free device memory
    cudaFree(d_values_);
    cudaFree(d_col_indices_);
    cudaFree(d_row_ptr_);
}

// DeviceVector implementation
DeviceVector::DeviceVector(const GPUContext& context, int size)
    : context_(context), size_(size) {

    cudaError_t cuda_status = cudaMalloc(&d_data_, size * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to allocate device vector memory");
    }

    // Initialize to zero
    cuda_status = cudaMemsetAsync(d_data_, 0, size * sizeof(float), context_.stream());
    if (cuda_status != cudaSuccess) {
        cudaFree(d_data_);
        throw CudaError("Failed to initialize device vector");
    }

    // Create cuSPARSE vector descriptor
    cusparseStatus_t cusparse_status = cusparseCreateDnVec(
        &cusparse_vec_, size, d_data_, CUDA_R_32F);

    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(d_data_);
        throw CudaError("Failed to create cuSPARSE dense vector");
    }
}

DeviceVector::DeviceVector(const GPUContext& context, int size, const float* host_data)
    : context_(context), size_(size) {

    cudaError_t cuda_status = cudaMalloc(&d_data_, size * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to allocate device vector memory");
    }

    cuda_status = cudaMemcpyAsync(d_data_, host_data, size * sizeof(float),
                              cudaMemcpyHostToDevice, context_.stream());
    if (cuda_status != cudaSuccess) {
        cudaFree(d_data_);
        throw CudaError("Failed to copy vector data");
    }

    // Create cuSPARSE vector descriptor
    cusparseStatus_t cusparse_status = cusparseCreateDnVec(
        &cusparse_vec_, size, d_data_, CUDA_R_32F);

    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(d_data_);
        throw CudaError("Failed to create cuSPARSE dense vector");
    }
}

DeviceVector::~DeviceVector() {
    cusparseDestroyDnVec(cusparse_vec_);
    cudaFree(d_data_);
}

void DeviceVector::CopyFromHost(const float* host_data) {
    cudaError_t cuda_status = cudaMemcpyAsync(
        d_data_, host_data, size_ * sizeof(float),
        cudaMemcpyHostToDevice, context_.stream());

    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to copy data from host to device vector");
    }
}

void DeviceVector::CopyToHost(float* host_data) const {
    cudaError_t cuda_status = cudaMemcpyAsync(
        host_data, d_data_, size_ * sizeof(float),
        cudaMemcpyDeviceToHost, context_.stream());

    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to copy data from device vector to host");
    }
}

// Kernel for the PGS iteration
__global__ void pgsIterationKernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    float* __restrict__ x,
    const float* __restrict__ b,
    const float* __restrict__ lo,
    const float* __restrict__ hi,
    float relaxation,
    int num_rows) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // Compute Ax_i excluding the diagonal
    float diagonal = 0.0f;
    float ax_i = 0.0f;

    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
        int col = col_indices[j];
        float val = values[j];

        if (col == row) {
            diagonal = val;
        } else {
            ax_i += val * x[col];
        }
    }

    if (diagonal == 0.0f) return; // Skip rows with zero diagonal

    // Compute new x_i
    float new_x = (b[row] - ax_i) / diagonal;

    // Apply relaxation
    if (relaxation != 1.0f) {
        new_x = (1.0f - relaxation) * x[row] + relaxation * new_x;
    }

    // Project to bounds
    if (new_x < lo[row]) new_x = lo[row];
    if (new_x > hi[row]) new_x = hi[row];

    // Update solution
    x[row] = new_x;
}

// Kernel to compute residual
__global__ void computeResidualKernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    const float* __restrict__ b,
    float* __restrict__ residual_vec,
    int num_rows) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    float ax_i = 0.0f;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; ++j) {
        int col = col_indices[j];
        ax_i += values[j] * x[col];
    }

    float res = b[row] - ax_i;
    residual_vec[row] = res * res; // Square for L2 norm
}

// PGSSolver implementation
PGSSolver::PGSSolver(const PGSSolverConfig& config)
    : config_(config), iterations_(0), residual_(0.0f) {

    // Get the number of GPUs
    int num_gpus = 0;
    cudaError_t status = cudaGetDeviceCount(&num_gpus);
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to get device count: " +
                                 std::string(cudaGetErrorString(status)));
    }

    if (num_gpus == 0) {
        throw std::invalid_argument("No GPU devices found");
    }

    // Initialize GPU contexts for each detected GPU
    for (int device_id = 0; device_id < num_gpus; ++device_id) {
        try {
            contexts_.push_back(std::make_unique<GPUContext>(device_id));
        } catch (const CudaError& e) {
            std::cerr << "Failed to initialize GPU " << device_id << ": " << e.what() << std::endl;
            throw;
        }
    }
}

PGSSolver::~PGSSolver() = default;

SolverStatus PGSSolver::Solve(
    const std::vector<SparseMatrix*>& A_blocks,
    DeviceVector* x,
    const DeviceVector* b,
    const DeviceVector* lo,
    const DeviceVector* hi) {

    if (A_blocks.empty() || !x || !b || !lo || !hi) {
        throw std::invalid_argument("Invalid input to PGS solver");
    }

    if (A_blocks.size() != contexts_.size()) {
        throw std::invalid_argument("Number of matrix blocks must match number of GPUs");
    }

    // Check dimensions
    int num_rows = A_blocks[0]->rows();
    if (x->size() != num_rows || b->size() != num_rows ||
        lo->size() != num_rows || hi->size() != num_rows) {
        throw std::invalid_argument("Dimension mismatch in PGS solver inputs");
    }

    // Create residual vector for convergence check
    auto residual_vec = std::make_unique<DeviceVector>(*contexts_[0], num_rows);
    float* h_residual_vec = new float[num_rows];

    // Iteration loop
    float residual = std::numeric_limits<float>::max();
    int iter;

    for (iter = 0; iter < config_.max_iterations; ++iter) {
        // Launch PGS iteration kernel on each GPU
        for (size_t i = 0; i < contexts_.size(); ++i) {
            const GPUContext& context = *contexts_[i];
            const SparseMatrix* A = A_blocks[i];

            // Set device
            cudaSetDevice(context.device_id());

            // Launch kernel
            int block_size = 256;
            int grid_size = (num_rows + block_size - 1) / block_size;

            pgsIterationKernel<<<grid_size, block_size, 0, context.stream()>>>(
                A->row_ptr(),
                A->col_indices(),
                A->values(),
                x->data(),
                b->data(),
                lo->data(),
                hi->data(),
                config_.relaxation,
                num_rows);

            // Check for kernel launch errors
            cudaError_t cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                delete[] h_residual_vec;
                throw CudaError("PGS kernel launch failed: " +
                                 std::string(cudaGetErrorString(cuda_status)));
            }
        }

        // Synchronize all GPUs
        for (const auto& context : contexts_) {
            cudaSetDevice(context->device_id());
            cudaStreamSynchronize(context->stream());
        }

        // Check convergence every few iterations to avoid overhead
        if (iter % 10 == 0 || iter == config_.max_iterations - 1) {
            // Reuse first GPU for residual computation
            const GPUContext& context = *contexts_[0];
            const SparseMatrix* A = A_blocks[0];

            cudaSetDevice(context.device_id());

            int block_size = 256;
            int grid_size = (num_rows + block_size - 1) / block_size;

            computeResidualKernel<<<grid_size, block_size, 0, context.stream()>>>(
                A->row_ptr(),
                A->col_indices(),
                A->values(),
                x->data(),
                b->data(),
                residual_vec->data(),
                num_rows);

            // Copy residual vector to host
            residual_vec->CopyToHost(h_residual_vec);
            cudaStreamSynchronize(context.stream());

            // Compute L2 norm of residual
            residual = 0.0f;
            for (int i = 0; i < num_rows; ++i) {
                residual += h_residual_vec[i];
            }
            residual = std::sqrt(residual);

            if (config_.verbose && (iter % 100 == 0 || iter == config_.max_iterations - 1)) {
                std::cout << "Iteration " << iter << ", residual: " << residual << std::endl;
            }

            // Check convergence
            if (residual < config_.tolerance) {
                break;
            }
        }
    }

    // Store results
    iterations_ = iter + 1;
    residual_ = residual;

    delete[] h_residual_vec;

    if (residual < config_.tolerance) {
        return SolverStatus::SUCCESS;
    } else {
        return SolverStatus::MAX_ITERATIONS_REACHED;
    }
}

} // namespace cuda_pgs
