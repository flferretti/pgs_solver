#include "pgs_solver.cuh"
#include <cstring>
#include <stdexcept>

namespace cuda_pgs {

// DLPack conversion implementations
SparseMatrix::SparseMatrix(const GPUContext& context, DLManagedTensor* dl_tensor)
    : context_(context) {

    if (!dl_tensor || !dl_tensor->dl_tensor.data) {
        throw std::invalid_argument("Invalid DLPack tensor");
    }

    // Verify that the tensor is a CUDA tensor
    if (dl_tensor->dl_tensor.device.device_type != kDLCUDA) {
        throw std::invalid_argument("DLPack tensor must be a CUDA tensor");
    }

    // Verify that this is a sparse CSR matrix with 3 arrays
    if (dl_tensor->dl_tensor.ndim != 2 || dl_tensor->dl_tensor.shape[0] != 3) {
        throw std::invalid_argument("Invalid sparse matrix format in DLPack tensor");
    }

    // Extract matrix dimensions from the DLPack tensor
    // Assuming the DLPack tensor contains metadata about the sparse matrix
    rows_ = static_cast<int>(dl_tensor->dl_tensor.shape[1]);
    cols_ = static_cast<int>(dl_tensor->dl_tensor.shape[2]);

    // We need to extract row_ptr, col_indices, and values arrays
    // TODO: This assumes a specific layout in the DLPack tensor

    // For simplicity, assume the DLPack tensor has indptr, indices, and data arrays
    // consecutively stored in the tensor data
    char* data_ptr = static_cast<char*>(dl_tensor->dl_tensor.data);

    // Extract row_ptr (indptr)
    int* h_row_ptr = reinterpret_cast<int*>(data_ptr);
    int row_ptr_size = rows_ + 1;

    // Extract nnz from the last element of row_ptr
    nnz_ = h_row_ptr[rows_];

    // Extract col_indices (indices)
    int* h_col_indices = h_row_ptr + row_ptr_size;

    // Extract values (data)
    float* h_values = reinterpret_cast<float*>(h_col_indices + nnz_);

    // Allocate device memory
    cudaError_t cuda_status;

    cuda_status = cudaMalloc(&d_row_ptr_, row_ptr_size * sizeof(int));
    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to allocate row_ptr memory");
    }

    cuda_status = cudaMalloc(&d_col_indices_, nnz_ * sizeof(int));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_row_ptr_);
        throw CudaError("Failed to allocate col_indices memory");
    }

    cuda_status = cudaMalloc(&d_values_, nnz_ * sizeof(float));
    if (cuda_status != cudaSuccess) {
        cudaFree(d_col_indices_);
        cudaFree(d_row_ptr_);
        throw CudaError("Failed to allocate values memory");
    }

    // Copy data from default GPU to the current GPU in multi-GPU setup
    if (context_.device_id() != dl_tensor->dl_tensor.device.device_id) {
        // Copy data to device
        cuda_status = cudaMemcpyAsync(d_row_ptr_, h_row_ptr, row_ptr_size * sizeof(int),
                                     cudaMemcpyDeviceToDevice, context_.stream());
        if (cuda_status != cudaSuccess) {
            cudaFree(d_values_);
            cudaFree(d_col_indices_);
            cudaFree(d_row_ptr_);
            throw CudaError("Failed to copy row_ptr data");
        }

        cuda_status = cudaMemcpyAsync(d_col_indices_, h_col_indices, nnz_ * sizeof(int),
                                     cudaMemcpyDeviceToDevice, context_.stream());
        if (cuda_status != cudaSuccess) {
            cudaFree(d_values_);
            cudaFree(d_col_indices_);
            cudaFree(d_row_ptr_);
            throw CudaError("Failed to copy col_indices data");
        }

        cuda_status = cudaMemcpyAsync(d_values_, h_values, nnz_ * sizeof(float),
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
        &cusparse_mat_, rows_, cols_, nnz_,
        d_row_ptr_, d_col_indices_, d_values_,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(d_values_);
        cudaFree(d_col_indices_);
        cudaFree(d_row_ptr_);
        throw CudaError("Failed to create cuSPARSE CSR matrix");
    }
}

// DLPack deleter function for tensors
void DLPackDeleter(DLManagedTensor* self) {
    if (self) {
        // Free the data and the tensor struct
        delete[] static_cast<char*>(self->dl_tensor.data);
        delete self;
    }
}

DLManagedTensor* SparseMatrix::ToDLPack() const {
    // Create a new DLManagedTensor
    DLManagedTensor* dl_tensor = new DLManagedTensor();

    // Set up the DLTensor
    dl_tensor->dl_tensor.data = nullptr;  // Will set later
    dl_tensor->dl_tensor.ndim = 2;
    dl_tensor->dl_tensor.shape = new int64_t[2];
    dl_tensor->dl_tensor.shape[0] = 3;  // For row_ptr, col_indices, values
    dl_tensor->dl_tensor.shape[1] = rows_;
    dl_tensor->dl_tensor.strides = nullptr;  // Assume compact layout
    dl_tensor->dl_tensor.byte_offset = 0;

    // Set the context to CUDA
    dl_tensor->dl_tensor.device.device_type = kDLCUDA;
    dl_tensor->dl_tensor.device.device_id = context_.device_id();

    // Set the data type to float32
    dl_tensor->dl_tensor.dtype.code = kDLFloat;
    dl_tensor->dl_tensor.dtype.bits = 32;
    dl_tensor->dl_tensor.dtype.lanes = 1;

    // Calculate sizes
    size_t row_ptr_size = (rows_ + 1) * sizeof(int);
    size_t col_indices_size = nnz_ * sizeof(int);
    size_t values_size = nnz_ * sizeof(float);
    size_t total_size = row_ptr_size + col_indices_size + values_size;

    // Allocate memory for the data
    char* data = new char[total_size];
    dl_tensor->dl_tensor.data = data;

    // Copy the data to the DLPack tensor
    cudaError_t cuda_status;

    cuda_status = cudaMemcpyAsync(data, d_row_ptr_, row_ptr_size,
                                    cudaMemcpyDeviceToHost, context_.stream());
    if (cuda_status != cudaSuccess) {
        delete[] data;
        delete[] dl_tensor->dl_tensor.shape;
        delete dl_tensor;
        throw CudaError("Failed to copy row_ptr to DLPack tensor");
    }

    cuda_status = cudaMemcpyAsync(data + row_ptr_size, d_col_indices_, col_indices_size,
                                    cudaMemcpyDeviceToHost, context_.stream());
    if (cuda_status != cudaSuccess) {
        delete[] data;
        delete[] dl_tensor->dl_tensor.shape;
        delete dl_tensor;
        throw CudaError("Failed to copy col_indices to DLPack tensor");
    }

    cuda_status = cudaMemcpyAsync(data + row_ptr_size + col_indices_size, d_values_, values_size,
                                    cudaMemcpyDeviceToHost, context_.stream());
    if (cuda_status != cudaSuccess) {
        delete[] data;
        delete[] dl_tensor->dl_tensor.shape;
        delete dl_tensor;
        throw CudaError("Failed to copy values to DLPack tensor");
    }

    // Synchronize to ensure the copy is complete
    cudaStreamSynchronize(context_.stream());

    // Set up the deleter
    dl_tensor->deleter = DLPackDeleter;
    dl_tensor->manager_ctx = nullptr;

    return dl_tensor;
}

// DeviceVector from DLPack tensor
DeviceVector::DeviceVector(const GPUContext& context, DLManagedTensor* dl_tensor)
    : context_(context) {

    if (!dl_tensor || !dl_tensor->dl_tensor.data) {
        throw std::invalid_argument("Invalid DLPack tensor");
    }

    // Verify that the tensor is a CUDA tensor
    if (dl_tensor->dl_tensor.device.device_type != kDLCUDA) {
        throw std::invalid_argument("DLPack tensor must be a CUDA tensor");
    }

    // Verify that this is a vector (1D tensor)
    if (dl_tensor->dl_tensor.ndim != 1) {
        throw std::invalid_argument("DLPack tensor must be a 1D tensor for DeviceVector");
    }

    // Extract size
    size_ = static_cast<int>(dl_tensor->dl_tensor.shape[0]);

    // Check data type (we support float32 only)
    if (dl_tensor->dl_tensor.dtype.code != kDLFloat || dl_tensor->dl_tensor.dtype.bits != 32) {
        throw std::invalid_argument("DeviceVector only supports float32 data type");
    }

    // Allocate memory
    cudaError_t cuda_status = cudaMalloc(&d_data_, size_ * sizeof(float));
    if (cuda_status != cudaSuccess) {
        throw CudaError("Failed to allocate device vector memory");
    }

    // Copy data from the DLPack tensor
    cuda_status = cudaMemcpyAsync(
        d_data_, dl_tensor->dl_tensor.data, size_ * sizeof(float),
        cudaMemcpyDeviceToDevice, context_.stream());

    if (cuda_status != cudaSuccess) {
        cudaFree(d_data_);
        throw CudaError("Failed to copy data from DLPack tensor");
    }

    // Create cuSPARSE vector descriptor
    cusparseStatus_t cusparse_status = cusparseCreateDnVec(
        &cusparse_vec_, size_, d_data_, CUDA_R_32F);

    if (cusparse_status != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(d_data_);
        throw CudaError("Failed to create cuSPARSE dense vector");
    }
}

DLManagedTensor* DeviceVector::ToDLPack() const {
    // Create a new DLManagedTensor
    DLManagedTensor* dl_tensor = new DLManagedTensor();

    // Set up the DLTensor
    dl_tensor->dl_tensor.data = nullptr;  // Will set later
    dl_tensor->dl_tensor.ndim = 1;
    dl_tensor->dl_tensor.shape = new int64_t[1];
    dl_tensor->dl_tensor.shape[0] = size_;
    dl_tensor->dl_tensor.strides = nullptr;  // Assume compact layout
    dl_tensor->dl_tensor.byte_offset = 0;

    // Set the context to CUDA
    dl_tensor->dl_tensor.device.device_type = kDLCUDA;
    dl_tensor->dl_tensor.device.device_id = context_.device_id();

    // Set the data type to float32
    dl_tensor->dl_tensor.dtype.code = kDLFloat;
    dl_tensor->dl_tensor.dtype.bits = 32;
    dl_tensor->dl_tensor.dtype.lanes = 1;

    // Allocate memory for the data (on host for transfer)
    float* h_data = new float[size_];

    // Copy the data from device to host
    cudaError_t cuda_status = cudaMemcpyAsync(
        h_data, d_data_, size_ * sizeof(float),
        cudaMemcpyDeviceToHost, context_.stream());

    if (cuda_status != cudaSuccess) {
        delete[] h_data;
        delete[] dl_tensor->dl_tensor.shape;
        delete dl_tensor;
        throw CudaError("Failed to copy data from device to host");
    }

    // Synchronize to ensure the copy is complete
    cudaStreamSynchronize(context_.stream());

    // Set the data pointer
    dl_tensor->dl_tensor.data = h_data;

    // Set up the deleter
    dl_tensor->deleter = DLPackDeleter;
    dl_tensor->manager_ctx = nullptr;

    return dl_tensor;
}

// Implementation of DLPack-based solver method
SolverStatus PGSSolver::SolveDLPack(
    DLManagedTensor** A_blocks, int num_blocks,
    DLManagedTensor* x_tensor,
    DLManagedTensor* b_tensor,
    DLManagedTensor* lo_tensor,
    DLManagedTensor* hi_tensor) {

    if (num_blocks <= 0 || !A_blocks || !x_tensor || !b_tensor || !lo_tensor || !hi_tensor) {
        throw std::invalid_argument("Invalid input to SolveDLPack");
    }

    if (static_cast<size_t>(num_blocks) != contexts_.size()) {
        throw std::invalid_argument("Number of matrix blocks must match number of GPUs");
    }

    // Convert DLPack tensors to our internal formats
    std::vector<std::unique_ptr<SparseMatrix>> A_matrices;
    for (int i = 0; i < num_blocks; ++i) {
        A_matrices.push_back(dlpack_utils::DLPackToSparseMatrix(*contexts_[i], A_blocks[i]));
    }

    // Convert vector tensors
    auto x = dlpack_utils::DLPackToDeviceVector(*contexts_[0], x_tensor);
    auto b = dlpack_utils::DLPackToDeviceVector(*contexts_[0], b_tensor);
    auto lo = dlpack_utils::DLPackToDeviceVector(*contexts_[0], lo_tensor);
    auto hi = dlpack_utils::DLPackToDeviceVector(*contexts_[0], hi_tensor);

    // Prepare pointers for the solver
    std::vector<SparseMatrix*> A_ptrs;
    for (const auto& A : A_matrices) {
        A_ptrs.push_back(A.get());
    }

    // Call the solver
    SolverStatus status = Solve(A_ptrs, x.get(), b.get(), lo.get(), hi.get());

    // Update the x_tensor with the solution
    // This happens automatically since we're working on the same memory

    return status;
}

namespace dlpack_utils {

    std::unique_ptr<SparseMatrix> DLPackToSparseMatrix(const GPUContext& context, DLManagedTensor* dl_tensor) {
        return std::make_unique<SparseMatrix>(context, dl_tensor);
    }

    std::unique_ptr<DeviceVector> DLPackToDeviceVector(const GPUContext& context, DLManagedTensor* dl_tensor) {
        return std::make_unique<DeviceVector>(context, dl_tensor);
    }

    DLManagedTensor* SparseMatrixToDLPack(const SparseMatrix& matrix) {
        return matrix.ToDLPack();
    }

    DLManagedTensor* DeviceVectorToDLPack(const DeviceVector& vector) {
        return vector.ToDLPack();
    }

} // namespace dlpack_utils

} // namespace cuda_pgs