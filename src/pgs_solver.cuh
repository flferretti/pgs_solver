#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <dlpack/dlpack.h>
#include <vector>
#include <memory>
#include <string>

namespace cuda_pgs {

// Error handling
class CudaError : public std::runtime_error {
public:
    explicit CudaError(const std::string& message) : std::runtime_error(message) {}
};

// Forward declarations
class GPUContext;
class SparseMatrix;
class PGSSolver;

// Enum for solver status
enum class SolverStatus {
    SUCCESS,
    MAX_ITERATIONS_REACHED,
    DIVERGED,
    FAILED
};

// GPU context class to manage CUDA resources
class GPUContext {
public:
    GPUContext(int device_id);
    ~GPUContext();

    cudaStream_t stream() const { return stream_; }
    cusparseHandle_t cusparse_handle() const { return cusparse_handle_; }
    int device_id() const { return device_id_; }

private:
    int device_id_;
    cudaStream_t stream_;
    cusparseHandle_t cusparse_handle_;
};

// Sparse matrix representation
class SparseMatrix {
public:
    // Create from CSR format
    SparseMatrix(const GPUContext& context, int rows, int cols,
                 int nnz, const int* row_ptr, const int* col_indices,
                 const float* values);

    // Create from DLPack tensor
    SparseMatrix(const GPUContext& context, DLManagedTensor* dl_tensor);

    ~SparseMatrix();

    // Convert to DLPack tensor
    DLManagedTensor* ToDLPack();

    // Getters
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int nnz() const { return nnz_; }
    const int* row_ptr() const { return d_row_ptr_; }
    const int* col_indices() const { return d_col_indices_; }
    const float* values() const { return d_values_; }

private:
    const GPUContext& context_;
    int rows_;
    int cols_;
    int nnz_;
    int* d_row_ptr_;
    int* d_col_indices_;
    float* d_values_;
    cusparseSpMatDescr_t cusparse_mat_;
};

// Vector wrapper for device memory
class DeviceVector {
public:
    DeviceVector(const GPUContext& context, int size);
    DeviceVector(const GPUContext& context, int size, const float* host_data);
    DeviceVector(const GPUContext& context, DLManagedTensor* dl_tensor);
    ~DeviceVector();

    // Convert to DLPack tensor
    DLManagedTensor* ToDLPack();

    // Getters/setters
    int size() const { return size_; }
    float* data() { return d_data_; }
    const float* data() const { return d_data_; }

    // Copy operations
    void CopyFromHost(const float* host_data);
    void CopyToHost(float* host_data) const;

private:
    const GPUContext& context_;
    int size_;
    float* d_data_;
    cusparseDnVecDescr_t cusparse_vec_;
};

// Configuration for the PGS solver
struct PGSSolverConfig {
    int max_iterations = 1000;
    float tolerance = 1e-6f;
    float relaxation = 1.0f;
    bool verbose = false;
};

// Multi-GPU PGS solver
class PGSSolver {
public:
    PGSSolver(const std::vector<int>& gpu_ids, const PGSSolverConfig& config = PGSSolverConfig());
    ~PGSSolver();

    // Solve Ax = b with constraints lo <= x <= hi
    SolverStatus Solve(
        const std::vector<SparseMatrix*>& A_blocks,
        DeviceVector* x,
        const DeviceVector* b,
        const DeviceVector* lo,
        const DeviceVector* hi);

    // Solve using DLPack tensors
    SolverStatus SolveDLPack(
        DLManagedTensor** A_blocks, int num_blocks,
        DLManagedTensor* x,
        DLManagedTensor* b,
        DLManagedTensor* lo,
        DLManagedTensor* hi);

    // Get iteration count from last solve
    int iterations() const { return iterations_; }

    // Get residual from last solve
    float residual() const { return residual_; }

private:
    std::vector<std::unique_ptr<GPUContext>> contexts_;
    PGSSolverConfig config_;
    int iterations_;
    float residual_;

    // Implementation details for multi-GPU execution
    void DistributeMatrix(const SparseMatrix& A);
    void GatherSolution(DeviceVector* x);
};

// DLPack utilities
namespace dlpack_utils {
    // Convert DLPack tensor to our internal formats
    std::unique_ptr<SparseMatrix> DLPackToSparseMatrix(const GPUContext& context, DLManagedTensor* dl_tensor);
    std::unique_ptr<DeviceVector> DLPackToDeviceVector(const GPUContext& context, DLManagedTensor* dl_tensor);

    // Create DLPack tensors from our internal formats
    DLManagedTensor* SparseMatrixToDLPack(const SparseMatrix& matrix);
    DLManagedTensor* DeviceVectorToDLPack(const DeviceVector& vector);
}

} // namespace cuda_pgs
