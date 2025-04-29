#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pgs_solver.cuh"

namespace py = pybind11;

// Wrapper for DLPack tensor conversion
static void* DLPackTensorToCapsule(DLManagedTensor* dl_tensor) {
    return static_cast<void*>(dl_tensor);
}

static DLManagedTensor* CapsuleToDLPackTensor(void* capsule) {
    return static_cast<DLManagedTensor*>(capsule);
}

// Create a DLManagedTensor from raw buffer for JAX integration
static DLManagedTensor* CreateDLTensorFromBuffer(void* data, int64_t* shape, int ndim,
                                                DLDataType dtype, DLDevice device) {
    DLManagedTensor* dl_tensor = new DLManagedTensor();

    // Set up the DLTensor
    dl_tensor->dl_tensor.data = data;
    dl_tensor->dl_tensor.ndim = ndim;
    dl_tensor->dl_tensor.shape = new int64_t[ndim];
    std::memcpy(dl_tensor->dl_tensor.shape, shape, ndim * sizeof(int64_t));
    dl_tensor->dl_tensor.strides = nullptr;  // Assume compact layout
    dl_tensor->dl_tensor.byte_offset = 0;
    dl_tensor->dl_tensor.dtype = dtype;
    dl_tensor->dl_tensor.device = device;

    // No deleter - JAX owns the memory
    dl_tensor->deleter = nullptr;
    dl_tensor->manager_ctx = nullptr;

    return dl_tensor;
}

// Function prototype for the custom call
extern "C" {
    void pgs_solver_custom_call(void** out, const void** in);
}

// Implementation of the custom call for JAX
void pgs_solver_custom_call(void** out, const void** in) {
    try {
        // Input structure:
        // in[0]: Packed parameters (max_iterations, tolerance, relaxation, verbose)
        // in[1]: Number of A matrices (for multi-GPU)
        // in[2:2+num_matrices*3]: CSR matrices data (each has row_ptr, col_indices, values)
        // in[last-3]: x vector (initial guess)
        // in[last-2]: b vector (RHS)
        // in[last-1]: lo vector (lower bounds)
        // in[last]: hi vector (upper bounds)

        // Extract solver parameters
        const float* config_data = static_cast<const float*>(in[0]);
        int max_iterations = static_cast<int>(config_data[0]);
        float tolerance = config_data[1];
        float relaxation = config_data[2];
        bool verbose = static_cast<bool>(static_cast<int>(config_data[3]));

        // Configure solver
        cuda_pgs::PGSSolverConfig config;
        config.max_iterations = max_iterations;
        config.tolerance = tolerance;
        config.relaxation = relaxation;
        config.verbose = verbose;

        // Create solver
        cuda_pgs::PGSSolver solver(config);

        // Extract number of matrices
        const int* num_matrices_ptr = static_cast<const int*>(in[1]);
        int num_matrices = *num_matrices_ptr;

        // Create GPU context for default device (device 0)
        cuda_pgs::GPUContext context(0);

        // Process matrices
        std::vector<DLManagedTensor*> A_tensors;
        int input_idx = 2;

        for (int i = 0; i < num_matrices; i++) {
            // For each matrix, we have row_ptr, col_indices, values
            const int* row_ptr = static_cast<const int*>(in[input_idx++]);
            const int* col_indices = static_cast<const int*>(in[input_idx++]);
            const float* values = static_cast<const float*>(in[input_idx++]);

            // Extract matrix dimensions
            int num_rows = static_cast<int>(row_ptr[0]);
            int num_cols = static_cast<int>(row_ptr[1]);
            int nnz = row_ptr[num_rows];

            // Create a sparse matrix using SparseMatrix constructor
            // This wraps the raw pointer into a proper matrix object
            cuda_pgs::SparseMatrix* sparse_matrix = new cuda_pgs::SparseMatrix(
                context, num_rows, num_cols, nnz, row_ptr, col_indices, values
            );

            // Convert to DLPack tensor
            DLManagedTensor* dl_tensor = sparse_matrix->ToDLPack();
            A_tensors.push_back(dl_tensor);
        }

        // Get device vectors for x, b, lo, hi
        const float* x_data = static_cast<const float*>(in[input_idx++]);
        const float* b_data = static_cast<const float*>(in[input_idx++]);
        const float* lo_data = static_cast<const float*>(in[input_idx++]);
        const float* hi_data = static_cast<const float*>(in[input_idx++]);

        // Get vector size from metadata (assuming first element contains size)
        int vector_size = static_cast<int>(reinterpret_cast<const int*>(x_data)[0]);

        // Create device vectors
        cuda_pgs::DeviceVector* x_vec = new cuda_pgs::DeviceVector(context, vector_size, x_data);
        cuda_pgs::DeviceVector* b_vec = new cuda_pgs::DeviceVector(context, vector_size, b_data);
        cuda_pgs::DeviceVector* lo_vec = new cuda_pgs::DeviceVector(context, vector_size, lo_data);
        cuda_pgs::DeviceVector* hi_vec = new cuda_pgs::DeviceVector(context, vector_size, hi_data);

        // Convert to DLPack tensors
        DLManagedTensor* x_tensor = x_vec->ToDLPack();
        DLManagedTensor* b_tensor = b_vec->ToDLPack();
        DLManagedTensor* lo_tensor = lo_vec->ToDLPack();
        DLManagedTensor* hi_tensor = hi_vec->ToDLPack();

        // Call the solver
        cuda_pgs::SolverStatus status = solver.SolveDLPack(
            A_tensors.data(),
            A_tensors.size(),
            x_tensor,
            b_tensor,
            lo_tensor,
            hi_tensor
        );

        // Copy results to output
        // The solution (x) has already been updated in-place in the input/output buffer

        // Fill additional output information
        int* status_out = static_cast<int*>(out[1]);
        *status_out = static_cast<int>(status);

        int* iterations_out = static_cast<int*>(out[2]);
        *iterations_out = solver.iterations();

        float* residual_out = static_cast<float*>(out[3]);
        *residual_out = solver.residual();

        // Clean up
        for (auto tensor : A_tensors) {
            if (tensor->deleter) {
                tensor->deleter(tensor);
            }
        }

        // Clean up vector tensors
        if (x_tensor->deleter) x_tensor->deleter(x_tensor);
        if (b_tensor->deleter) b_tensor->deleter(b_tensor);
        if (lo_tensor->deleter) lo_tensor->deleter(lo_tensor);
        if (hi_tensor->deleter) hi_tensor->deleter(hi_tensor);

        // Clean up device vector objects
        delete x_vec;
        delete b_vec;
        delete lo_vec;
        delete hi_vec;

    } catch (const std::exception& e) {
        fprintf(stderr, "Error in PGS solver custom call: %s\n", e.what());
    }
}

// Function to get the custom call capsule for JAX
static py::capsule get_pgs_solver_capsule() {
    return py::capsule(reinterpret_cast<void*>(&pgs_solver_custom_call), "xla._CUSTOM_CALL_TARGET");
}

PYBIND11_MODULE(_pgs_solver, m) {
    m.doc() = "CUDA-based Projected Gauss-Seidel solver with multi-GPU support";

    // Add function to get JAX custom call capsule
    m.def("get_pgs_solver_capsule", &get_pgs_solver_capsule,
        "Returns a capsule containing the PGS solver custom call function for JAX integration");

    // Add function to create a SparseMatrix from DLPack tensors
    m.def("SparseMatrix_from_dlpack",
        [](cuda_pgs::GPUContext& context, int rows, int cols, int nnz,
            py::capsule indptr_dlpack, py::capsule indices_dlpack, py::capsule data_dlpack) {
            // Extract DLManaged tensors from capsules
            DLManagedTensor* indptr_tensor = static_cast<DLManagedTensor*>(indptr_dlpack.get_pointer());
            DLManagedTensor* indices_tensor = static_cast<DLManagedTensor*>(indices_dlpack.get_pointer());
            DLManagedTensor* data_tensor = static_cast<DLManagedTensor*>(data_dlpack.get_pointer());

            // Validate tensors are on GPU
            if (indptr_tensor->dl_tensor.device.device_type != kDLCUDA ||
                indices_tensor->dl_tensor.device.device_type != kDLCUDA ||
                data_tensor->dl_tensor.device.device_type != kDLCUDA) {
                throw std::runtime_error("All tensors must be CUDA tensors");
            }

            // Create the sparse matrix directly from the DLPack tensors
            // The implementation already handles DeviceToDevice copies
            return std::make_unique<cuda_pgs::SparseMatrix>(
                context,
                rows,
                cols,
                nnz,
                static_cast<int*>(indptr_tensor->dl_tensor.data),
                static_cast<int*>(indices_tensor->dl_tensor.data),
                static_cast<float*>(data_tensor->dl_tensor.data)
            );
        });

    // Create a wrapper class as a workaround for enum class
    struct SolverStatusWrapper {
        cuda_pgs::SolverStatus value;

        // Default constructor
        SolverStatusWrapper() : value(cuda_pgs::SolverStatus::SUCCESS) {}

        // Constructor
        SolverStatusWrapper(cuda_pgs::SolverStatus val) : value(val) {}

        // Comparison operators (if needed)
        bool operator==(const SolverStatusWrapper& rhs) const { return value == rhs.value; }
        bool operator!=(const SolverStatusWrapper& rhs) const { return value != rhs.value; }

        // Conversion operator (if needed)
        operator cuda_pgs::SolverStatus() const { return value; }
    };

    py::class_<SolverStatusWrapper>(m, "SolverStatus")
        .def(py::init<>())
        .def(py::init<cuda_pgs::SolverStatus>())
        .def_readonly("value", &SolverStatusWrapper::value)
        .def_property_readonly_static("SUCCESS", [](py::object) {
            return SolverStatusWrapper(cuda_pgs::SolverStatus::SUCCESS);
        })
        .def_property_readonly_static("MAX_ITERATIONS_REACHED", [](py::object) {
            return SolverStatusWrapper(cuda_pgs::SolverStatus::MAX_ITERATIONS_REACHED);
        })
        .def_property_readonly_static("DIVERGED", [](py::object) {
            return SolverStatusWrapper(cuda_pgs::SolverStatus::DIVERGED);
        })
        .def_property_readonly_static("FAILED", [](py::object) {
            return SolverStatusWrapper(cuda_pgs::SolverStatus::FAILED);
        });

    // Bind GPUContext class
    py::class_<cuda_pgs::GPUContext>(m, "GPUContext")
        .def(py::init<int>())
        .def_property_readonly("device_id", &cuda_pgs::GPUContext::device_id);

    // Bind SparseMatrix class
    py::class_<cuda_pgs::SparseMatrix>(m, "SparseMatrix")
        .def(py::init<const cuda_pgs::GPUContext&, int, int, int, const int*, const int*, const float*>())
        .def("__dlpack__", [](cuda_pgs::SparseMatrix& self) {
            return py::capsule(DLPackTensorToCapsule(self.ToDLPack()), "dltensor");
        })
        .def_static("from_dlpack", [](py::capsule capsule, const cuda_pgs::GPUContext& context) {
            DLManagedTensor* dl_tensor = CapsuleToDLPackTensor(capsule.get_pointer());
            return new cuda_pgs::SparseMatrix(context, dl_tensor);
        })
        .def_property_readonly("rows", &cuda_pgs::SparseMatrix::rows)
        .def_property_readonly("cols", &cuda_pgs::SparseMatrix::cols)
        .def_property_readonly("nnz", &cuda_pgs::SparseMatrix::nnz);

    // Bind DeviceVector class
    py::class_<cuda_pgs::DeviceVector>(m, "DeviceVector")
        .def(py::init<const cuda_pgs::GPUContext&, int>())
        .def(py::init<const cuda_pgs::GPUContext&, int, const float*>())
        .def("__dlpack__", [](cuda_pgs::DeviceVector& self) {
            return py::capsule(DLPackTensorToCapsule(self.ToDLPack()), "dltensor");
        })
        .def_static("from_dlpack", [](py::capsule capsule, const cuda_pgs::GPUContext& context) {
            DLManagedTensor* dl_tensor = CapsuleToDLPackTensor(capsule.get_pointer());
            return new cuda_pgs::DeviceVector(context, dl_tensor);
        })
        .def("copy_from_host", &cuda_pgs::DeviceVector::CopyFromHost)
        .def("copy_to_host", [](const cuda_pgs::DeviceVector& self, py::array_t<float> array) {
            if (array.size() != self.size()) {
                throw std::runtime_error("Array size mismatch");
            }
            self.CopyToHost(array.mutable_data());
        })
        .def_property_readonly("size", &cuda_pgs::DeviceVector::size);

    // Bind PGSSolverConfig struct
    py::class_<cuda_pgs::PGSSolverConfig>(m, "PGSSolverConfig")
    .def(py::init<int, float, float, bool>(),
         py::arg("max_iterations") = 1000,
         py::arg("tolerance") = 1e-6f,
         py::arg("relaxation") = 1.0f,
         py::arg("verbose") = false)
    .def_readwrite("max_iterations", &cuda_pgs::PGSSolverConfig::max_iterations)
    .def_readwrite("tolerance", &cuda_pgs::PGSSolverConfig::tolerance)
    .def_readwrite("relaxation", &cuda_pgs::PGSSolverConfig::relaxation)
    .def_readwrite("verbose", &cuda_pgs::PGSSolverConfig::verbose);

    // Bind PGSSolver class
    py::class_<cuda_pgs::PGSSolver>(m, "PGSSolver")
        .def(py::init<const cuda_pgs::PGSSolverConfig&>(),
             py::arg("config") = cuda_pgs::PGSSolverConfig())
        .def("solve", &cuda_pgs::PGSSolver::Solve)
        .def("solve_dlpack", [](cuda_pgs::PGSSolver& self,
                               std::vector<py::capsule>& A_capsules,
                               py::capsule x_capsule,
                               py::capsule b_capsule,
                               py::capsule lo_capsule,
                               py::capsule hi_capsule) {
            // Convert capsules to DLPack tensors
            std::vector<DLManagedTensor*> A_tensors;
            for (const auto& capsule : A_capsules) {
                A_tensors.push_back(CapsuleToDLPackTensor(capsule.get_pointer()));
            }

            DLManagedTensor* x_tensor = CapsuleToDLPackTensor(x_capsule.get_pointer());
            DLManagedTensor* b_tensor = CapsuleToDLPackTensor(b_capsule.get_pointer());
            DLManagedTensor* lo_tensor = CapsuleToDLPackTensor(lo_capsule.get_pointer());
            DLManagedTensor* hi_tensor = CapsuleToDLPackTensor(hi_capsule.get_pointer());

            // Call the DLPack-based solver
            cuda_pgs::SolverStatus status = self.SolveDLPack(
                A_tensors.data(),
                A_tensors.size(),
                x_tensor,
                b_tensor,
                lo_tensor,
                hi_tensor
            );

            // Wrap the enum in the wrapper class before returning
            return SolverStatusWrapper(status);
        })
        .def_property_readonly("iterations", &cuda_pgs::PGSSolver::iterations)
        .def_property_readonly("residual", &cuda_pgs::PGSSolver::residual);
}