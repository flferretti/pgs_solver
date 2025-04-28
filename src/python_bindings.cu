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

PYBIND11_MODULE(pgs_solver, m) {
    m.doc() = "CUDA-based Projected Gauss-Seidel solver with multi-GPU support";

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
        .def(py::init<>())
        .def_readwrite("max_iterations", &cuda_pgs::PGSSolverConfig::max_iterations)
        .def_readwrite("tolerance", &cuda_pgs::PGSSolverConfig::tolerance)
        .def_readwrite("relaxation", &cuda_pgs::PGSSolverConfig::relaxation)
        .def_readwrite("verbose", &cuda_pgs::PGSSolverConfig::verbose);

    // Bind PGSSolver class
    py::class_<cuda_pgs::PGSSolver>(m, "PGSSolver")
        .def(py::init<const std::vector<int>&, const cuda_pgs::PGSSolverConfig&>(),
             py::arg("gpu_ids"), py::arg("config") = cuda_pgs::PGSSolverConfig())
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
            return self.SolveDLPack(A_tensors.data(), A_tensors.size(),
                                   x_tensor, b_tensor, lo_tensor, hi_tensor);
        })
        .def_property_readonly("iterations", &cuda_pgs::PGSSolver::iterations)
        .def_property_readonly("residual", &cuda_pgs::PGSSolver::residual);
}