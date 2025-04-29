cmake_minimum_required(VERSION 3.18)
project(pgs_solver LANGUAGES CXX CUDA)

# Set policies for CMake to use FindCUDA if available
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Set GNUInstallDirs to use the standard installation directories
include(GNUInstallDirs)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(PGS_BUILD_EXAMPLES "Build example applications" ON)
option(PGS_BUILD_PYTHON "Build Python bindings" ON)

# Include directories
include_directories(src)
find_package(dlpack REQUIRED)

# Find CUDA and cuSPARSE
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Main library
add_library(cuda_pgs STATIC
    src/pgs_solver.cu
    src/dlpack_integration.cu
)
target_link_libraries(cuda_pgs
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    dlpack::dlpack
)

# Enable PIC for the static library
set_target_properties(cuda_pgs PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS cuda_pgs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Examples
if(PGS_BUILD_EXAMPLES)
    add_executable(poisson_cuda_example examples/poisson_cuda_example.cu)
    target_link_libraries(poisson_cuda_example
        cuda_pgs
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
        dlpack::dlpack
    )
    set_target_properties(poisson_cuda_example PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/examples
    )
endif()

# Python bindings
if(PGS_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(_pgs_solver src/python_bindings.cu)
    target_link_libraries(_pgs_solver PRIVATE cuda_pgs)

    pybind11_extension(_pgs_solver)
    pybind11_strip(_pgs_solver)

    # Install the Python extension module
    install(TARGETS _pgs_solver
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/pgs_solver
    )
endif()
