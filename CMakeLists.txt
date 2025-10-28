cmake_minimum_required(VERSION 3.18)
project(pgs_solver LANGUAGES CXX)

# Set policies for CMake to use FindCUDA if available
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# Set GNUInstallDirs to use the standard installation directories
include(GNUInstallDirs)

# C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options
option(PGS_BUILD_EXAMPLES "Build example applications" ON)
option(PGS_BUILD_PYTHON "Build Python bindings" ON)
option(PGS_USE_ROCM "Use ROCm/HIP instead of CUDA (experimental)" OFF)

# GPU backend selection
if(PGS_USE_ROCM)
    message(STATUS "Building with ROCm/HIP support (experimental)")
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(rocsparse REQUIRED)

    set(GPU_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
    set(GPU_LIBRARIES hip::host roc::rocsparse)
    set(GPU_LANGUAGE HIP)
    set(GPU_STANDARD 14)

    # HIP source files (will use .hip extension or compile .cu as HIP)
    set(GPU_SOURCES
        src/pgs_solver.cu
        src/dlpack_integration.cu
    )

    # Add HIP-specific definitions
    add_definitions(-D__HIP_PLATFORM_AMD__)
    add_definitions(-DPGS_USE_ROCM)
else()
    message(STATUS "Building with CUDA support")
    enable_language(CUDA)
    find_package(CUDA REQUIRED)

    set(CMAKE_CUDA_STANDARD 14)
    set(GPU_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
    set(GPU_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY})
    set(GPU_LANGUAGE CUDA)

    set(GPU_SOURCES
        src/pgs_solver.cu
        src/dlpack_integration.cu
    )
endif()

# Include directories
include_directories(src)
find_package(dlpack REQUIRED)
include_directories(${GPU_INCLUDE_DIRS})

# Main library
if(PGS_USE_ROCM)
    # For ROCm/HIP, use set_source_files_properties to mark as HIP
    set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    add_library(cuda_pgs STATIC ${GPU_SOURCES})
    target_link_libraries(cuda_pgs
        ${GPU_LIBRARIES}
        dlpack::dlpack
    )
else()
    # For CUDA, standard compilation
    add_library(cuda_pgs STATIC ${GPU_SOURCES})
    target_link_libraries(cuda_pgs
        ${GPU_LIBRARIES}
        dlpack::dlpack
    )
endif()

# Enable PIC for the static library
set_target_properties(cuda_pgs PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS cuda_pgs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# Examples
if(PGS_BUILD_EXAMPLES)
    if(PGS_USE_ROCM)
        set_source_files_properties(examples/poisson_cuda_example.cu PROPERTIES LANGUAGE HIP)
    endif()

    add_executable(poisson_cuda_example examples/poisson_cuda_example.cu)
    target_link_libraries(poisson_cuda_example
        cuda_pgs
        ${GPU_LIBRARIES}
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

    if(PGS_USE_ROCM)
        set_source_files_properties(src/python_bindings.cu PROPERTIES LANGUAGE HIP)
    endif()

    pybind11_add_module(_pgs_solver src/python_bindings.cu)
    target_link_libraries(_pgs_solver PRIVATE cuda_pgs)

    pybind11_extension(_pgs_solver)
    pybind11_strip(_pgs_solver)

    # Install the Python extension module
    install(TARGETS _pgs_solver
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
    )
endif()
