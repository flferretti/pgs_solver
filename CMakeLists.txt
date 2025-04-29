cmake_minimum_required(VERSION 3.18)
project(pgs_solver LANGUAGES CXX CUDA)

# Set policies for CMake to use FindCUDA if available
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

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
add_library(pgs_solver SHARED
    src/pgs_solver.cu
    src/dlpack_integration.cu
)
target_link_libraries(pgs_solver
    ${CUDA_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    dlpack::dlpack
)

# Examples
if(PGS_BUILD_EXAMPLES)
    # Poisson equation solver example
    add_executable(poisson_cuda_example examples/poisson_cuda_example.cu)
    target_link_libraries(poisson_cuda_example
        pgs_solver
        ${CUDA_LIBRARIES}
        ${CUDA_cusparse_LIBRARY}
    )
endif()

# Python bindings
if(PGS_BUILD_PYTHON)
    # Python bindings configuration would go here
    # This requires finding pybind11 and setting up the Python module
endif()

# Installation rules
install(TARGETS pgs_solver
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(FILES src/pgs_solver.cuh DESTINATION include/pgs_solver)
