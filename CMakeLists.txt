cmake_minimum_required(VERSION 3.18)
project(pgs_solver LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "$ENV{CUDA_NVCC_FLAGS} -Xcompiler -fPIC")
# Automatically set the CUDA architecture (works with CMake 3.18+)
# if not DEFINED CMAKE_CUDA_ARCHITECTURES
#     message(STATUS "CMake version is 3.18 or higher, setting CUDA architecture to 'auto'.")
#     set(CMAKE_CUDA_ARCHITECTURES "auto")
# else()
#     message(STATUS "CMake version is lower than 3.18, using user-defined CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}.")

    # Pybind11 settings
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Include DLPack
# TODO: Make this smarter
find_path(DLPACK_INCLUDE_DIR NAMES dlpack.h PATHS $ENV{CONDA_PREFIX}/include)
include_directories(${DLPACK_INCLUDE_DIR})

# Include paths
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/external/dlpack/include
)

# Static library with CUDA code
add_library(cuda_pgs STATIC
    src/pgs_solver.cu
    src/dlpack_integration.cu
)

# Enable position independent code (PIC) for the static library
set_target_properties(cuda_pgs PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Pybind11 Python module
pybind11_add_module(pgs_solver src/python_bindings.cu)
target_link_libraries(pgs_solver PRIVATE cuda_pgs)

# Make pgs_solver a real Python extension
pybind11_extension(pgs_solver)
pybind11_strip(pgs_solver)

# Installation
include(GNUInstallDirs)

install(TARGETS cuda_pgs
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(TARGETS pgs_solver
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages)

install(DIRECTORY include/ DESTINATION include)
