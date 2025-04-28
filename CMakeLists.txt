cmake_minimum_required(VERSION 3.18)
project(pgs_solver LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_NVCC_FLAGS "$ENV{CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

# Pybind11 settings
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Include DLPack
find_package(dlpack REQUIRED)

find_package(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

# Static library with CUDA code
add_library(cuda_pgs STATIC
    src/pgs_solver.cu
    src/dlpack_integration.cu
)

target_link_libraries(cuda_pgs PRIVATE ${CUDA_cusparse_LIBRARY} )
target_link_libraries(cuda_pgs PRIVATE dlpack::dlpack)

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
