// GPU Compatibility Header
// Provides unified interface for CUDA and ROCm/HIP
#pragma once

#ifdef PGS_USE_ROCM
// ROCm/HIP backend
#include <hip/hip_runtime.h>
#include <rocsparse/rocsparse.h>

// Map CUDA types and functions to HIP equivalents
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemset hipMemset
#define cudaFree hipFree
#define cudaMalloc hipMalloc
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp hipDeviceProp_t

// cuSPARSE to rocSPARSE
#define cusparseHandle_t rocsparse_handle
#define cusparseStatus_t rocsparse_status
#define cusparseCreate rocsparse_create_handle
#define cusparseDestroy rocsparse_destroy_handle
#define CUSPARSE_STATUS_SUCCESS rocsparse_status_success

// Kernel launch syntax is the same in HIP
#define __CUDA_ARCH__ __HIP_DEVICE_COMPILE__

#define GPU_BACKEND "ROCm/HIP"
#define GPU_SPARSE_LIBRARY "rocSPARSE"

#else
// CUDA backend (default)
#include <cuda_runtime.h>
#include <cusparse.h>

#define GPU_BACKEND "CUDA"
#define GPU_SPARSE_LIBRARY "cuSPARSE"

#endif

// Common GPU utility macros
#define GPU_CHECK(call)                                                        \
  do {                                                                         \
    auto err = call;                                                           \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "GPU error at %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Device function qualifiers (same for both)
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif

// Thread indexing helpers (work for both CUDA and HIP)
#define GPU_THREAD_IDX_X (blockIdx.x * blockDim.x + threadIdx.x)
#define GPU_THREAD_IDX_Y (blockIdx.y * blockDim.y + threadIdx.y)
#define GPU_THREAD_IDX_Z (blockIdx.z * blockDim.z + threadIdx.z)

#define GPU_BLOCK_IDX_X blockIdx.x
#define GPU_BLOCK_IDX_Y blockIdx.y
#define GPU_BLOCK_IDX_Z blockIdx.z

#define GPU_BLOCK_DIM_X blockDim.x
#define GPU_BLOCK_DIM_Y blockDim.y
#define GPU_BLOCK_DIM_Z blockDim.z

#define GPU_GRID_DIM_X gridDim.x
#define GPU_GRID_DIM_Y gridDim.y
#define GPU_GRID_DIM_Z gridDim.z

#define GPU_SYNC_THREADS __syncthreads()

// Atomic operations (same API for both)
#define GPU_ATOMIC_ADD atomicAdd
#define GPU_ATOMIC_MIN atomicMin
#define GPU_ATOMIC_MAX atomicMax

// Math functions (same for both)
#define GPU_SQRT sqrtf
#define GPU_ABS fabsf
#define GPU_MIN fminf
#define GPU_MAX fmaxf

// Print backend info
inline void print_gpu_backend_info() {
  printf("PGS Solver compiled with %s backend\n", GPU_BACKEND);
  printf("Using %s for sparse operations\n", GPU_SPARSE_LIBRARY);
}
