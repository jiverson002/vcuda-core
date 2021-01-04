// SPDX-License-Identifier: MIT
#ifndef VCUDA_CORE_H
#define VCUDA_CORE_H 1

#include <cstddef>
#include <vector>

#include "vcuda/core/export.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
typedef struct dim3 {
  unsigned x;
  unsigned y;
  unsigned z;

#ifdef __cplusplus
  dim3(void) : dim3(1) { }
  dim3(unsigned x) : dim3(x, 1) { }
  dim3(unsigned x, unsigned y) : dim3(x, y, 1) { }
  dim3(unsigned x, unsigned y, unsigned z) : x(x), y(y), z(z) { }
#endif
} dim3;

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
typedef enum cudaError {
  CUDA_SUCCESS = 0,
  CUDA_ERROR,
  CUDA_ERROR_NOT_INITIALIZED,
  CUDA_ERROR_INVALID_VALUE,
  CUDA_ERROR_LAUNCH_FAILED,
  CUDA_ERROR_INVALID_MEMCPY_DIRECTION,
  CUDA_ERROR_OUT_OF_MEMORY
} CUresult;

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
enum cudaMemcpyKind {
  cudaMemcpyDeviceToHost,
  cudaMemcpyHostToDevice
};

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
typedef std::size_t CUstream;

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
typedef void *CUdeviceptr;

/*----------------------------------------------------------------------------*/
/*! Device function abstraction. */
/*----------------------------------------------------------------------------*/
class CUfunction {
  private:
    using device_fn = void (*)(size_t, const dim3, const dim3, const dim3,
                               const dim3, void **, void **);

  public:
    CUfunction() : CUfunction(NULL) { }
    CUfunction(device_fn fn) : fn(fn), argc(0), argSize() { }
    CUfunction(device_fn fn, const std::vector<size_t>& argSize)
      : fn(fn), argc(argSize.size()), argSize(argSize) { }

    device_fn fn;
    const int argc;
    const std::vector<size_t> argSize;
};

#endif // VCUDA_CORE_H
