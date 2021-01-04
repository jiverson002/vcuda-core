// SPDX-License-Identifier: MIT
#ifndef VCUDA_H
#define VCUDA_H 1

#include "vcuda/auto_cast.h"

/*----------------------------------------------------------------------------*/
/*! Black magic. */
/*----------------------------------------------------------------------------*/
#define _SELECT(_1, _2, _3, _4, NAME, ...) NAME
#define SELECT(f,...) _SELECT( __VA_ARGS__\
                             , f ## 4\
                             , f ## 3\
                             , f ## 2\
                             , f ## 1\
                             , XXX\
                             )(__VA_ARGS__)

#define VCUDA_GET_ARG4(arg1, ...)\
  VCUDA_GET_ARG1(arg1) VCUDA_GET_ARG3(__VA_ARGS__)
#define VCUDA_GET_ARG3(arg1, ...)\
  VCUDA_GET_ARG1(arg1) VCUDA_GET_ARG2(__VA_ARGS__)
#define VCUDA_GET_ARG2(arg1, ...)\
  VCUDA_GET_ARG1(arg1) VCUDA_GET_ARG1(__VA_ARGS__)
#define VCUDA_GET_ARG1(arg1) arg1 = auto_cast(kernelParams[VCUDA_kp_ctr++]);

#define VCUDA_get_args(...) SELECT(VCUDA_GET_ARG, __VA_ARGS__)

#define VCUDA_defn_make(scope, name) \
  static void\
  VCUDA_kernel_ ## name(\
    size_t VCUDA_kp_ctr,\
    const dim3 gridDim,\
    const dim3 blockDim,\
    const dim3 blockIdx,\
    const dim3 threadIdx,\
    void **    kernelParams,\
    void **    extra\
  )\
  {\
    VCUDA_get_args

#define VCUDA_defn_args(scope, name) VCUDA_defn_make(scope, name)
#define VCUDA_defn_name(rest)        rest
#define VCUDA_defn_void              VCUDA_defn_name(
#define VCUDA_defn_type(rest)        VCUDA_defn_ ## rest)
#define VCUDA_defn_extern            VCUDA_defn_args(extern, VCUDA_defn_type(
#define VCUDA_defn_static            VCUDA_defn_args(static, VCUDA_defn_type(
#define VCUDA_defn_scope(rest)       VCUDA_defn_ ## rest))
#define __global__                   VCUDA_defn_scope
#define __global_init__              (void)gridDim; (void)blockDim;\
                                     (void)blockIdx; (void)threadIdx;\
                                     (void)kernelParams; (void)extra;}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
#define __syncthreads() _Pragma("omp barrier")

/*----------------------------------------------------------------------------*/
/*! Driver API. */
/*----------------------------------------------------------------------------*/
#ifndef VCUDA_NO_DRIVER
# include "vcuda/driver.h"
#endif

/*----------------------------------------------------------------------------*/
/*! Runtime API. */
/*----------------------------------------------------------------------------*/
#ifndef VCUDA_NO_RUNTIME
# include "vcuda/runtime.h"
#endif

#endif // VCUDA_H
