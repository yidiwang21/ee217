#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <stdint.h>
#include "support.cu"

static __global__ void GPUSpin(uint64_t duration, uint64_t *block_times, uint32_t *block_smids);

static __global__ void SharedMem_GPUSpin1024(uint64_t duration, uint64_t *block_times, uint32_t *block_smids);

static __device__ uint32_t UseSharedMemory1024(void);

#endif