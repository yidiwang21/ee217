#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <stdint.h>

static __global__ void GPUSpin(uint64_t duration, uint64_t *block_times, uint32_t *block_smids);

static __global__ void SharedMem_GPUSpin1024(uint64_t duration, uint64_t *block_times, uint32_t *block_smids);

static __device__ uint32_t UseSharedMemory1024(void);

#endif