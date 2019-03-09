#ifndef __KERNEL_CU__
#define __KERNEL_CU__

#include "kernel.cuh"

#define ITER_NUM    8*1024   // just an amplifier
#define ITER_NUM_LARGE  1000000

static __device__ uint32_t UseSharedMemory1024(void) {
    __shared__ uint32_t shared_mem_arr[1024];
    uint32_t num_threads, elts_per_thread, i;
    num_threads = blockDim.x;
    elts_per_thread = 1024 / num_threads;
    for (int j = 0; j < ITER_NUM; j++) {
        for (i = 0; i < elts_per_thread; i++) {
            shared_mem_arr[threadIdx.x * elts_per_thread + i] = threadIdx.x;
        }
    }
    return shared_mem_arr[threadIdx.x * elts_per_thread];
}

static __global__ void SharedMem_GPUSpin1024(uint64_t spin_duration, uint64_t *block_times, uint32_t *block_smids) {
    uint32_t shared_mem_res;
    uint64_t start_time = GlobalTimer64();
    if (threadIdx.x == 0) {
        block_times[blockIdx.x * 2] = start_time;
        block_smids[blockIdx.x] = GetSMID();
    }
    __syncthreads();
    shared_mem_res = UseSharedMemory1024();
    while ((GlobalTimer64() - start_time) < spin_duration) {
        continue;
    }
    if (shared_mem_res == 0) {
        block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
    }
}

static __global__ void GPUSpin(uint64_t spin_duration, uint64_t *block_times, uint32_t *block_smids) {
    uint64_t start_time = GlobalTimer64();
    if (threadIdx.x == 0) {
        block_times[blockIdx.x * 2] = start_time;
        block_smids[blockIdx.x] = GetSMID();
    }
    __syncthreads();
    while ((GlobalTimer64() - start_time) < spin_duration) {
        continue;
    }
    if (threadIdx.x == 0) {
        block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
    }
}



#endif