#ifndef __KERNEL_CU__
#define __KERNEL_CU__

#include <stdint.h>
#include "support.cu"

#define ITER_NUM    10000   // just an amplifier
#define ITER_NUM_LARGE  1000000

// My kernels do something meaningless like reversing content in shared memory
// just spin for a while and allocate some shared memory

typedef struct {
    int *input;
    int input_num;
}InputStruct;

__global__ void kernel_basic_reverse(int *d, int n, int delay) {
    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();
}

__global__ void kernel_shared_mem_1024(int *d, int n, int delay) {
    __shared__ uint8_t SharedMemArr[1024];
    int t = threadIdx.x;
    int tr = n - t - 1;
    SharedMemArr[t] = d[t];
    __syncthreads();
    
    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();

    d[t] = SharedMemArr[tr];
}

__global__ void kernel_shared_mem_4096(int *d, int n, int delay) {
    __shared__ uint8_t SharedMemArr[4096];
    int t = threadIdx.x;
    int tr = n - t - 1;
    SharedMemArr[t] = d[t];
    __syncthreads();
    
    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();

    d[t] = SharedMemArr[tr];
}

__global__ void kernel_shared_mem_16384(int *d, int n, int delay) {
    __shared__ uint8_t SharedMemArr[16384];
    int t = threadIdx.x;
    int tr = n - t - 1;
    SharedMemArr[t] = d[t];
    __syncthreads();
    
    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();

    d[t] = SharedMemArr[tr];
}

// TODO: compute exe time for each block

__global__ void lazyKernel_0(int delay) {
    int bytes_per_thread = 0;
    // for (int j = 0; j < ITER_NUM; j++) {
    //     bytes_per_thread = threadIdx.x;
    // }

    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay){
        for (int j = 0; j < ITER_NUM; j++) {
            bytes_per_thread = threadIdx.x + blockDim.x * gridDim.x;
        }
    }
    unsigned int stop = kernelTimer();
}

__global__ void lazyKernel_1024(int delay) {
    __shared__ uint8_t SharedMemArr[1024];
 
    int bytes_per_thread = 1024 / blockDim.x;
    for (int j = 0; j < ITER_NUM; j++) {
        for (unsigned int i = 0; i < bytes_per_thread; i++) 
            SharedMemArr[bytes_per_thread * threadIdx.x + i] = threadIdx.x;
    }

    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();
}

__global__ void lazyKernel_4096(int delay) {
    __shared__ uint8_t SharedMemArr[4096];
 
    int bytes_per_thread = 4096 / blockDim.x;
    for (int j = 0; j < ITER_NUM; j++) {
        for (unsigned int i = 0; i < bytes_per_thread; i++) 
            SharedMemArr[bytes_per_thread * threadIdx.x + i] = threadIdx.x;
    }

    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();
}

__global__ void lazyKernel_8192(int delay) {
    __shared__ uint8_t SharedMemArr[8192];
 
    int bytes_per_thread = 8192 / blockDim.x;
    for (int j = 0; j < ITER_NUM; j++) {
        for (unsigned int i = 0; i < bytes_per_thread; i++) 
            SharedMemArr[bytes_per_thread * threadIdx.x + i] = threadIdx.x;
    }

    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();
}

__global__ void lazyKernel_16384(int delay) {
    __shared__ uint8_t SharedMemArr[16384];
 
    // do something boring!
    int bytes_per_thread = 16384 / blockDim.x;
    for (int j = 0; j < ITER_NUM; j++) {
        for (unsigned int i = 0; i < bytes_per_thread; i++) 
            SharedMemArr[bytes_per_thread * threadIdx.x + i] = threadIdx.x;
    }

    unsigned int start = kernelTimer();
    while (kernelTimer() - start < delay);
    unsigned int stop = kernelTimer();
}

__global__ void kernel_no_mem(int delay) {
    int tid = threadIdx.x + blockDim.x * gridDim.x;
    int x;
    int n = 1000;
    for (int j = 0; j < ITER_NUM; j++) {
        int tmp = j % 64;
        for (int i = threadIdx.x; i < n; i += 32) {
            if (i + tmp >= n) tmp = 0;
            x = 3 * x;
        }
    }
}
#endif