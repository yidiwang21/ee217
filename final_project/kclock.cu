#include <stdio.h>
#include <stdlib.h>

#define STREAM_NUM  33
#define BLOCK_SIZE  32
#define GRID_SIZE   19

#define ITER_NUM    8 * 1024

typedef unsigned long long  uint64_t; 
typedef unsigned long   uint32_t;

__device__ inline uint64_t GlobalTimer64(void) {
    /* volatile uint64_t first_reading;
    // volatile uint32_t second_reading;
    volatile unsigned int second_reading;
    uint32_t high_bits_first;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
    high_bits_first = first_reading >> 32;
    asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
    if (high_bits_first == second_reading) {
      return first_reading;
    }
    // Return the value with the updated high bits, but the low bits set to 0.
    return ((uint64_t) second_reading) << 32;
     */
    //  volatile uint64_t ret;
    //  asm volatile("mov.u64  %0, %clock64;": "=l"(ret));
    //  return ret;
     volatile uint64_t reading;
     asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(reading));
     return reading;
}

static __device__ __inline__ unsigned int GetSMID(void) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

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
    // while ((GlobalTimer64() - start_time) < spin_duration) {
    //     continue;
    // }
    // Record the kernel and block end times.
    if (shared_mem_res == 0) {
        block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
    }
}



int main() {
    uint64_t **block_times, *block_times_d;
    uint32_t **block_smids, *block_smids_d;
    block_times = (uint64_t **)malloc(sizeof(uint64_t*) * STREAM_NUM);
    block_smids = (uint32_t **)malloc(sizeof(uint32_t*) * STREAM_NUM);

    cudaError_t cuda_ret;
    
    for (int k = 0; k < STREAM_NUM; k++) {
        block_times[k] = (uint64_t *)malloc(sizeof(uint64_t) * GRID_SIZE * 2);
        block_smids[k] = (uint32_t *)malloc(sizeof(uint32_t) * GRID_SIZE);
    }

    // FIXME: cudamalloc these in vector form
    cudaMalloc((void**) &block_times_d, sizeof(uint64_t) * GRID_SIZE * 2 * STREAM_NUM);
    cudaMalloc((void**) &block_smids_d, sizeof(uint32_t) * GRID_SIZE * STREAM_NUM);

    // for (int k = 0; k < STREAM_NUM; k++) {
    //     cudaMalloc((void**) &block_times_d[k], sizeof(uint64_t) * GRID_SIZE * 2);
    //     cudaMalloc((void**) &block_smids_d[k], sizeof(uint32_t) * GRID_SIZE);
    // }
    
    cudaDeviceSynchronize();

    printf("Launching kernel...\n"); fflush(stdout);
    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    cudaDeviceSynchronize();

    cudaEvent_t *events;
    events = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * STREAM_NUM * 2);
    for (int i = 0; i < STREAM_NUM * 2; i++) {
        cudaEventCreate(&events[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int j = 0;
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaEventRecord(events[j]);
        // cudaEventRecord(start);
        SharedMem_GPUSpin1024 <<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(0, &block_times_d[i * GRID_SIZE * 2], &block_smids_d[i * GRID_SIZE]);
        cudaEventRecord(events[j+1]);
        cudaEventSynchronize(events[j+1]);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        j += 2;
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaMemcpy(block_times[i], &block_times_d[i * GRID_SIZE * 2], sizeof(uint64_t) * 2 * GRID_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_smids[i], &block_smids_d[i * GRID_SIZE], sizeof(uint32_t) * GRID_SIZE, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    float ms = 0;

    for (int j = 0; j < STREAM_NUM; j++) {
        printf("=========================================\n");
        printf("kernel id: %d\n", j);
        cudaEventElapsedTime(&ms, events[j * 2], events[j*2+1]);
        // cudaEventElapsedTime(&ms, start, stop);
        // printf("Duration: %f\n", ms * 1000);
        for (int i = 0; i < GRID_SIZE; i++) {
            printf("Block index: %d\n", i);
            printf("SM id: %d\n", block_smids[j][i]);
            printf("start time: %d\n", block_times[j][i*2] / 1000 / 1000);
            printf("stop time: %d\n", block_times[j][i*2+1] / 1000 / 1000);
            printf("elapsed time: %d\n\n", (block_times[j][2*i+1] - block_times[j][2*i]) / 1000 / 1000);
            // printf("start time: %d\n", block_times[i]);
            // printf("stop time: %d\n", block_times[i+1]);
            // printf("elapsed time: %d\n\n", (block_times[i+1] - block_times[i]));
        }
    }
}

