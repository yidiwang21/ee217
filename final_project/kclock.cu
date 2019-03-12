#include <stdio.h>
#include <stdlib.h>
#include "support.cu"
#include "kernel.cu"

#define STREAM_NUM  3

#define MAX(a,b) (((a)>(b))?(a):(b))

// #define GRID_SIZE   15

#define DELAY_BETWEEN_LAUNCH    0 //0.005
#define SPIN_TIME   20000000         // 20ms

// typedef unsigned long long  uint64_t; 
// typedef unsigned long   uint32_t;

typedef struct {
    int gird_size;
    int block_size;
}Kernel_Info;

int main() {
    Kernel_Info *kernel_list = (Kernel_Info *)malloc(sizeof(Kernel_Info) * STREAM_NUM);
    kernel_list[0].gird_size = 15 * 3;
    kernel_list[0].block_size = 512;
    kernel_list[1].gird_size = 15 * 5;
    kernel_list[1].block_size = 256;
    kernel_list[2].gird_size = 15;
    kernel_list[2].block_size = 1024;

    int GRID_SIZE = MAX(MAX(kernel_list[0].gird_size, kernel_list[1].gird_size), kernel_list[2].gird_size);

    uint64_t **block_times, *block_times_d;
    uint32_t **block_smids, *block_smids_d;
    block_times = (uint64_t **)malloc(sizeof(uint64_t*) * STREAM_NUM);
    block_smids = (uint32_t **)malloc(sizeof(uint32_t*) * STREAM_NUM);

    cudaError_t cuda_ret;
    Timer timer;
    
    for (int k = 0; k < STREAM_NUM; k++) {
        block_times[k] = (uint64_t *)malloc(sizeof(uint64_t) * GRID_SIZE * 2);
        block_smids[k] = (uint32_t *)malloc(sizeof(uint32_t) * GRID_SIZE);
    }
    
    for (int i = 0; i < STREAM_NUM; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            block_times[i][2*j] = 0;
            block_times[i][2*j+1] = 0;
            block_smids[i][j] = 0;
        }
    }

    cudaMalloc((void**) &block_times_d, sizeof(uint64_t) * GRID_SIZE * 2 * STREAM_NUM);
    cudaMalloc((void**) &block_smids_d, sizeof(uint32_t) * GRID_SIZE * STREAM_NUM);
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaMemcpy(&block_times_d[i], block_times[i * GRID_SIZE * 2], sizeof(uint64_t) * 2 * GRID_SIZE, cudaMemcpyHostToDevice);
        cudaMemcpy(&block_smids_d[i], block_smids[i * GRID_SIZE], sizeof(uint32_t) * GRID_SIZE, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();

    printf("Launching kernel...\n"); fflush(stdout);
    printf("Kernel number: %d\n", STREAM_NUM);
    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    cudaDeviceSynchronize();

    // for (int i = 0; i < STREAM_NUM; i++) {
    //     GPUSpin <<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(SPIN_TIME, &block_times_d[i * GRID_SIZE * 2], &block_smids_d[i * GRID_SIZE]);
    //     // startTime(&timer);
    //     // do {
    //     //     stopTime(&timer);
    //     // }while(elapsedTime(timer) < DELAY_BETWEEN_LAUNCH);
    // }

    for (int i = 0; i < STREAM_NUM; i++) {
        GPUSpin <<<kernel_list[i].gird_size, kernel_list[i].block_size, 0, streams[i]>>>(SPIN_TIME, &block_times_d[i * GRID_SIZE * 2], &block_smids_d[i * GRID_SIZE]);
    }
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "Unable to launch kernel!\n"); exit(-1); }

    for (int i = 0; i < STREAM_NUM; i++) {
        cudaMemcpy(block_times[i], &block_times_d[i * GRID_SIZE * 2], sizeof(uint64_t) * 2 * GRID_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_smids[i], &block_smids_d[i * GRID_SIZE], sizeof(uint32_t) * GRID_SIZE, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamDestroy(streams[i]);

    float ms = 0;

    for (int j = 0; j < STREAM_NUM; j++) {
        printf("=========================================\n");
        // printf("kernel id: %d\n", j);
        // cudaEventElapsedTime(&ms, events[j * 2], events[j*2+1]);
        // cudaEventElapsedTime(&ms, start, stop);
        // printf("Duration: %f\n", ms * 1000);
        for (int i = 0; i < GRID_SIZE; i++) {   // print each block
            block_times[j][i*2] = (block_times[j][i*2] / 1000 / 1000) % 10000;
            block_times[j][i*2+1] = (block_times[j][i*2+1] / 1000 / 1000) % 10000;
            if (block_times[j][i*2] != 0 && block_times[j][i*2+1] != 0) {
                printf("Block index: %d\n", i);
                printf("kernel id: %d\n", j);
                printf("SM id: %d\n", block_smids[j][i]);
                printf("start time: %d\n", block_times[j][i*2]);
                printf("stop time: %d\n", block_times[j][i*2+1]);
                printf("elapsed time: %d\n\n", block_times[j][2*i+1] - block_times[j][2*i]);
            }
            // printf("start time: %d\n", block_times[i]);
            // printf("stop time: %d\n", block_times[i+1]);
            // printf("elapsed time: %d\n\n", (block_times[i+1] - block_times[i]));
        }
    }
}

