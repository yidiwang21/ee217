#include <stdio.h>
#include <stdlib.h>

#define ITER_NUM    1024 * 8
#define STREAM_NUM  33
#define BLOCK_SIZE  32
#define GRID_SIZE   15
#define VEC_SIZE    1000


__global__ void dummyKernel(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < ITER_NUM; j++) {
        int tmp = j % 64;
        for (int i = threadIdx.x; i < n; i += BLOCK_SIZE) {
            if (i + tmp >= n) tmp = 0;
            c[i] = a[i] + b[i];
        }
    }
}

int main() {
    printf("\nSetting up the problem..."); fflush(stdout);
    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    unsigned int size = VEC_SIZE;

    A_h = (float*) malloc( sizeof(float)*size);
    for (unsigned int i=0; i < size; i++) 
        A_h[i] = (rand()%100)/100.00; 
    
        B_h = (float*) malloc( sizeof(float)*size);
    for (unsigned int i=0; i < size; i++) 
        B_h[i] = (rand()%100)/100.00;

    C_h = (float*) malloc( sizeof(float)*size);

    printf("Allocating device variables..."); fflush(stdout);
    size_t bytes = sizeof(float) * VEC_SIZE;
    cudaMalloc((void**) &A_d, bytes);
    cudaMalloc((void**) &B_d, bytes);
    cudaMalloc((void**) &C_d, bytes);
    cudaDeviceSynchronize();

    printf("Copying data from host to device..."); fflush(stdout);
    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    printf("Launching kernel..."); fflush(stdout);
    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        printf("streams[%d] = %d", i, streams[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++) {
        dummyKernel <<<GRID_SIZE, BLOCK_SIZE, 0, streams[i]>>>(A_d, B_d, C_d, VEC_SIZE);
    }
    // dummyKernel <<<GRID_SIZE, BLOCK_SIZE, 0, streams[0]>>>(A_d, B_d, C_d, VEC_SIZE);
    // dummyKernel <<<GRID_SIZE, BLOCK_SIZE, 0, streams[1]>>>(A_d, B_d, C_d, VEC_SIZE);
    // dummyKernel <<<GRID_SIZE, BLOCK_SIZE, 0, streams[2]>>>(A_d, B_d, C_d, VEC_SIZE);
    // dummyKernel <<<GRID_SIZE, BLOCK_SIZE, 0, streams[3]>>>(A_d, B_d, C_d, VEC_SIZE);
    
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamDestroy(streams[i]);
    free(A_h);
    free(B_h);
    free(C_h);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}