/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.cu"
#include "support.cu"

#define STREAM_NUM  3
#define BLOCK_SIZE  512

int main (int argc, char *argv[])
{
    srand(217);

    Timer timer;

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned VecSize;
    if (argc == 1) VecSize = 1000000;
    else if (argc == 2) VecSize = atoi(argv[1]);   
    else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    int SegSize = VecSize / STREAM_NUM + 1;
    int offset = VecSize - STREAM_NUM * SegSize; // for the last partition
    const unsigned int GRID_SIZE = (int)ceil((float(SegSize + offset)) / BLOCK_SIZE);

    float *A, *B, *C;
    cudaHostAlloc((void **) &A, VecSize * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **) &B, VecSize * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **) &C, VecSize * sizeof(float), cudaHostAllocDefault);
    for (int i = 0; i < VecSize; i++) A[i] = (rand()%100)/100.00;
    for (int i = 0; i < VecSize; i++) B[i] = (rand()%100)/100.00;
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    size Of vector: %u\n  ", VecSize);
    
    // partition vectors
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);
    float *d_A0, *d_B0, *d_C0;
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    
    cudaMalloc((void**) &d_A0, SegSize * sizeof(float));
    cudaMalloc((void**) &d_B0, SegSize * sizeof(float));
    cudaMalloc((void**) &d_C0, SegSize * sizeof(float));
    
    cudaMalloc((void**) &d_A1, SegSize * sizeof(float));
    cudaMalloc((void**) &d_B1, SegSize * sizeof(float));
    cudaMalloc((void**) &d_C1, SegSize * sizeof(float));

    cudaMalloc((void**) &d_A2, (SegSize+offset) * sizeof(float));
    cudaMalloc((void**) &d_B2, (SegSize+offset) * sizeof(float));
    cudaMalloc((void**) &d_C2, (SegSize+offset) * sizeof(float));

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Creating streams...\n");
    startTime(&timer);
    cudaStream_t streams[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    for (int i = 0; i < VecSize; i += STREAM_NUM * SegSize) {
        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);
        cudaMemcpyAsync(d_A0, A+i,              SegSize * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        cudaMemcpyAsync(d_B0, B+i,              SegSize * sizeof(float), cudaMemcpyHostToDevice, streams[0]);
        
        cudaMemcpyAsync(d_A1, A+i+SegSize,      SegSize * sizeof(float), cudaMemcpyHostToDevice, streams[1]);
        cudaMemcpyAsync(d_B1, B+i+SegSize,      SegSize * sizeof(float), cudaMemcpyHostToDevice, streams[1]);

        cudaMemcpyAsync(d_A2, A+i+2*SegSize, (offset + SegSize) * sizeof(float), cudaMemcpyHostToDevice, streams[2]);
        cudaMemcpyAsync(d_B2, B+i+2*SegSize, (offset + SegSize) * sizeof(float), cudaMemcpyHostToDevice, streams[2]);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        
        printf("Launching kernel..."); fflush(stdout);
        startTime(&timer);
        VecAdd <<<GRID_SIZE, BLOCK_SIZE, 0, streams[0]>>> (SegSize ,d_A0, d_B0, d_C0);
        VecAdd <<<GRID_SIZE, BLOCK_SIZE, 0, streams[1]>>> (SegSize ,d_A1, d_B1, d_C1);
        VecAdd <<<GRID_SIZE, BLOCK_SIZE, 0, streams[2]>>> (SegSize+offset ,d_A2, d_B2, d_C2);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
        
        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);
        cudaMemcpyAsync(C+i,            d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, streams[0]);
        cudaMemcpyAsync(C+i+SegSize,    d_C1, SegSize * sizeof(float), cudaMemcpyDeviceToHost, streams[1]);
        cudaMemcpyAsync(C+i+2*SegSize,  d_C2, (SegSize+offset) * sizeof(float), cudaMemcpyDeviceToHost, streams[2]);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    }

    cudaDeviceSynchronize();

    printf("Verifying results..."); fflush(stdout);
    verify(A, B, C, VecSize);

    cudaFreeHost(A); 
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFree(d_A0);
    cudaFree(d_B0);
    cudaFree(d_C0);
    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFree(d_A2);
    cudaFree(d_B2);
    cudaFree(d_C2);

    return 0;
}