#include <stdio.h>

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);

        cudaEvent_t start, stop, midpoint;
        cudaEventCreate(&start);
        cudaEventCreate(&midpoint);
        cudaEventCreate(&stop);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        cudaEventRecord(start);
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time1;
        cudaEventElapsedTime(&time1, start, stop);
        printf ("Time for the kernel: %f ms\n", time1);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}