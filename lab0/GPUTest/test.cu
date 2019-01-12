#include <cstdio> 
#include <cstdlib> 
#define UPPER_LIMIT 100 
#define OFFSET 3 
#define LENGTH 32
// GPU kernel: Add offset
__global__ void addOffset(int* dev_array, int length)
{
    int tid=threadIdx.x + blockIdx.x*blockDim.x; 
    if(tid < length)
    {
        dev_array[tid] += OFFSET;
    }
}

// Main host code
int main()
{
    // Host and Device Arrays
    int arr_h[LENGTH], res_h[LENGTH]; 
    int* arr_d;
    cudaMalloc((void**) &arr_d, sizeof(int) * LENGTH);
    // Assigning random values to the host array
    for(int i=0; i<LENGTH; i++)
    {
        arr_h[i] = rand() % UPPER_LIMIT;
    }
    // Copying the host array to the GPU memory
    cudaMemcpy(arr_d, arr_h, sizeof(int) * LENGTH, cudaMemcpyHostToDevice);
    // Launching GPU Kernel
    addOffset <<<1, 1024>>> (arr_d, LENGTH);
    // Copying the GPU array back to the result array on the host
    cudaMemcpy(res_h, arr_d, sizeof(int) * LENGTH, cudaMemcpyDeviceToHost);
    // Verify the results using CPU
    for(int i=0; i<LENGTH; i++)
    {
        register int expected = arr_h[i] + OFFSET; 
        printf("%d / %d\n", res_h[i], expected); 
        if(res_h[i] != expected)
        {
            printf("FAILURE at i=%d", i);
            break; 
        }
    }
    // Freeing up the GPU memory
    cudaFree(arr_d);
    return 0; 
}