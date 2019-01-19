/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__global__ void reduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    extern __shared__ float partialSum[2 * BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    partialSum[tid] = in[start + tid];
    partialSum[blockDim.x + tid] = in[start + blockDim.x + tid];

    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        if(tid % stride == 0)
            partialSum[2 * tid] += partialSum[2 * tid + stride];
    }

    if(tid == 0)
        out[blockIdx.x] = partialSum[0];
}
