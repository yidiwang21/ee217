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
    int SHARED_SIZE = 2 * BLOCK_SIZE;
    __shared__ float partialSum[];

    if (SHARED_SIZE != blockDim.x)
        for (int i = threadIdx.x; i < SHARED_SIZE; i += blockDim.x) 
            partialSum[i] = 0;
    __syncthreads();

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    partialSum[t] = in[start + t];
    partialSum[blockDim.x + t] = in[start + blockDim.x + t];

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }

    if (t == 0)
        out[blockIdx.x] = partialSum[0];
}
