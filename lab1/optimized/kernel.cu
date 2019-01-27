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
    __shared__ float partialSum[2*BLOCK_SIZE];

    unsigned int full_block_num = sizeof(in) / (2 * blockDim.x);
    unsigned int lazy_thread_num = blockDim.x - (sizeof(in) % (2 * blockDim.x)) / 2;

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    partialSum[t] = in[start + t];
    partialSum[blockDim.x + t] = in[start + blockDim.x + t];

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            if (blockIdx.x <= full_block_num - 1 || t < blockDim.x - lazy_thread_num)
                partialSum[t] += partialSum[t + stride];
        }
    }

    if (t == 0)
        out[blockIdx.x] = partialSum[0];
}
