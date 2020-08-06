/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to

#define BLOCK_SIZE 512
// INSERT KERNEL(S) HERE

__global__ void histogram_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {
    
    extern __shared__ unsigned int histo_private[];    // size = num_bins
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // threads cooperatively initialize the private histogram to 0
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        histo_private[i] = 0;
    }

    __syncthreads();
    // build private histogram
    int stride = blockDim.x * gridDim.x;
    for (unsigned int j = tid; j < num_elements; j += stride) {
        // by default, the randomly generated value should be in range (0, 4095)
        int position = input[j];
        if (position >= 0 && position <= num_bins - 1) {
            atomicAdd(&histo_private[position], 1);
        }
    }

    __syncthreads();
    // build final histogram
    for (unsigned int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], histo_private[i]);
    }
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // INSERT CODE HERE
    const unsigned int GRID_SIZE = (int)ceil((float(num_elements)) / BLOCK_SIZE);
    histogram_kernel <<<GRID_SIZE, BLOCK_SIZE, num_bins*sizeof(unsigned int)>>> (input, bins, num_elements, num_bins);
}


