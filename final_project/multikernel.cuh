#ifndef _MULTIKERNEL_H_
#define _MULTIKERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "third_party/cJSON.h"
#include <algorithm>    // std::swap
#include "kernel.cuh"

// currently support one GPU
#define SM_NUM  15  // 1070

// keep the information for each kernel
typedef struct {
    char *kernel_name;
    int kernel_id;
    int block_size;     // # of threads per block
    int grid_size;      // # of blocks
    int shared_mem;     // = 0 for now
    int duration;
    bool valid;         // for bin packing
    uint64_t *block_times, *block_times_d;
    uint32_t *block_smids, *block_smids_d;
}KernelInfo;

// available resource on each SM
typedef struct {
    int available_thread_num;       
    int available_shared_mem;       // byte, not needed for now
    int available_register_num;     // not needed for now
}ResourceInfo;

class MultiKernel
{   
public:
    MultiKernel(char *fn);
    // ~MultiKernel();

    int kernel_num;
    int sched_policy;
    // char *config_file;
    
    cudaDeviceProp devProp;

    void kernelLauncher();
private:
    KernelInfo *kernel_list;
    ResourceInfo *sm_list;
    cJSON *parser;

    void kernelInfoInit();
    void scheduler();
    void GPUResourceInit();
    void cleanUp();

    void sortDurationDecending();
};

#endif