#ifndef __LAUNCHER_H__
#define __LAUNCHER_H__

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
    uint64_t *block_times;
    uint32_t *block_smids;
}KernelInfo;

// available resource on each SM
typedef struct {
    int available_thread_num;       
    int available_shared_mem;       // byte, not needed for now
    int available_register_num;     // not needed for now
}ResourceInfo;

class Launcher
{   
public:
    Launcher(char *config_file);
    ~Launcher();

    int kernel_num;
    int sched_policy;
    cudaDeviceProp devProp;
private:
    KernelInfo *kernel_list;
    ResourceInfo *sm_list;
    cJSON *parser;

    void kernelInfoInit();
    void cleanUp();

};

#endif