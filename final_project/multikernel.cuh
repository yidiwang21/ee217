#ifndef _MULTIKERNEL_H_
#define _MULTIKERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include "third_party/cJSON.h"
#include <algorithm>    // std::swap
#include "kernel.cuh"

// currently support one GPU
#define SM_NUM  15  // 1070
#define MIN(a,b) ((a) < (b) ? (a) : (b))

// binary tree:
// left subtree: spaces in the left of below part
// right subtree: spaces in the right of above part
typedef struct {
    int x;
    int y;
}Point;

typedef struct Node{
    bool used;
    int height;
    int width;
    int kernel_id;  // 0 for empty space
    struct Node *left;
    struct Node *right;
    struct Node *parent;
    struct Node *grandparent;
    Point start_point;
    Point end_point;
}Node;

// keep the information for each kernel
typedef struct {
    char *kernel_name;
    int kernel_id;
    int block_size;     // # of threads per block
    int grid_size;      // # of blocks
    int shared_mem;     // = 0 for now
    int duration;
    int start_time;
    uint64_t *block_times, *block_times_d;
    uint32_t *block_smids, *block_smids_d;
    Node *fit;
}KernelInfo;

typedef struct {
    int kernel_id;
    int block_size; // height
    // int block_num_per_sm;    
    int duration;   // width
    int start_time;
}BlockInfo;

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
    void kernelLauncher();

    int kernel_num;
    int sched_policy;    
    cudaDeviceProp devProp;
private:
    KernelInfo *kernel_list;
    BlockInfo *block_list;
    ResourceInfo *sm_list;
    cJSON *parser;
    int count;

    void kernelInfoInit();
    void blockInfoInit();
    void scheduler();
    void GPUResourceInit();
    void cleanUp();

    void sortDurationDecending();
    void sortStartTimeAscending();
    Node* newNode();
    Node* findBestFit(Node *root, int w, int h);
    Node* splitNode(Node **node, int w, int h, int kid);
};



#endif