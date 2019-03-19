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
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// #define DRAW_TIMELINE

// binary tree:
// left subtree: spaces in the left of below part
// right subtree: spaces in the right of above part
typedef struct {
    int x;
    int y;
}Point;

typedef struct Node{
    bool used;
    bool closed;
    int height;
    uint64_t width;
    int kernel_id;  // 0 for empty space
    bool growable;  // left always cannot grow
    struct Node *left;
    struct Node *right;
    struct Node *parent;
    struct Node *grandparent;
    Point start_point;
    Point end_point;
}Node;

// keep the information for each kernel
typedef struct {
    // char *kernel_name;
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
    MultiKernel(char *fn, int sp);
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

    void sortDurationAscending();   // for policy 1
    void printBlocks();

    Node* root = this->newNode();

    void sortDurationDecending();
    void sortStartTimeAscending();
    Node* newNode();
    Node* findBestFit(Node *node, int w, int h);
    // Node* splitNode(Node **node, int w, int h, int kid);
    // FIXME:
    Node* splitNode(Node *node, int w, int h, int kid);
    Node* growNode(Node *node, int w, int h, int kid);
    Node* searchNode(Node* node, int key, Node *node_const);
    int findMinUnusedToGrow(Node *node, int h);
    void updateParentsRight(Node *node, int w, int h, int stp);
};



#endif