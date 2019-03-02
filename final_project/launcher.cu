#ifndef __LAUNCHER_CU__
#define __LAUNCHER_CU__

#include <stdio.h>
#include "third_party/cJSON.h"
#include <algorithm>    // std::swap
#include "kernel.cu"

// keep the information for each kernel
typedef struct {
    char *kernel_name;
    int kernel_id;
    int block_size;     // # of threads per block
    int grid_size;      // # of blcoks
    int shared_mem;
    int duration;
}KernelInfo;

// keep the information for the benchmark
typedef struct {
    int kernel_num;
    int iteration;
    int sched_policy;
    KernelInfo *kernel_config;
}BenchmarkInfo;

typedef struct {
    int available_thread_num;
    int available_shared_mem_size;  // byte
    int available_register_num;
}ResourceInfo;


BenchmarkInfo benchmark_config;


// input value: kernel_id
void release(int idx, void* stream_id, int *in, int n) {
    KernelInfo kernel_config_tmp;
    kernel_config_tmp = benchmark_config.kernel_config[idx];

    printf("# Launching kernel[%d]...\n", kernel_config_tmp.kernel_id);
    
    switch(kernel_config_tmp.shared_mem) {  // test for 1070
        case 0:
            printf("    Shared mem size = 0\n");
            // kernel_basic_reverse <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, streams[i]>>> (input_struct.input, input_struct.input_num, kernel_config_tmp.duration);
            lazyKernel_0 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, stream_id>>> (kernel_config_tmp.duration);
            break;
        case 1024:
            printf("    Shared mem size = 1024\n");
            // kernel_shared_mem_1024 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 1024, stream_id>>> (in, n, kernel_config_tmp.duration);
            lazyKernel_1024 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, stream_id>>> (kernel_config_tmp.duration);
            break;
        case 4096:
            printf("    Shared mem size = 4096\n");
            // kernel_shared_mem_4096 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 4096, stream_id>>> (in, n, kernel_config_tmp.duration);
            lazyKernel_4096 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, stream_id>>> (kernel_config_tmp.duration);
            break;
        case 8192:
            printf("    Shared mem size = 8192\n");
            // kernel_shared_mem_4096 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 4096, stream_id>>> (in, n, kernel_config_tmp.duration);
            lazyKernel_8192 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, stream_id>>> (kernel_config_tmp.duration);
            break;
        case 16384:
            printf("    Shared mem size = 16348\n");
            // kernel_shared_mem_16384 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 16384, stream_id>>> (in, n, kernel_config_tmp.duration);
            lazyKernel_16384 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, stream_id>>> (kernel_config_tmp.duration);
            break;
        default:
            fprintf(stderr, "# Invalid shared memory size!\n");
    }
}

void sort_with_used_threads(KernelInfo *kernel_config, int size) {
    printf("# Policy 1: Sorting based on available threads number...\n");
    bool swapped;

    do {
        swapped = false;
        for (int i = 0; i < size - 1; i++) {
            if (kernel_config[i].grid_size * kernel_config[i].block_size > kernel_config[i+1].grid_size * kernel_config[i+1].block_size) {
                std::swap(kernel_config[i], kernel_config[i+1]);
                swapped = true; 
            }
        }
    } while(swapped);   
    printf("# Sorting finished.\n");
}

// This seems to influence more on kernel execution time
// so, let big kernel go first
void sort_with_used_shared_mem(KernelInfo *kernel_config, int size) {
    printf("# Policy 2: Sorting based on available shared memory size...\n");
    bool swapped;

    do {
        swapped = false;
        for (int i = 0; i < size - 1; i++) {
            if (kernel_config[i].shared_mem < kernel_config[i+1].shared_mem) {
                std::swap(kernel_config[i], kernel_config[i+1]);
                swapped = true; 
            }
        }
    } while(swapped);   
    printf("# Sorting finished.\n");
}

void CleanUp(cJSON *obj, KernelInfo *kobj) {
    if (obj) cJSON_Delete(obj);
    if (kobj) free(kobj);
}

int scheduler(char *config_file, int *in_d, int n) {
    KernelInfo *kernel_config = NULL;
    cJSON *parser = cJSON_Parse(config_file);
    cJSON *entry = NULL;
    cJSON *kernel_entry = NULL;
    cJSON *iter = NULL;

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);

    printf("# Parsing kernel config file...\n");
    printf("[\n");
    // Step 1: get kernel information from config file
    if (parser == NULL) {
        fprintf(stderr, "# Invalid kernel config file!\n");
        CleanUp(parser, kernel_config);
        return 0;
    }

    entry = cJSON_GetObjectItem(parser, "kernel_number");
    if (!entry || entry->type != cJSON_Number) {
        fprintf(stderr, "# Invalid kernel number!\n");
        CleanUp(parser, kernel_config);
        return 0;
    }
    benchmark_config.kernel_num = entry->valueint;
    printf("    kernel_num = %d\n", benchmark_config.kernel_num);

    entry = cJSON_GetObjectItem(parser, "iterations");
    if (!entry || entry->type != cJSON_Number) {
        fprintf(stderr, "Invalid iteration!\n");
        CleanUp(parser, kernel_config);
        return 0;
    }
    benchmark_config.iteration = entry->valueint;
    printf("    iteration = %d\n", benchmark_config.iteration);

    // scheduling policy: 0 for naive; 1 for # of threads; 2 for shared memory size
    entry = cJSON_GetObjectItem(parser, "sched_policy");
    if (!entry || entry->type != cJSON_Number || entry->valueint > 2 || entry->valueint < 0) {
        fprintf(stderr, "Invalid iteration! (must be 1 ,2, 3)\n");
        CleanUp(parser, kernel_config);
        return 0;
    }
    benchmark_config.sched_policy = entry->valueint;
    printf("    sched_policy = %d\n", benchmark_config.sched_policy);
    
    kernel_config = (KernelInfo*)malloc(benchmark_config.kernel_num * sizeof(KernelInfo));
    
    kernel_entry = cJSON_GetObjectItem(parser, "kernel_entry");
    if (!kernel_entry) {
        fprintf(stderr, "Invalid kernel entry!\n");
        CleanUp(parser, kernel_config);
        return 0;
    }

    int count = 0;
    cJSON_ArrayForEach(iter, kernel_entry) {
        cJSON *kernel_name = cJSON_GetObjectItem(iter, "kernel_name");
        cJSON *kernel_id = cJSON_GetObjectItem(iter, "kernel_id");
        cJSON *grid_size = cJSON_GetObjectItem(iter, "grid_size");
        cJSON *block_size = cJSON_GetObjectItem(iter, "block_size");
        cJSON *shared_mem = cJSON_GetObjectItem(iter, "shared_mem");
        cJSON *duration = cJSON_GetObjectItem(iter, "duration");
        
        kernel_config[count].kernel_id = kernel_id->valueint;
        printf("    kernel[%d]: {\n", count+1);
        kernel_config[count].grid_size = grid_size->valueint;
        printf("        grid size = %d\n", kernel_config[count].grid_size);
        kernel_config[count].duration = duration->valueint;
        printf("        duration = %d\n", kernel_config[count].duration);
        if (block_size->valueint < 1 || block_size->valueint > dev_prop.maxThreadsPerBlock) {
            fprintf(stderr, "# Invalid block size at kernel[%d] (must be in range of 1 ~ %d)\n", count+1, dev_prop.maxThreadsPerBlock);
            CleanUp(parser, kernel_config);
            return 0;
        }
        kernel_config[count].block_size = block_size->valueint;
        printf("        block size = %d\n", kernel_config[count].block_size);
        
        if (shared_mem->valueint < 0 || shared_mem->valueint > dev_prop.sharedMemPerBlock) {
            fprintf(stderr, "# Invalid shared memory size at kernel[%d] (must be in range of 0 ~ %zu)\n", count+1, dev_prop.sharedMemPerBlock);
            CleanUp(parser, kernel_config);
            return 0;
        }
        kernel_config[count].shared_mem = shared_mem->valueint;
        printf("        shared mem size = %d\n", kernel_config[count].shared_mem);
        printf("    }\n");
        count++;
    }
    printf("]\n");
    
    // Step 2: scheduling algorithm
    // TODO:
    // How to know the number of registers a kernel is going to use?
    // How to sort the kernels based on their "size"? (three metrics)
    // Don't need to control release time, just need re-order


    int KERNEL_NUM = benchmark_config.kernel_num;
    benchmark_config.kernel_config = (KernelInfo*)malloc(KERNEL_NUM * sizeof(KernelInfo));
    
    cudaError_t cuda_ret;

    cudaStream_t *streams;
    streams = (cudaStream_t*)malloc(KERNEL_NUM * sizeof(cudaStream_t));
    for (int i = 0; i < KERNEL_NUM; i++) {
        // cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        cudaStreamCreate(&streams[i]);
    }
    cudaDeviceSynchronize();

    // Scheduling Begins Here
    // FIXME: suppose that all kernels arriving at the same time

    switch(benchmark_config.sched_policy) {
        case 0: 
            printf("# Naive, do nothing\n");
            break;
        case 1: 
            sort_with_used_threads(kernel_config, KERNEL_NUM);
            break;
        case 2:
            sort_with_used_shared_mem(kernel_config, KERNEL_NUM);
            break;
        default:
            fprintf(stderr, "# Invalid policy!\n");
            exit(-1);
    }
    // Policy 1: sort based on available thread number
    // samller (gridsize * blocksize) goes first
    // Policy 2: sort baesed on shared memory size
    // smaller (shared_mem * grid_size) goes first
    benchmark_config.kernel_config = kernel_config;

    Timer timer;
    for (int idx = 0; idx < KERNEL_NUM; idx++) {
        release(idx, streams[idx], in_d, n);   // release i th kernel in the sorted quue
        // cuda_ret = cudaDeviceSynchronize();
        // if(cuda_ret != cudaSuccess) fprintf(stderr, "Unable to launch kernel!\n");
        printf("Fuck gpu!\n");
    }
    cudaDeviceSynchronize();

    // Clean up works
    CleanUp(parser, kernel_config);
    for (int i = 0; i < KERNEL_NUM; i++) cudaStreamDestroy(streams[i]);
    
    // TODO: for each SM?

    return 0;
}

#endif