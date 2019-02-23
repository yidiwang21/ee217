#include <stdio.h>
#include <cjson/cJSON.h>






// keep the information for each kernel
typedef struct {
    char *kernel_name;
    int kernel_id;
    int block_size;     // # of threads per block
    int grid_size;      // # of blcoks
    int shared_mem;
    int launch_time;
}KernelInfo;

// keep the information for the benchmark
typedef struct {
    int kernel_num;
    int iteration;
    KernelInfo *kernel_config;
}BenchmarkInfo;

typedef struct {
    int available_thread_num;
    int available_shared_mem_size;  // byte
    int available_register_num;
}ResourceInfo;


BenchmarkInfo benchmark_config;


// input value: kernel_id
void release(int kernel_id, cudaStream_t stream_id) {
    KernelInfo kernel_config_tmp;
    // TODO: map kernel_id to one kernel, then launch
    // better to use a heap
    for (int i = 0; i < benchmark_config.kernel_num; i++) {
        if (benchmark_config.kernel_config[i].kernel_id == kernel_id) {
            kernel_config_tmp = benchmark_config.kernel_config[i];
            switch(kernel_config_tmp.shared_mem) {  // test for 1070
                case 0:
                    kernel_shared_mem_0 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 0, stream_id>>> (kernel_config_tmp.duration);
                    break;
                case 1024:
                    kernel_shared_mem_1024 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 1024, stream_id>>> (kernel_config_tmp.duration);
                    break;
                case 4096:
                    kernel_shared_mem_4096 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 4096, stream_id>>> (kernel_config_tmp.duration);
                    break;
                case 16384:
                    kernel_shared_mem_16384 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 16384, stream_id>>> (kernel_config_tmp.duration);
                    break;
                case 49152:
                    kernel_shared_mem_49152 <<<kernel_config_tmp.grid_size, kernel_config_tmp.block_size, 49152, stream_id>>> (kernel_config_tmp.duration);
                    break;
                default:
                    fprintf("# Error: Invalid shared memory size!\n");
            }
        }
    }
}

int *scheduler(char *config_file) {
    KernelInfo *kernel_config = NULL;
    cudaDeviceProp dev_prop;
    cJSON *parser = cJSON_parse(config_file);
    cJSON *entry = NULL;
    cJSON *list_entry = NULL;

    // Step 1: get kernel information from config file
    if (parser == NULL) {
        fprintf("ERROR: invalid kernel config file!\n");
        goto CleanUp;
    }

    entry = cJSON_GetObjectItem(parser, "kernel_number");
    if (!entry || entry->type != cJSON_Number) {
        fprintf("ERROR: invalid kernel number!\n");
        goto CleanUp;
    }
    benchmark_config.kernel_num = entry->valueint;

    entry = cJSON_GetObjectItem(parser, "iteration");
    if (!entry || entry->type != cJSON_Number) {
        fprintf("ERROR: invalid iteration!\n");
        goto CleanUp;
    }
    benchmark_config.iteration = entry->valueint;

    kernel_entry = cJSON_GetObjectItem(parser, "multikernel");
    kernel_config = (KernelInfo*)malloc(benchmark_config.kernel_num * sizeof(KernelInfo));

    for (int i = 0; i < benchmark_config.kernel_num; i++) {
        cJSON *kernel_name = cJSON_GetObjectItem(kernel_entry, "kernel_name");
        cJSON *kernel_id = cJSON_GetObjectItem(kernel_entry, "kernel_id");
        cJSON *grid_size = cJSON_GetObjectItem(kernel_entry, "grid_size");
        cJSON *block_size = cJSON_GetObjectItem(kernel_entry, "block_size");
        cJSON *shared_mem = cJSON_GetObjectItem(kernel_entry, "shared_mem");
        cJSON *duration = cJSON_GetObjectItem(kernel_entry, "durbation");

        if (!cJSON_IsString(kernel_name) || !cJSON_IsNumber(kernel_id) || !cJSON_IsNumber(grid_size) || !cJSON_IsNumber(block_size)) {
            fprintf("ERROR: invalid kernel_name/kernel_id/grid_size/block_size!\n");
            goto CleanUp;
        }
    
        kernel_config[i].kernel_name = strdup(kernel_name);
        kernel_config[i].kernel_id = kernel_id->valueint;
        kernel_config[i].grid_size = grid_size->valueint;
        kernel_config[i].duration = duration->valueint;
        if (block_size->valueint > dev_prop.maxThreadsPerBlock) {
            fprintf("# ERROR: invalid block size at kernel[%d]\n", i+1);
            goto CleanUp;
        }
        kernel_config[i].block_size = block_size->valueint;
        if (shared_mem->valueint < 0 || shared_mem->valueint > static_cast<int>(dev_prop.sharedMemPerBlock)) {
            fprintf("# ERROR: invalid shared memory size at kernel[%d]\n", i+1);
            goto CleanUp;
        }
        kernel_config[i].shared_mem = shared_mem->valueint;

        kernel_entry = kernel_entry->next;
    }
    
    // Step 2: scheduling algorithm
    // TODO:
    // How to know the number of registers a kernel is going to use?
    // How to sort the kernels based on their "size"?
    // Don't need to control release time, just need re-order
    int *launch_queue = (int*)malloc(benchmark_config.kernel_num * sizeof(int));

    int KERNEL_NUM = benchmark_config.kernel_num;

    cudaStream_t *streams;
    streams = (cudaStream_t*)malloc(KERNEL_NUM * sizeof(cudaStream_t));
    for (int i = 0; i < KERNEL_NUM; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);


    // TODO: tomorrow, Feb 23
    // 1. benchmarks
    // 2. alg
    


CleanUp:
    cJSON_Delete(parser);
    free(kernel_config);

}