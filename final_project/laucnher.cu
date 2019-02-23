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
}MultiprocessorInfo;

int *scheduler(char *config_file) {
    KernelInfo *kernel_config = NULL;
    BenchmarkInfo benchmark_config;
    cudaDeviceProp dev_prop;
    cJSON *parser = cJSON_parse(config_file);
    cJSON *entry = NULL;
    cJSON *list_entry = NULL;

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

        if (!cJSON_IsString(kernel_name) || !cJSON_IsNumber(kernel_id) || !cJSON_IsNumber(grid_size) || !cJSON_IsNumber(block_size)) {
            fprintf("ERROR: invalid kernel_name/kernel_id/grid_size/block_size!\n");
            goto CleanUp;
        }
    
        kernel_config[i].kernel_name = strdup(kernel_name);
        kernel_config[i].kernel_id = kernel_id->valueint;
        kernel_config[i].grid_size = grid_size->valueint;
        if (block_size->valueint > dev_prop.maxThreadsPerBlock) {
            fprintf("# ERROR: invalid block size at kernel[%d]\n", i+1);
            goto CleanUp;
        }
        kernel_config[i].block_size = block_size->valueint;
        if (shared_mem->b=valueint > dev_prop.sharedMemPerBlock) {
            fprintf("# ERROR: invalid shared memory size at kernel[%d]\n", i+1);
            goto CleanUp;
        }
        kernel_config[i].shared_mem = shared_mem->valueint;
        

        kernel_entry = kernel_entry->next;
    }
    
    


CleanUp:
    cJSON_Delete(parser);
    free(kernel_config);

}