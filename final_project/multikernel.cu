#include "multikernel.cuh"

// init MultiKernel class with the input arguments
MultiKernel::MultiKernel(char *config_file) {
    parser = cJSON_Parse(config_file);    
    if (parser == NULL) {
        fprintf(stderr, "# Invalid kernel config file!\n");
        cleanUp();
    }
}

void MultiKernel::kernelInfoInit() {
    cJSON *entry = NULL;
    cJSON *iter = NULL;
    // cudaGetDeviceProperties(&devProp, 0);

    printf("# Parsing kernel config file...\n");
    // printf("[\n");

    entry = cJSON_GetObjectItem(parser, "kernel_number");
    if (!entry || entry->type != cJSON_Number) {
        fprintf(stderr, "# Invalid kernel number!\n");
        cleanUp();
    }
    kernel_num = entry->valueint;
    entry = NULL;
    entry = cJSON_GetObjectItem(parser, "sched_policy");
    if (!entry || entry->type != cJSON_Number || entry->valueint > 2 || entry->valueint < 0) {
        fprintf(stderr, "Invalid policy! (must be 1 ,2, 3)\n");
        cleanUp();
    }
    sched_policy = entry->valueint;
    entry = NULL;

    entry = cJSON_GetObjectItem(parser, "kernel_entry");
    if (!entry) {
        fprintf(stderr, "Invalid kernel entry!\n");
        cleanUp();
    }

    kernel_list = (KernelInfo*)malloc(kernel_num * sizeof(KernelInfo));
    
    int idx = 0;
    cJSON_ArrayForEach(iter, entry) {
        cJSON *grid_size_entry = cJSON_GetObjectItem(iter, "grid_size");
        cJSON *block_size_entry = cJSON_GetObjectItem(iter, "block_size");
        cJSON *shared_mem_entry = cJSON_GetObjectItem(iter, "shared_mem");
        cJSON *duration_entry = cJSON_GetObjectItem(iter, "duration");

        kernel_list[idx].grid_size = grid_size_entry->valueint;
        printf("        grid size = %d\n", kernel_list[idx].grid_size);
        kernel_list[idx].block_size = block_size_entry->valueint;   // must be in a range
        printf("        block size = %d\n", kernel_list[idx].block_size);
        kernel_list[idx].duration = duration_entry->valueint;
        printf("        duration = %d\n", kernel_list[idx].duration);
        kernel_list[idx].shared_mem = shared_mem_entry->valueint;
        printf("        shared mem size = %d\n", kernel_list[idx].shared_mem);
        
        idx++;
    }
}

// TODO: 
void MultiKernel::scheduler() {}

// this should take sorted kernels as input
void MultiKernel::GPUResourceInit() {
    for (int i = 0; i < kernel_num; i++) {
        kernel_list[i].block_times = (uint64_t *)malloc(sizeof(uint64_t) * kernel_list[i].grid_size * 2);
        kernel_list[i].block_smids = (uint32_t *)malloc(sizeof(uint32_t) * kernel_list[i].grid_size);
        memset(kernel_list[i].block_times, 0, sizeof(kernel_list[i].block_times));
        memset(kernel_list[i].block_smids, 0, sizeof(kernel_list[i].block_smids));

        cudaMalloc((void**) &kernel_list[i].block_times_d, sizeof(uint64_t) * kernel_list[i].grid_size * 2);
        cudaMalloc((void**) &kernel_list[i].block_smids_d, sizeof(uint32_t) * kernel_list[i].grid_size);
    }
    cudaDeviceSynchronize();
    for (int i = 0; i < kernel_num; i++) {
        cudaMemcpy(kernel_list[i].block_times_d, kernel_list[i].block_times, sizeof(uint64_t) * kernel_list[i].grid_size * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(kernel_list[i].block_smids_d, kernel_list[i].block_smids, sizeof(uint32_t) * kernel_list[i].grid_size, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
}

void MultiKernel::kernelLauncher() {
    // scheduler();
    kernelInfoInit();
    GPUResourceInit();

    cudaError_t cuda_ret;

    cudaStream_t *streams;
    streams = (cudaStream_t*)malloc(kernel_num * sizeof(cudaStream_t));
    for (int i = 0; i < kernel_num; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaDeviceSynchronize();

    for (int i = 0; i < kernel_num; i++) {
        GPUSpin <<<kernel_list[i].grid_size, kernel_list[i].block_size, 0, streams[i]>>> (kernel_list[i].duration, kernel_list[i].block_times_d, kernel_list[i].block_smids_d);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < kernel_num; i++) {
        cudaMemcpy(kernel_list[i].block_times, kernel_list[i].block_times_d, sizeof(uint64_t) * kernel_list[i].grid_size * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(kernel_list[i].block_smids, kernel_list[i].block_smids_d, sizeof(uint32_t) * kernel_list[i].grid_size, cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();

    // print
    for (int i = 0; i < kernel_num; i++) {
        printf("=========================================\n");
        for (int j = 0; j < kernel_list[i].grid_size; j++) {
            // nano sec timer to ms
            kernel_list[i].block_times[j*2] = (kernel_list[i].block_times[j*2] / 1000 / 1000) % 10000;
            kernel_list[i].block_times[j*2+1] = (kernel_list[i].block_times[j*2+1] / 1000 / 1000) % 10000;
            printf("Block index: %d\n", j);
            printf("kernel id: %d\n", i);
            printf("SM id: %d\n", kernel_list[i].block_smids[j]);
            printf("start time: %lu\n", kernel_list[i].block_times[j*2]);
            printf("stop time: %lu\n", kernel_list[i].block_times[j*2+1]);
            printf("elapsed time: %lu\n\n", kernel_list[i].block_times[j*2+1] - kernel_list[i].block_times[j*2]);
        }
    }
}

void MultiKernel::cleanUp() {
    cJSON_Delete(parser);
    free(kernel_list);
    exit(0);
}