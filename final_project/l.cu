#include "launcher.h"

// init launcher class with the input arguments
Launcher::Launcher(char *config_file) {
    cJSON *parser = cJSON_Parse(config_file);
}

void Launcher::kernelInfoInit() {
    // temporary variables
    cJSON *entry = NULL;
    cJSON *iter = NULL;
    cudaGetDeviceProperties(&devProp, 0);

    printf("# Parsing kernel config file...\n");
    printf("[\n");
    // Step 1: get kernel information from config file
    if (parser == NULL) {
        fprintf(stderr, "# Invalid kernel config file!\n");
        cleanUp();
    }
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

    kernel_list = (KernelInfo*)malloc(kernel_num * sizeof(KernelInfo));

    entry = cJSON_GetObjectItem(parser, "kernel_entry");
    if (!entry) {
        fprintf(stderr, "Invalid kernel entry!\n");
        cleanUp();
    }
    int idx = 0;
    cJSON_ArrayForEach(iter, entry) {
        cJSON *grid_size_entry = cJSON_GetObjectItem(iter, "grid_size");
        cJSON *block_size_entry = cJSON_GetObjectItem(iter, "block_size");
        cJSON *shared_mem_entry = cJSON_GetObjectItem(iter, "shared_mem");
        cJSON *duration_entry = cJSON_GetObjectItem(iter, "duration");

        kernel_list[idx].grid_size = grid_size_entry->valueint;
    }




}

void Launcher::cleanUp() {
    cJSON_Delete(parser);
    // TODO: 
    exit(0);
}