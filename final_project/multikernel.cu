#include "multikernel.cuh"

// init MultiKernel class with the input arguments
MultiKernel::MultiKernel(char *config_file) {
    parser = cJSON_Parse(config_file);    
    if (parser == NULL) {
        fprintf(stderr, "# Invalid kernel config file!\n");
        cleanUp();
    }
}

Node* MultiKernel::newNode() {
    Node *new_node = (Node*)malloc(sizeof(Node));
    new_node->kernel_id = 0;

    return new_node;
}

Node* MultiKernel::splitNode(Node **node, int w, int h, int kid) {
    (*node)->used = 1;
    (*node)->kernel_id = kid;

    (*node)->left = newNode();
    (*node)->left->used = 0;
    (*node)->left->growable = 0;
    (*node)->left->kernel_id = (*node)->kernel_id;
    (*node)->left->parent = (*node);
    (*node)->left->left = NULL;
    (*node)->left->right = NULL;
    (*node)->left->start_point.x = (*node)->start_point.x;
    (*node)->left->start_point.y = (*node)->start_point.y + h;
    // (*node)->left->end_point->x = (*node)->end_point->x;
    // (*node)->left->end_point->y = (*node)->end_point->;
    (*node)->left->width = (*node)->width;
    (*node)->left->height = (*node)->height - h;

    (*node)->right = newNode();
    (*node)->right->used = 0;
    (*node)->right->growable = 1;
    (*node)->right->kernel_id = 0;
    (*node)->parent = (*node);
    (*node)->right->left = NULL;
    (*node)->right->right = NULL;
    (*node)->right->start_point.x = (*node)->start_point.x + w;
    (*node)->right->start_point.y = (*node)->start_point.y;
    (*node)->right->width = (*node)->width - w;
    (*node)->right->height = (*node)->height;

    return (*node);
}

Node* MultiKernel::findBestFit(Node *node, int w, int h) {
    if (node->used == 1) {
        Node *ret;
        if (ret = findBestFit(node->right, w, h)){
            printf("right\n");
            return ret;
        }
        else {
            printf("left\n");
            return findBestFit(node->left, w, h);
        }
    }else if ((w <= node->width) && (h <= node->height)) {
        return node;
    }   
    else {
        return NULL;
    }
}

Node* MultiKernel::growNode(Node *node, int w, int h, int kid) {
    // bool canGrowRight = (w < root->width);
    if (node == root) { // then grow right
        printf("The node to grow is root.\n");
        root->used = 1;
        root->start_point.x = 0;
        root->start_point.y = 0;
        root->width = root->width + w;
        root->height = root->height;
        root->left = root;
        root->right->start_point.x = root->width;
        root->right->start_point.y = 0;
        root->right->width = w;
        root->right->height = root->height;

        Node *new_node = findBestFit(root, w, h);
        if (new_node != NULL) 
            return splitNode(&new_node, w, h, kid);
        else
            return NULL;
    }else {
        printf("The node to grow is not root.\n");
        node->used = 1;
        node->growable = 0;
        node->width = w;

        root->used = 1;
        root->start_point.x = 0;
        root->start_point.y = 0;
        root->right->start_point.x = root->width;
        root->right->start_point.y = 0;
        root->right->width = node->start_point.x + node->width - root->width;
        root->right->height = root->height;

        // best fit is node i this case
        return splitNode(&node, w, h, kid);
        // Node *new_node = findBestFit(node, w, h);
        // if (new_node != NULL) 
        //     return splitNode(&new_node, w, h, kid);
        // else
        //     return NULL;
    }
}

int MultiKernel::findMinUnusedToGrow(Node *node, int h) {
    // get the min available value
    int min = INT_MAX;
    if (node->used == 0 && node->height >= h && node->growable == 1)
        min = node->start_point.x;
    if (node->left != NULL)
        min = MIN(min, findMinUnusedToGrow(node->left, h));
    if (node->right != NULL)
        min = MIN(min, findMinUnusedToGrow(node->right, h));
    return min;
}

Node* MultiKernel::searchNode(Node *node, int key) {
    if (node != NULL) {
        if (node->start_point.x == key)
            return node;
        else {
            Node* found_node = searchNode(node->left, key);
            if (found_node == NULL) {
                found_node = searchNode(node->right, key);
            }
            return found_node;
        }
    }else 
        return NULL;
}

int MultiKernel::maxBlockSize() {
    int area;
    int max_area;
    for (int i = 0; i < kernel_num; i++) {
        max_area = MAX(kernel_list[i].duration * kernel_list[i].block_size, area);
        area = kernel_list[i].duration * kernel_list[i].block_size;
    }
}

void MultiKernel::blockInfoInit() {
    count = 0;  // number of blocks averagely per SM
    for (int i = 0; i < kernel_num; i++) {
        for (int j = 0; j < (kernel_list[i].grid_size-1)/SM_NUM+1; j++) {
            count++;
        }
    }

    block_list = (BlockInfo*)malloc(sizeof(BlockInfo) * count);

    int idx = 0;
    for (int i = 0; i < kernel_num; i++) {
        for (int j = 0; j < (kernel_list[i].grid_size-1)/SM_NUM+1; j++) {
            block_list[idx+j].kernel_id = kernel_list[i].kernel_id;
            block_list[idx+j].block_size = kernel_list[i].block_size;
            block_list[idx+j].duration = kernel_list[i].duration;
        }
        idx += (kernel_list[i].grid_size-1)/SM_NUM+1;
    }
}

void MultiKernel::sortDurationDecending() {
    bool swapped;
    do {
        swapped = false;
        for (int i = 0; i < kernel_num - 1; i++) {
            if (kernel_list[i].duration < kernel_list[i+1].duration || (kernel_list[i].duration == kernel_list[i+1].duration && kernel_list[i].block_size < kernel_list[i+1].block_size)) {
                std::swap(kernel_list[i], kernel_list[i+1]);
                swapped = true; 
            }
        }
    } while(swapped);   
}

void MultiKernel::sortStartTimeAscending() {
    int tmp = 0;
    int c = 0;
    for (int i = 0; i < kernel_num; i++) {
        c = 0;
        tmp = 0;
        for (int j = 0; j < count; j++) {
            if (kernel_list[i].kernel_id == block_list[j].kernel_id) {
                if (c > 0) {
                    kernel_list[i].start_time = MIN(block_list[j].start_time, tmp);
                }else {
                    kernel_list[i].start_time = block_list[j].start_time;
                }
                tmp = kernel_list[i].start_time;
                c++;
            }
        }
    }

    bool swapped;
    do {
        swapped = false;
        for (int i = 0; i < kernel_num - 1; i++) {
            if (kernel_list[i].start_time > kernel_list[i+1].start_time) {
                std::swap(kernel_list[i], kernel_list[i+1]);
                swapped = true; 
            }
        }
    } while(swapped); 

    for (int k = 0; k < kernel_num; k++) {
        printf("start time of kernel[%d] with kernel_id(%d): %d\n", k, kernel_list[k].kernel_id, kernel_list[k].start_time);
    }
}

void MultiKernel::scheduler() {
    sortDurationDecending();
    blockInfoInit();

    // Step 1: initialize resources
    printf("# Initializing resource map...\n");
    // root = newNode();
    root->used = 0;
    root->height = devProp.maxThreadsPerMultiProcessor;
    root->width = kernel_list[0].duration;          // TODO: this is growable
    root->start_point.x = 0;
    root->start_point.y = 0;
    root->growable = 1;
    root->left = NULL;
    root->right = NULL;
    
    // Step 2: fit
    printf("# Starting to fit...\n");
    for (int i = 0; i < count; i++) {
        Node *node;
        Node *block_node;
        printf("Assigning block %d:\n", i);
        node = findBestFit(root, block_list[i].duration, block_list[i].block_size);
        if (node != NULL) {
            block_node = splitNode(&node, block_list[i].duration, block_list[i].block_size, block_list[i].kernel_id);
            block_list[i].start_time = block_node->start_point.x;
            printf("x = %d\n", block_node->start_point.x);
            printf("y = %d\n", block_node->start_point.y);
        }
        else if (node == NULL ){
            printf("Growing node...\n");
            int min = findMinUnusedToGrow(root, block_list[i].block_size);
            printf("The min x to grow is %d\n", min);
            Node* node_to_grow = searchNode(root, min);
            printf("Searching the node to grow...\n");

            block_node = growNode(node_to_grow, block_list[i].duration, block_list[i].block_size, block_list[i].kernel_id);
            block_list[i].start_time = block_node->start_point.x;
            printf("x = %d\n", block_node->start_point.x);
            printf("y = %d\n", block_node->start_point.y);
        }
    }
    sortStartTimeAscending();
}

void MultiKernel::kernelInfoInit() {
    cJSON *entry = NULL;
    cJSON *iter = NULL;
    cudaGetDeviceProperties(&devProp, 0);

    // printf("# Parsing kernel config file...\n");
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
        cJSON *kernel_id_entry = cJSON_GetObjectItem(iter, "kernel_id");

        kernel_list[idx].grid_size = grid_size_entry->valueint;
        kernel_list[idx].block_size = block_size_entry->valueint;   // must be in a range
        kernel_list[idx].duration = duration_entry->valueint;
        kernel_list[idx].shared_mem = shared_mem_entry->valueint;
        kernel_list[idx].kernel_id = kernel_id_entry->valueint;
        // printf("        grid size = %d\n", kernel_list[idx].grid_size);
        // printf("        block size = %d\n", kernel_list[idx].block_size);
        // printf("        duration = %d\n", kernel_list[idx].duration);
        // printf("        shared mem size = %d\n", kernel_list[idx].shared_mem);
        
        idx++;
    }
}

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
    kernelInfoInit();
    GPUResourceInit();

    scheduler();

    cudaError_t cuda_ret;

    cudaStream_t *streams;
    streams = (cudaStream_t*)malloc(kernel_num * sizeof(cudaStream_t));
    for (int i = 0; i < kernel_num; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaDeviceSynchronize();

    printf("Launching kernel...\n"); fflush(stdout);
    printf("Kernel number: %d\n", kernel_num);
    for (int i = 0; i < kernel_num; i++) {
        GPUSpin <<<kernel_list[i].grid_size, kernel_list[i].block_size, 0, streams[i]>>> (kernel_list[i].duration * 1000 * 1000, kernel_list[i].block_times_d, kernel_list[i].block_smids_d);
    }
    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "Unable to launch kernel!\n"); exit(-1); }

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
        #if 0
            printf("Block index: %d\n", j);
            printf("kernel id: %d\n", i);
            printf("SM id: %d\n", kernel_list[i].block_smids[j]);
            printf("start time: %lu\n", kernel_list[i].block_times[j*2]);
            printf("stop time: %lu\n", kernel_list[i].block_times[j*2+1]);
            printf("elapsed time: %lu\n\n", kernel_list[i].block_times[j*2+1] - kernel_list[i].block_times[j*2]);
        #endif
        }
    }
}

void MultiKernel::cleanUp() {
    cJSON_Delete(parser);
    free(kernel_list);
    exit(0);
}