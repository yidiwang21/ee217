#include "multikernel.cuh"

// init MultiKernel class with the input arguments
MultiKernel::MultiKernel(char *config_file, int sp, int draw) {
    parser = cJSON_Parse(config_file);
    if (parser == NULL) {
        fprintf(stderr, "# Invalid kernel config file!\n");
        cleanUp();
    }
    sched_policy = sp;
    draw_timeline = draw;
}

Node* MultiKernel::newNode() {
    Node *new_node = (Node*)malloc(sizeof(Node));
    new_node->kernel_id = 0;

    return new_node;
}

Node* MultiKernel::splitNode(Node *node, int w, int h, int kid) {
    node->used = 1;
    node->kernel_id = kid;

    node->left = newNode();
    node->left->used = 0;
    node->left->growable = 0;
    node->left->kernel_id = 0;
    node->left->parent = node;
    node->left->left = NULL;
    node->left->right = NULL;
    node->left->start_point.x = node->start_point.x;
    node->left->start_point.y = node->start_point.y + h;
    // (*node)->left->end_point->x = (*node)->end_point->x;
    // (*node)->left->end_point->y = (*node)->end_point->;
    node->left->width = node->width;
    node->left->height = node->height - h;

    node->right = newNode();
    node->right->used = 0;
    node->right->growable = 1;
    node->right->kernel_id = 0;
    node->right->parent = node;
    node->right->left = NULL;
    node->right->right = NULL;
    node->right->start_point.x = node->start_point.x + w;
    node->right->start_point.y = node->start_point.y;
    node->right->width = node->width - w;
    node->right->height = node->height;
    

    // FIXME: 
    node->width = w;
    node->height = h;
    node->growable = 0;

    // FIXME: tunning
    if (node->start_point.x + node->width == root->width && node != root) {
        node->right->width = 0;
        node->right->height = 0;
        node->right->closed = 1;
    }

    return node;
}

Node* MultiKernel::findBestFit(Node *node, int w, int h) {
    if (node->used == 1) {
        Node *ret;
        if (ret = findBestFit(node->right, w, h)){
            // printf("right\n");
            return ret;
        }
        else {
            // printf("left\n");
            return findBestFit(node->left, w, h);
        }
    }else if ((w <= node->width) && (h <= node->height)) {
        return node;
    }   
    else {
        return NULL;
    }
}

void MultiKernel::updateParentsRight(Node *node, int w, int h, int stp) {
    if (node->parent == this->root)
        return;
    else if(node == this->root) return;
    if (node == NULL || node->parent == NULL || node->parent->parent == NULL || node->parent->parent->right == NULL) {
        return;
    }
    node = node->parent->parent->right;

    // printf("judge range = (%d, %d)\n", stp, stp+w);
    // printf("point = %d\n", node->start_point.x);
    while (1) {
        if (node->parent == root || node == root)
            break;

        if (node->start_point.x < stp + w && node->start_point.x >= stp) {
            if (node->height >= h) {
                node->height -= h;
            }       
        }
        if (node->parent == NULL || node->parent->parent == NULL || node->parent->parent->right == NULL) continue;
        node = node->parent->parent->right;
    }
    return;
}

Node* MultiKernel::growNode(Node *node, int w, int h, int kid) {
    // bool canGrowRight = (w < root->width);
    if (node == root) { // then grow right
        root->used = 1;
        root->start_point.x = 0;
        root->start_point.y = 0;
        root->width = root->width + w;
        root->height = root->height;
        // root->left = root;
        root->right->start_point.x = root->width;
        root->right->start_point.y = 0;
        root->right->width = w;
        root->right->height = root->height;

        Node *new_node = findBestFit(root, w, h);
        if (new_node != NULL) 
            return splitNode(new_node, w, h, kid);
        else
            return NULL;
    }else {
        node->used = 1;
        node->growable = 0;
        node->width = w;

        root->used = 1;
        root->start_point.x = 0;
        root->start_point.y = 0;
        root->right->start_point.x = root->width;
        root->right->start_point.y = 0;
        root->right->width = node->start_point.x + node->width - root->width;

        // best fit is node i this case
        return splitNode(node, w, h, kid);
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

Node* MultiKernel::searchNode(Node *node, int key, Node *node_const) {
    if (node != NULL) {
        if (node->start_point.x == key && node != node_const && node->closed != 1)
            return node;
        else {
            Node* found_node = searchNode(node->left, key, node_const);
            if (found_node == NULL) {
                found_node = searchNode(node->right, key, node_const);
            }
            return found_node;
        }
    }else 
        return NULL;
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
}

void MultiKernel::sortDurationAscending() {
    bool swapped;
    do {
        swapped = false;
        for (int i = 0; i < kernel_num - 1; i++) {
            if (kernel_list[i].duration  > kernel_list[i+1].duration) {
                std::swap(kernel_list[i], kernel_list[i+1]);
                swapped = true; 
            }
        }
    } while(swapped);
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
        // printf("=======================================\n");
        // printf("# Assigning block %d:\n", i+1);
        node = findBestFit(root, block_list[i].duration, block_list[i].block_size);
        if (node != NULL) {
            block_node = splitNode(node, block_list[i].duration, block_list[i].block_size, block_list[i].kernel_id);
            block_list[i].start_time = block_node->start_point.x;
            // printf("x = %d\n", block_node->start_point.x);
            // printf("y = %d\n", block_node->start_point.y);
            
            // FIXME:
            updateParentsRight(block_node, (block_node)->width, (block_node)->height, (block_node)->start_point.x);
            // printf("right space width = %d\n", block_node->right->width);
            // printf("right space height = %d\n", block_node->right->height);
            // printf("above space width = %d\n", block_node->left->width);
            // printf("above space height = %d\n", block_node->left->height);
        }
        else if (node == NULL ){
            // printf("Growing node... height = %d\n", block_list[i].block_size);
            int min = findMinUnusedToGrow(root, block_list[i].block_size);
            // printf("The min x to grow is %d\n", min);
            Node* node_to_grow = searchNode(root, min, NULL);
            // printf("The parent of growing node is %d\n", node_to_grow->parent->kernel_id);
            // printf("found node node->height is %d\n", node_to_grow->height);

            block_node = growNode(node_to_grow, block_list[i].duration, block_list[i].block_size, block_list[i].kernel_id);
            block_list[i].start_time = block_node->start_point.x;
            // printf("x = %d\n", block_node->start_point.x);
            // printf("y = %d\n", block_node->start_point.y);

            updateParentsRight(block_node, (block_node)->width, (block_node)->height, (block_node)->start_point.x);

            // printf("right space width = %d\n", block_node->right->width);
            // printf("right space height = %d\n", block_node->right->height);
            // printf("above space width = %d\n", block_node->left->width);
            // printf("above space height = %d\n", block_node->left->height); 
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

    if (sched_policy == 2) {
        printf("# Scheduling policy 2: minimum execution time...\n");
        scheduler();
    }else if (sched_policy == 1) {
        printf("# Scheduling policy 1: minimum AWT...\n");
        sortDurationAscending();
    }else {
        printf("# Naive scheduling policy...\n");
    }

    printf("# Kernel submission order: \n");
    for (int k = 0; k < kernel_num; k++) {
        // printf("start time of kernel[%d] with kernel_id(%d): %d\n", k, kernel_list[k].kernel_id, kernel_list[k].start_time);
        printf("Kernel[%d] with kernel_id(%d)\n", k, kernel_list[k].kernel_id);
    }

    cudaError_t cuda_ret;

    cudaStream_t *streams;
    streams = (cudaStream_t*)malloc(kernel_num * sizeof(cudaStream_t));
    for (int i = 0; i < kernel_num; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaDeviceSynchronize();

    printf("Kernel number: %d\n", kernel_num);
    printf("Launching kernel...\n"); fflush(stdout);
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

    if (draw_timeline == 1)
        printBlocks();
}

void MultiKernel::printBlocks() {
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