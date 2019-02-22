#include <stdio.h>
#include <cjson/cJSON.h>






// keep the information for each kernel
typedef struct {
    char* kernel_name;
    int kernel_id;
    int block_num;
    int grid_num;
    int thread_num;
    int launch_time;
}KernelInfo;

// keep the information for the benchmark
typedef struct {
    int kernel_num;

}BenchmarkInfo;

void kernelLauncher() {
    
}