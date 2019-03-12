# Kernel Scheduling @ shmem = 0
## Prerequisite(s)
* [cJSON](https://github.com/DaveGamble/cJSON)

## Usage
```sh run.sh -i [filename] -s [scheduling policy: 0 for naive, 1 for optimum]```

## Methodology


## Baselines
### Platform
* GPU model:    GeForce GTX 1070 </br>
* CUDA version: Cuda 10.1

### Resource Limitation
* Number of multiprocessors:                  :  15
* Maximum number of blocks per multiprocessor:   32
* Maximum number of threads per multiprocessor:  2048
* Maximum number of threads per block:           1024

### Parameters
* Block duration: it's set in "duration" field in the json file. Measuring unit is millisecond. 
* Actual computation time: this is neglectable (~2us) compared to the set running time.
* Actual block running time: this equals to what is set to "duration", and does not change when block size or grid size is set differently.

### Single Kernel
#### Scenario 1: below the thread number limitation
* Duration: 20ms
* Block size: 1024
* Grid size: 15
* Actual kernel execution time: ~20ms
In this case, all 15 blocks can be assigned to SMs at the same time, and they finishes at the same time.

#### Scenario 2: above the thread number limitation
* Duration: 20ms
* Block size: 1024
* Grid size: 45
* Actual kernel execution time: ~40ms
In this case, the first 30 blocks are assigned to 15 SMs, and start running simultaneously. Due to the resource limitation, the last 15 blocks will wait until there are sufficient resources.



## Demo
(I represent i th block of j the kernel with K[j]:i)
### Scenario 1: blocked by kernels with larger block size (example1.json)

|          | Grid size | Block size | Duration(ms) |
|----------|-----------|------------|--------------|
| kernel 1 | 15 * 3    | 512        | 20           |
| kernel 2 | 15        | 1024       | 20           |
| kernel 3 | 15 * 5    | 256        | 20           |

#### Naive Scheduling Policy (FIFO)
* Kernel launching order: {k1, k2, k3}
* Expected results: (on one SM)
At t = 0, the resources are sufficient, 3 blocks of k1 will be assigned to a SM, the available number of threads on this SM = 2048 - 3 * 512 = 512. Since one block of k2 requires 1024 threads, it cannot start running at t = 0. 
At t = 20, all of the 3 blocks of k1 finishes, then k2 can start running. Now the number of available threads = 2048 - 1024 = 1024, so, another 4 blocks of k3 can be also assigned.
At t = 40, blocks which start running at t = 20 finishes, then the last block of K2 can start.
* Expected total running time: 3 * (single kernel time)
* Actual elapsed time: ~60ms </br>
![example1_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example1_naive.png?raw=true)

#### Optimal Scheduling Policy
* Kernel launching order: {k2, k1, k3}
* Expected results: (on one SM)
At t = 0, the resources are sufficient, one block of k2 will be assigned to a SM, the available number of threads on this SM = 2048 - 1024 = 1024, which is sufficient for 2 blocks of k1 to run simultaneously.
At t = 20, all of the blocks which started at t = 0 finishes. There are sufficient resources for all of the rest blocks.
* Expected total running time: 2 * (single kernel time)
* Actual elapsed time: ~40ms </br>
![example1_opt](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example1_opt.png?raw=true)

### Scenario 2: blocked by kernels with smaller block size (example2.json)
(A counterexample on "Longer execution time first")

|          | Grid size | Block size | Duration(ms) |
|----------|-----------|------------|--------------|
| kernel 1 | 15        | 1024       | 20           |
| kernel 2 | 15        | 512        | 16           |
| kernel 3 | 15        | 1024       | 12           |
| kernel 4 | 15        | 512        | 10           |

#### Naive Scheduling Policy (FIFO)
* Kernel launching order: {k1, k2, k3, k4}
* Expected results: (on one SM)
At the beginning, k1 and k2 can be launched, k3 will wait since the available number of threads = 2048 - 1024 - 512 = 512 is not sufficient, and it blocks k4. After k2 finishes execution, k3 can start at t = 16. Finally k4 can start after k1 finishes at t = 20.
* Elapsed time: ~30ms </br>
![example2_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example2_naive.png?raw=true)

#### Optimal Scheduling Policy
* Kernel launching order: {k1, k2, k4, k3}
* Expected results: (on one SM)
At the beginning, k1, k2 and k4 can be launched, and k3 will wait. After k2 finishes execution at t = 16, k3 can start running.
* Elapsed time: ~28ms </br>
![example2_opt](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example2_opt.png?raw=true)

### Scenario 3: blocked by kernels with smaller block duration (example3.json)
(A counterexample on "Bigger block size first")

|          | Grid size | Block size | Duration(ms) |
|----------|-----------|------------|--------------|
| kernel 1 | 15        | 1024       | 20           |
| kernel 2 | 15        | 1024       | 30           |
| kernel 3 | 15        | 512        | 50           |

#### Naive Scheduling Policy (FIFO)
* Kernel launching order: {k1, k2, k3}
* Expected results: (on one SM)
At t = 0, k1 and k2 will be launched. At t = 20, k1 finishes execution and k3 will start.
* Elapsed time: ~70ms </br>
![example3_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example3_naive.png?raw=true)

#### Optimal Scheduling Policy
* Kernel launching order: {k3, k2, k1}
* Expected results: (on one SM)
At t = 0, k3 and k2 will be launched. At t = 30, k2 finishes execution and k1 will start.
* Elapsed time: ~50ms </br>
![example3_opt](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example3_opt.png?raw=true)



## References
* <https://github.com/yalue/cuda_scheduling_examiner_mirror>
* [Binary Tree Bin Packing Algorithm in C](https://stackoverflow.com/questions/53536607/binary-tree-bin-packing-algorithm-in-c)
