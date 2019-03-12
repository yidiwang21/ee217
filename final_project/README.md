# GPU Block-Level Kernel Scheduling @ shmem = 0
## Prerequisite(s)
* [cJSON](https://github.com/DaveGamble/cJSON)

## Usage
```sh run.sh -i [filename] -s [scheduling policy: 0 for naive, 1 for minimum AWT, 2 for minimum exe time]```
## Platform
* GPU model:    GeForce GTX 1070 </br>
* CUDA version: Cuda 10.1

## Resource Limitation
* Number of multiprocessors:                  :  15
* Maximum number of blocks per multiprocessor:   32
* Maximum number of threads per multiprocessor:  2048
* Maximum number of threads per block:           1024

## Methodology
### Naive Policy: FIFO

### Scheduling Policy 1: SJF
This policy is not optimal (having minimum AWT) when applied to GPU tasks because resources limitation may prevent the blocks in one kernel to run concurrently.

### Scheduling Policy 2: Minimum Execution Time (MET)
In this case, the optimal solution has an minimum average execution time. The solution is achieved by maximize GPU resource utilization. In this project, only the cases with 0 shared memory have been considered.

#### Problem Simplification
In order to simulate the block timelines, the maximum average number of blocks on each SM is calculated. For example, a kernel has 16 blocks and 1070 has 15 SMs, then I considered is (16 - 1) / 15 + 1 = 2 blocks per SM.

#### Dynamic Resources Mapping
I created a rectangle to illustrate the resource usage. The y axis represent the number of used threads on one SM, and it has a ceiling of 2048 in this case. The x axis represent the time, and the rectangle is growable in this direction. The map is splitted by smaller rectangles according to the size of assigned blocks. </br>
![timeline_policy2](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/timeline_policy2.png?raw=true)

#### Growing 2D Bin Packing
At initialization, the kernels are sorted with block duration in descending order. 
A binary tree is used to store the sub-rectangles. Every time a block is assigned, it will split the newly occupied space into two smaller rectangles, which will be the children of the current space. The left child will always be marked "ungrowable". </br>
Each assignment of a block, I will look for an available space it can fit into. Since the current block can not have a longer duration than the previous ones, I don't need to worry about the situation that a space at the left is found to be the best to grow, then it may overlap to its right occupied spaces. 

#### Greedy Choice
The launching order of the kernels is determined by "best fit (to spaces that have sufficient resources)" policy. If it does not exist, then grow the rectangle that has sufficient resources and has the minimum start point.


## Baselines
### Parameters
* Block duration: it's set in "duration" field in the json file. Measuring unit is millisecond. 
* Actual computation time: this is neglectable (~2us) compared to the set running time.
* Actual block running time: this equals to what is set to "duration", and does not change when block size or grid size is set differently.

### Single Kernel
#### Scenario 1: below the thread number limitation
* Duration: 20ms
* Block size: 1024
* Grid size: 15
* Actual kernel execution time: ~20ms </br>
In this case, all 15 blocks can be assigned to SMs at the same time, and they finishes at the same time.

#### Scenario 2: above the thread number limitation
* Duration: 20ms
* Block size: 1024
* Grid size: 45
* Actual kernel execution time: ~40ms </br>
In this case, the first 30 blocks are assigned to 15 SMs, and start running simultaneously. Due to the resource limitation, the last 15 blocks will wait until there are sufficient resources.

## Evaluation
(I represent i th block of j th kernel with K[j]:i)
### Measurements
* Execution time
* Average response time 
* Average waiting time

### Scenario 1: blocked by kernels with larger block size (example1.json)

|          | Grid size | Block size | Duration(ms) |
|----------|-----------|------------|--------------|
| kernel 1 | 15 * 3    | 512        | 20           |
| kernel 2 | 15        | 1024       | 20           |
| kernel 3 | 15 * 5    | 256        | 20           |

#### Naive Scheduling Policy (FIFO)
* Kernel launching order: {k1, k2, k3}
* Behavious: (average block number on one SM)
At t = 0, the resources are sufficient, 3 blocks of k1 will be assigned to a SM, the available number of threads on this SM = 2048 - 3 * 512 = 512. Since one block of k2 requires 1024 threads, it cannot start running at t = 0. 
At t = 20, all of the 3 blocks of k1 finishes, then k2 can start running. Now the number of available threads = 2048 - 1024 = 1024, so, another 4 blocks of k3 can be also assigned.
At t = 40, blocks which start running at t = 20 finishes, then the last block of k3 can start.
![example1_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example1_naive.png?raw=true)

#### Scheduling Policy 1 (SJF)
* Kernel launching order: {k1, k2, k3}
* Behavious: (average block number on one SM) The same as naive policy.

#### Scheduling Policy 2 (MET)
* Kernel launching order: {k2, k1, k3}
* Behavious: (average block number on one SM)
At t = 0, the resources are sufficient, one block of k2 will be assigned to a SM, the available number of threads on this SM = 2048 - 1024 = 1024, which is sufficient for 2 blocks of k1 to run simultaneously.
At t = 20, all of the blocks which started at t = 0 finishes. There are sufficient resources for all of the rest blocks.
![example1_opt](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example1_opt.png?raw=true)

|         | Execution time (ms) | Average response time (ms) | Average waiting time (ms) |
|---------|---------------------|----------------------------|---------------------------|
| Naive   | 60                  | (20+40+60)/3=40            | (0+20+20)/3=13.33         |
| SJF     | 60                  | (20+40+60)/3=40            | (0+20+20)/3=13.33         |
| MET     | 40                  | (20+40)/3=20               | (0+0+20)/3=6.67           |

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
* Behavious: (average block number on one SM)
At the beginning, k1 and k2 can be launched, k3 will wait since the available number of threads = 2048 - 1024 - 512 = 512 is not sufficient, and it blocks k4. After k2 finishes execution, k3 can start at t = 16. Finally k4 can start after k1 finishes at t = 20. </br>
![example2_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example2_naive.png?raw=true)

#### Scheduling Policy 1 (SJF)
* Kernel launching order: {k4, k3, k2, k1}
* Behavious: (average block number on one SM) At t = 0, k4, k3, and k2 start running simultaneously. k1 cannot start until k3 finishes execution at t = 12. </br>
![example2_opt1](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example2_opt1.png?raw=true)

#### Scheduling Policy 2
* Kernel launching order: {k1, k2, k4, k3}
* Behavious: (average block number on one SM)
At the beginning, k1, k2 and k4 can be launched, and k3 will wait. After k2 finishes execution at t = 16, k3 can start running. </br>
![example2_opt2](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example2_opt.png?raw=true)

|         | Execution time (ms) | Average response time (ms) | Average waiting time (ms) |
|---------|---------------------|----------------------------|---------------------------|
| Naive   | 30                  | (20+16+28+30)/4=23.5       | (0+0+16+20)/4=9           |
| SJF     | 32                  | (10+12+16+32)/4=18.5       | (0+0+0+16)/4=4            |
| MET     | 28                  | (20+16+10+28)/4=18.5       | (0+0+0+16)/4=4            |

### Scenario 3: blocked by kernels with smaller block duration (example3.json)
(A counterexample on "Bigger block size first")

|          | Grid size | Block size | Duration(ms) |
|----------|-----------|------------|--------------|
| kernel 1 | 15        | 1024       | 20           |
| kernel 2 | 15        | 1024       | 30           |
| kernel 3 | 15        | 512        | 50           |

#### Naive Scheduling Policy (FIFO)
* Kernel launching order: {k1, k2, k3}
* Behavious: (average block number on one SM)
At t = 0, k1 and k2 will be launched. At t = 20, k1 finishes execution and k3 will start.
![example3_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example3_naive.png?raw=true)

#### Scheduling Policy 1 (SJF)
* Kernel launching order: {k1, k2, k3}
* Behavious: (average block number on one SM) The same as naive policy.

#### Scheduling Policy 2 (MET)
* Kernel launching order: {k3, k2, k1}
* Behavious: (average block number on one SM)
At t = 0, k3 and k2 will be launched. At t = 30, k2 finishes execution and k1 will start. </br>
![example3_opt](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example3_opt.png?raw=true)

|         | Execution time (ms) | Average response time (ms) | Average waiting time (ms) |
|---------|---------------------|----------------------------|---------------------------|
| Naive   | 70                  | (20+30+70)/3=40            | (0+0+20)/3=6.67           |
| SJF     | 70                  | (20+30+70)/3=40            | (0+0+20)/3=6.67           |
| MET     | 50                  | (50+30+50)/3=43.33         | (0+0+30)/3=10             |

### Scenario 4: random kernels
(A counterexample on "shortest job first")

|           | Grid size | Block size | Duration (ms) |
|-----------|-----------|------------|---------------|
| kernel 1  | 48        | 512        | 20            |
| kernel 2  | 3         | 256        | 50            |
| kernel 3  | 17        | 256        | 10            |
| kernel 4  | 15        | 1024       | 35            |
| kernel 5  | 23        | 128        | 15            |
| kernel 6  | 10        | 512        | 40            |
| kernel 7  | 1         | 1024       | 80            |
| kernel 8  | 30        | 256        | 20            |
| kernel 9  | 12        | 128        | 25            |
| kernel 10 | 47        | 512        | 40            |

#### Naive Scheduling Policy (FIFO)
* Kernel launching order: {k1, k2, k3, k4, k5, k6, k7, k8, k9, k10}
* Behavious: (average block number ) </br>
![example4_naive](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example4_naive.png?raw=true)

#### Scheduling Policy 1 (SJF)
* Kernel launching order: {k3, k5, k1, k8, k9, k4, k6, k10, k2, k7}
* Behavious: (average block number ) </br>
![example4_opt1](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example4_opt1.png?raw=true)

#### Scheduling Policy 2 (MET)
* Kernel launching order: {k7, k2, k6, k10, k4, k9, k1, k8, k5, k3}
* Behavious: (average block number ) </br>
![example4_opt2](https://github.com/yidiwang21/ee217/blob/master/final_project/figs/example4_opt2.png?raw=true)

|       | Execution time (ms) | Average response time (ms)                | Average waiting time (ms)            |
|-------|---------------------|-------------------------------------------|--------------------------------------|
| Naive | 115                 | (20+50+10+55+35+60+115+55+75+100)/10=57.5 | (0+0+0+20+20+20+35+35+35+55)/10=22   |
| SJF   | 155                 | (10+15+35+40+45+70+75+95+120+155)/10=66   | (0+0+0+15+20+20+35+35+55+75)/10=25.5 |
| MET   | 95                  | (80+50+40+80+75+65+95+95+90+90)/10=76     | (0+0+0+0+40+40+40+75+75+75)/10=34.5  |


## References
* <https://github.com/yalue/cuda_scheduling_examiner_mirror>
* [Binary Tree Bin Packing Algorithm in C](https://stackoverflow.com/questions/53536607/binary-tree-bin-packing-algorithm-in-c)
