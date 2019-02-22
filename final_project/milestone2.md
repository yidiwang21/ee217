Midterm Progress Report

Final project description:
One of the GPU scheduling rules is that kernels cannot begin execution if there are no sufficient GPU resources for at least one block. Under this rule, suppose that we have multiple kernels that are going to be released in order. There is a large kernel that will run out of GPU resources in the middle of every two small kernels. By default, due to the rule, the following small kernel can not be dequeued as long as a large kernel has not been completed execution. If the queue can be sorted and the kernel release time can be given based on each kernel's property, the small kernels can run concurrently and do not need to reduce total execution time. 
The purpose of this project is to test whether there will be a total execution time improvement if the release time of multiple kernels which are pushed into different streams based on the GPU resources they will use are scheduled. The scheduler will be implemented at the application level.

Planned tasks:
1. Write a benchmark, consisting of several typical kernels with different block sizes, block numbers, thread numbers, durations, etc.
2. Implement a kernel launcher to determine the release time of each kernel, to try to minimize to total execution time.
3. Use semaphores to control kernel launch time.

Expected outcomes:
The scheduling algorithm can reduce the total execution time.

Tools:
API: Cuda 10.0
Testing platform: GeForce GTX 1070