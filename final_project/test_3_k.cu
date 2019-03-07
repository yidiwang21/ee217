#include <stdio.h>
#include <stdlib.h>
#include "support.cu"
#include "kernel.cu"

#define STREAM_NUM  3

#define BLOCK_SIZE_1  512
#define GRID_SIZE_1   15 * 3
#define BLOCK_SIZE_2  256
#define GRID_SIZE_2   15 * 5
#define BLOCK_SIZE_3  1024
#define GRID_SIZE_3   15

#define DELAY_BETWEEN_LAUNCH    0 //0.005
#define SPIN_TIME   20000000         // 20m

int main() {

}