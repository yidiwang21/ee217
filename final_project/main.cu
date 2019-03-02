#include <stdio.h>
#include <iostream>
#include <fstream>
#include "kernel.cu"
#include "support.cu"
#include "launcher.cu"

int main (int argc, char *argv[]) {
    // read kernel config file
    FILE *f = fopen("config/config_1.json", "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *fc; 
    fc = (char *)malloc((fsize+1) * sizeof(char));
    fread(fc, 1, fsize, f);
    fclose(f);
    fc[fsize] = 0;

    // init kernel input
    // int *in;
    // int n = 1000 * 1000;
    // in = (int*)malloc(n * sizeof(int));
    // int *output;
    // output = (int*)malloc(n * sizeof(int));

    // for (int i = 0; i < n; i++) {
    //     in[i] = i % 255;
    // }
    // int *in_d;
    // cudaMalloc((void**) &in_d, n * sizeof(int));
    // cudaMemcpy(in_d, in, n * sizeof(int), cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();

    scheduler(fc, NULL, 0);

    // cudaMemcpy(output, in_d, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("# Done!\n");
}