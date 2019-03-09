#include <stdio.h>
#include <iostream>
#include <fstream>
#include "unistd.h"
#include "support.h"
#include "multikernel.cu"
#include "kernel.cu"

int main (int argc, char *argv[]) {

    std::string filename = "config.json";

    if (argc == 2) filename = argv[1];
    else if (argc > 2) { fprintf(stderr, "# Usage: ./exe [file]\n"); exit(0); }

    char *fn = (char *)malloc(sizeof(char) * (filename.length() + 1));
    strcpy(fn, filename.c_str());

    FILE *f = fopen(fn, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *fc; 
    fc = (char *)malloc((fsize+1) * sizeof(char));
    fread(fc, 1, fsize, f);
    fclose(f);
    fc[fsize] = 0;

    MultiKernel multi_kernel(fc);
    multi_kernel.kernelLauncher();
    
    return 0;
}