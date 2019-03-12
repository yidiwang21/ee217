#include <stdio.h>
#include <iostream>
#include <fstream>
#include "unistd.h"
#include "support.h"
#include "multikernel.cu"
#include "kernel.cu"

int main (int argc, char *argv[]) {
    int sp = 2;
    std::string filename = "example1.json";
    if (access( filename.c_str(), F_OK ) == -1) { fprintf(stderr, "# File doesn't exist!\n"); exit(-1); }

    if (argc == 2) filename = argv[1];
    else if (argc == 3) {
        filename = argv[1];
        sp = atoi(argv[2]);
        if (sp != 2 && sp != 1 && sp != 0) fprintf(stderr, "# Scheduling policy must be 0 (naive) or 1 (minimum AWT) or 2 (minimum execution time)\n");
    }
    else if (argc > 3) { fprintf(stderr, "# Usage: ./exe [file] [sched_policy]\n"); exit(0); }

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

    MultiKernel multi_kernel(fc, sp);
    multi_kernel.kernelLauncher();
    
    return 0;
}