
/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/
#ifndef __SUPPORT_CU__
#define __SUPPORT_CU__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "support.h"

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

__device__ inline unsigned int kernelTimer(void) {
    clock_t k_time = clock();
    return k_time;
}

__device__ inline uint64_t GlobalTimer64(void) {
    volatile uint64_t reading;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(reading));
    return reading;
}

static __device__ __inline__ unsigned int GetSMID(void) {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

#endif