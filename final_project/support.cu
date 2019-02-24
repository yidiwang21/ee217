
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

// FIXME: 
__device__ inline unsigned int kernelTimer(void) {
    // unsigned int clock;
    // asm("mov.u32 %0, %clock;" : "=r"(clock) );
    // return clock;
    clock_t k_time = clock();
    return k_time;
}

#endif