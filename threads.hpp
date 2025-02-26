#pragma once

#include "sample_compute.hpp"
#include "patch.hpp"

// Extended ThreadData structure to include outputBuffer
// Thread pool data structures
typedef struct ThreadData
{
    SampleCompute *sampleCompute;
    int threadCount;
    int threadNo;
    float *outputBuffer;
    bool shouldExit;
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    bool hasWork;
    void *(*workFunction)(void *);
} ThreadData;

// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency

void *ProcessVoicesThreadWrapper(void *threadArg);
void *SumSamplesThreadWrapper(void *threadArg);
void *ThreadWorker(void *arg);

