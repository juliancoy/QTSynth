
// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency

#include <rtaudio/RtAudio.h>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <mutex>
#include <sched.h>
#include <threads.hpp>

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData);

// Global thread pool variables
static pthread_t *threads;
static ThreadData *threadData;
static int numThreadsInPool;

void InitThreadPool(int numThreads, SampleCompute * compute)
{
    if (threads != nullptr)
    {
        return; // Thread pool already initialized
    }

    numThreadsInPool = numThreads;
    threads = new pthread_t[numThreads];
    threadData = new ThreadData[numThreads];

    for (int t = 0; t < numThreads; t++)
    {
        threadData[t].threadNo = t;
        threadData[t].threadCount = numThreads;
        threadData[t].shouldExit = false;
        threadData[t].hasWork = false;
        threadData[t].compute = compute;
        pthread_mutex_init(&threadData[t].mutex, nullptr);
        pthread_cond_init(&threadData[t].condition, nullptr);

        pthread_create(&threads[t], nullptr, ThreadWorker, &threadData[t]);
    }
}

void DestroyThreadPool()
{
    if (threads == nullptr)
    {
        return;
    }

    // Signal all threads to exit
    for (int t = 0; t < numThreadsInPool; t++)
    {
        pthread_mutex_lock(&threadData[t].mutex);
        threadData[t].shouldExit = true;
        pthread_cond_signal(&threadData[t].condition);
        pthread_mutex_unlock(&threadData[t].mutex);
    }

    // Wait for all threads to finish
    for (int t = 0; t < numThreadsInPool; t++)
    {
        pthread_join(threads[t], nullptr);
        pthread_mutex_destroy(&threadData[t].mutex);
        pthread_cond_destroy(&threadData[t].condition);
    }

    delete[] threads;
    delete[] threadData;
    threads = nullptr;
    threadData = nullptr;
}

void RunMultithread(int numThreads, float *outputBuffer, void *(*threadFunc)(void *), SampleCompute * compute)
{
    // Update thread data and signal work
    for (int t = 0; t < numThreads; t++)
    {
        pthread_mutex_lock(&threadData[t].mutex);
        threadData[t].compute = compute;
        threadData[t].outputBuffer = outputBuffer;
        threadData[t].workFunction = threadFunc;
        threadData[t].hasWork = true;
        pthread_cond_signal(&threadData[t].condition);
        pthread_mutex_unlock(&threadData[t].mutex);
    }

    // Wait for all threads to complete their work
    for (int t = 0; t < numThreads; t++)
    {
        while (true)
        {
            pthread_mutex_lock(&threadData[t].mutex);
            bool workDone = !threadData[t].hasWork;
            pthread_mutex_unlock(&threadData[t].mutex);
            if (workDone)
                break;
            sched_yield(); // Give other threads a chance to run
        }
    }
}
                  

void *ProcessVoicesThreadWrapper(void *threadArg)
{
    ((ThreadData *)threadArg)->compute->ProcessVoices(((ThreadData *)threadArg)->threadNo,
                  ((ThreadData *)threadArg)->threadCount,
                  ((ThreadData *)threadArg)->outputBuffer);
    return nullptr;
}

void *SumSamplesThreadWrapper(void *threadArg)
{
    ((ThreadData *)threadArg)->compute->SumSamples(((ThreadData *)threadArg)->threadNo,
               ((ThreadData *)threadArg)->threadCount,
               ((ThreadData *)threadArg)->outputBuffer);
    return nullptr;
}

void *ThreadWorker(void *arg)
{
    ThreadData *data = (ThreadData *)arg;

    while (true)
    {
        pthread_mutex_lock(&data->mutex);
        while (!data->hasWork && !data->shouldExit)
        {
            pthread_cond_wait(&data->condition, &data->mutex);
        }

        if (data->shouldExit)
        {
            pthread_mutex_unlock(&data->mutex);
            break;
        }

        void *(*workFunction)(void *) = data->workFunction;
        pthread_mutex_unlock(&data->mutex);

        // Execute the work function
        if (workFunction == ProcessVoicesThreadWrapper)
        {
            data->compute->ProcessVoices(data->threadNo, data->threadCount, data->outputBuffer);
        }
        else if (workFunction == SumSamplesThreadWrapper)
        {
            data->compute->SumSamples(data->threadNo, data->threadCount, data->outputBuffer);
        }

        // Mark work as complete
        pthread_mutex_lock(&data->mutex);
        data->hasWork = false;
        pthread_mutex_unlock(&data->mutex);
    }

    return nullptr;
}
static RtAudio *dac = nullptr;

void InitAudio(int buffercount, unsigned int framesPerDispatch, int * outchannels, unsigned int * sampleRate)
{
    // Set up RTMIDI
    if (!dac)
    {
        dac = new RtAudio(RtAudio::LINUX_PULSE);
    }
    unsigned int devices = dac->getDeviceCount();
    if (devices < 1)
    {
        std::cerr << "No audio devices found!" << std::endl;
        return;
    }

    std::cout << "Available audio devices:" << std::endl;
    RtAudio::DeviceInfo info;
    for (unsigned int i = 0; i < devices; i++)
    {
        try
        {
            info = dac->getDeviceInfo(i);
            std::cout << "Device " << i << ": " << info.name << std::endl;
        }
        catch (RtAudioError &error)
        {
            std::cerr << error.getMessage() << std::endl;
        }
    }

    // Set output parameters
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac->getDefaultOutputDevice();
    
    // Get the default output device's info
    RtAudio::DeviceInfo deviceInfo = dac->getDeviceInfo(parameters.deviceId);
    
    // Use the device's native sample rate and channel count
    *outchannels = deviceInfo.outputChannels;
    parameters.nChannels = deviceInfo.outputChannels;
    *sampleRate = deviceInfo.preferredSampleRate; // Use the preferred sample rate

    parameters.firstChannel = 0;

    std::cout << "Opening output stream with " << parameters.nChannels << " channels and " << *sampleRate << " Hz sample rate" << std::endl;

    // Open the stream with minimal buffering for low latency
    try
    {
        RtAudio::StreamOptions options;
        options.flags = RTAUDIO_SCHEDULE_REALTIME;
        options.numberOfBuffers = buffercount; // Minimum number of buffers for stable playback
        // options.flags = RTAUDIO_MINIMIZE_LATENCY; // Request minimum latency

        dac->openStream(&parameters, nullptr, RTAUDIO_FLOAT32,
                        *sampleRate, &framesPerDispatch, &audioCallback,
                        nullptr, &options);
        dac->startStream();
    }
    catch (RtAudioError &e)
    {
        std::cerr << "Error: " << e.getMessage() << std::endl;
        return;
    }
}

void DeInitAudio()
{
    if (dac)
    {
        try
        {
            if (dac->isStreamOpen())
            {
                if (dac->isStreamRunning())
                {
                    std::cout << "Stopping audio stream" << std::endl;
                    dac->stopStream();
                }
                std::cout << "Closing audio stream" << std::endl;
                dac->closeStream();
            }
            delete dac;
            dac = nullptr;
        }
        catch (RtAudioError &e)
        {
            std::cerr << "Error while closing audio: " << e.what() << std::endl;
        }
    }
}
