#ifndef SAMPLE_COMPUTE_HPP
#define SAMPLE_COMPUTE_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <pthread.h>

#include <nlohmann/json.hpp>
#include <cpp-base64/base64.h>
#include <fstream>
#include <iostream>
#include <mutex>

using json = nlohmann::json;

#define tickTime 1
#define SAMPLE_SET_COUNT 1
#define SAMPLE_MAX_TIME_SECONDS 15
#define SAMPLE_BUFFER_SIZE_SAMPLES 16777216
#define MIDI_COUNT 128
#define LOOP 0
#define FILTER_STEPS 512
// cdef.h

class SampleCompute
{
public:
    SampleCompute(int polyphony, int samplesPerDispatch, int lfoCount, int envLenPerPatch, int outchannels, float bendDepth, float outSampleRate, int threadCount);
    ~SampleCompute();
    void Dump(const char *filename);
    void HardStop();
    void Test();

    // Parameters of the engine
    int threadCount = 2;
    int polyphony;
    int outSampleRate = 44100; // Default sample rate
    int outchannels;
    int envLenPerPatch;
    bool loop = false;
    float masterVolume;
    bool rhodesEffectOn = false;
    int lfoCount;
    unsigned int framesPerDispatch;

    // One Mutex for all samples
    std::mutex samplesMutex;

    const double PI = 3.14159265358979323846;

    std::mutex threadVoiceLocks[128]; // Fixed array of locks, one per voice

    int strikeIndex = 0;

    // Global variable to track current tuning system
    int currentTuningSystem = 0;

    // Internal state of the engine
    std::vector<float> lfoPhase;
    std::vector<float> lfoIncreasePerDispatch;
    std::vector<float> dispatchFrameNo;
    std::vector<float> dispatchPhaseClipped;
    std::vector<std::vector<float>> outputPhaseFloor;
    std::vector<std::vector<float>> samplesNextWeighted;
    std::vector<std::vector<std::vector<float>>> samples;
    std::vector<std::vector<float>> fadeOut;
    std::vector<float> rhodesEffect;
    std::vector<float> xfadeTracknot;
    std::vector<float> xfadeTrack;
    std::vector<SampleData *> voiceSamplePtr;
    std::vector<float> slaveFade;
    std::vector<float> voiceDetune;
    std::vector<float> noLoopFade;
    std::vector<std::vector<float>> accumulation;
    std::vector<std::vector<float>> sampleWithinDispatchPostBend;
    std::vector<float> pitchWheel;
    std::vector<float> portamento;
    std::vector<float> portamentoAlpha;
    std::vector<float> portamentoTarget;
    std::vector<float> releaseVol;
    std::vector<float> combinedEnvelope;
    std::vector<float> velocityVol;
    std::vector<float> indexInEnvelope;
    std::vector<int> envelopeEnd;
    std::vector<float> currEnvelopeVol;
    std::vector<float> nextEnvelopeVol;

    // Thread pool management
    void InitThreadPool(int numThreads);
    void DestroyThreadPool();

    // Internal "private" functions
    void ProcessVoices(int threadNo, int numThreads, float *outputBuffer);
    void SumSamples(int threadNo, int numThreads, float *outputBuffer);
    void RunMultithread(int numThreads, float *outputBuffer, void *(*threadFunc)(void *));
};

double midiNoteTo12TETFreq(int note);


/*
// 5.1 surround angles (excluding LFE)
std::vector<float> channelAngles5_1 = {
    -30.0f * M_PI / 180.0f,  // Front Left
    0.0f * M_PI / 180.0f,   // Center
    30.0f * M_PI / 180.0f,  // Front Right
    -110.0f * M_PI / 180.0f, // Surround Left
    110.0f * M_PI / 180.0f  // Surround Right
};
*/

#endif // SAMPLE_COMPUTE_HPP
