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

typedef struct SampleCompute
{
    int threadCount = 2;
    int polyphony;
    int sampleRate = 44100; // Default sample rate
    bool sustainPedalOn = false;
    float modulationDepth = 0;
    float expression = 0;
    float bendDepth = 2;

    std::vector<float> lfoPhase;
    std::vector<float> lfoIncreasePerDispatch;

    std::vector<float> dispatchFrameNo;
    std::vector<float> dispatchPhaseClipped;

    std::vector<std::vector<float>> outputPhaseFloor;
    std::vector<std::vector<float>> samplesNextWeighted;
    std::vector<std::vector<std::vector<float>>> samples;
    std::vector<std::vector<float>> fadeOut;

    bool rhodesEffectOn = false;
    std::vector<float> rhodesEffect;
    std::vector<float> xfadeTracknot;
    std::vector<float> xfadeTrack;

    int outchannels;
    int envLenPerPatch;
    float loop;
    std::vector<float> voiceLoopStart;
    std::vector<float> voiceLoopEnd;
    std::vector<float> voiceLoopLength;
    std::vector<float> slaveFade;
    std::vector<float> voiceStart;
    std::vector<float> voiceFrameCount;
    std::vector<float> voiceDetune;
    std::vector<float> voiceChannelCount;
    std::vector<std::vector<std::vector<float>>> voiceChannelVol;
    std::vector<float> noLoopFade;

    std::vector<std::vector<float>> accumulation;
    std::vector<std::vector<float>> sampleWithinDispatchPostBend;

    std::vector<std::vector<int>> key2sampleIndex;
    std::vector<std::vector<float>> key2sampleDetune;
    std::vector<std::vector<int>> key2voiceIndex;

    json key2samples; 
    float masterVolume;

    std::vector<float> pitchWheel;
    std::vector<float> portamento;
    std::vector<float> portamentoAlpha;
    std::vector<float> portamentoTarget;

    std::mutex blobMutex;
    std::vector<float> binaryBlob;

    std::vector<float> releaseVol;
    std::vector<float> combinedEnvelope;
    std::vector<float> velocityVol;
    std::vector<float> indexInEnvelope;
    std::vector<int> envelopeEnd;

    std::vector<float> currEnvelopeVol;
    std::vector<float> nextEnvelopeVol;
    int lfoCount;
    unsigned int samplesPerDispatch;
} SampleCompute;


// Extended ThreadData structure to include outputBuffer
typedef struct ThreadData {
    SampleCompute *sampleCompute;
    int threadCount;
    int threadNo;
    float *outputBuffer;
} ThreadData;

// Internal "private" functions
void Init(int polyphony, int samplesPerDispatch, int lfoCount, int envLenPerPatch, int outchannels, float bendDepth, float sampleRate, int threadCount);
void SetVolume(float newVol);
void InitAudio(int bufferCount);
void DeInitAudio();
void SetPitchBend(float bend, int index);
void UpdateDetune(float detune, int index);
int GetEnvLenPerPatch();
int AdvanceEnvelope();
void ApplyPanning();
int AppendSample(std::vector<float> npArray, int channels);
void DeleteMem(int startAddr, int endAddr);
void ProcessVoices(int threadNo, int numThreads, float *outputBuffer);
void SumSamples(int threadNo, int numThreads, float *outputBuffer);
int Strike(int sampleNo, float velocity, float sampleDetune, float *patchEnvelope);
void HardStop(int voiceIndex);
void RunMultithread(int numThreads, float *outputBuffer);
void Dump(const char* filename);
int Release(int voiceIndex, float *env);

// Public API
int LoadRestAudioB64(const json &sample);
void ProcessMidi(std::vector<unsigned char> *message);
void LoadSoundJSON(const std::string &filename);
void Test();


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
