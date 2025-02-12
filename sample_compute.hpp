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


using json = nlohmann::json;

// Structure to hold decoded sample data
struct SampleData
{
    std::vector<float> samples;
    int sampleRate;
    int channels;
};

#define tickTime 1
#define SAMPLE_SET_COUNT 1
#define SAMPLE_MAX_TIME_SECONDS 15
#define SAMPLE_FREQUENCY 44100
#define SAMPLE_BUFFER_SIZE_SAMPLES 16777216
#define SAMPLES_PER_DISPATCH 128
#define ALLOCATED_POLYPHONY 256
#define POLYPHONY_PER_SHADER 64
#define OUTCHANNELS 2
#define NUM_THREADS 8
#define MIDI_COUNT 128
#define LOOP 0
#define LFO_COUNT 16
#define INCHANNELS 1
#define FILTER_STEPS 512
#define ENVLENPERPATCH 512
#define ENVELOPE_LENGTH 512
// cdef.h

typedef struct SampleCompute
{
    int POLYPHONY;
    float panning;
    float lfoPhase[LFO_COUNT];
    float lfoIncreasePerDispatch[LFO_COUNT];

    float dispatchPhase[ALLOCATED_POLYPHONY];
    float dispatchPhaseClipped[ALLOCATED_POLYPHONY];

    float outputPhaseFloor[ALLOCATED_POLYPHONY][SAMPLES_PER_DISPATCH];

    float samplesNextWeighted[ALLOCATED_POLYPHONY][SAMPLES_PER_DISPATCH];
    float samples[ALLOCATED_POLYPHONY][SAMPLES_PER_DISPATCH];
    float mono[SAMPLES_PER_DISPATCH];
    float fadeOut[ALLOCATED_POLYPHONY][SAMPLES_PER_DISPATCH];

    float xfadeTracknot[ALLOCATED_POLYPHONY];
    float xfadeTrack[ALLOCATED_POLYPHONY];

    float loop;
    float loopStart[ALLOCATED_POLYPHONY];
    float loopEnd[ALLOCATED_POLYPHONY];
    float loopLength[ALLOCATED_POLYPHONY];
    float slaveFade[ALLOCATED_POLYPHONY];
    float sampleLen[ALLOCATED_POLYPHONY];
    float sampleEnd[ALLOCATED_POLYPHONY];
    float voiceDetune[ALLOCATED_POLYPHONY];
    float noLoopFade[ALLOCATED_POLYPHONY];

    float accumulation[ALLOCATED_POLYPHONY][SAMPLES_PER_DISPATCH];
    float sampleWithinDispatchPostBend[ALLOCATED_POLYPHONY][SAMPLES_PER_DISPATCH]; // Adjusted to be a 2D array

    float OVERVOLUME;

    float pitchBend[ALLOCATED_POLYPHONY];
    float portamento[ALLOCATED_POLYPHONY];
    float portamentoAlpha[ALLOCATED_POLYPHONY];
    float portamentoTarget[ALLOCATED_POLYPHONY];

    std::vector<float> binaryBlob; // Vector for binary blob data

    float releaseVol[ALLOCATED_POLYPHONY];
    float combinedEnvelope[ALLOCATED_POLYPHONY * ENVLENPERPATCH]; // Size needs to be defined
    float velocityVol[ALLOCATED_POLYPHONY];
    float indexInEnvelope[ALLOCATED_POLYPHONY];
    float envelopeEnd[ALLOCATED_POLYPHONY];

    float currEnvelopeVol[ALLOCATED_POLYPHONY];
    float nextEnvelopeVol[ALLOCATED_POLYPHONY];

} SampleCompute;

typedef struct ThreadData{
    SampleCompute *sampleCompute;
    int threadNo;
} ThreadData;

// Internal "private" functions
void Init(int POLYPHONY);
void InitAudio();
void SetPitchBend(float bend, int index);
void UpdateDetune(float detune, int index);
int GetEnvLenPerPatch();
int AdvanceEnvelope();
void ApplyPanning();
int AppendSample(const float* npArray, int npArraySize);
void DeleteMem(int startAddr, int endAddr);
void Run(int threadNo, float* outputBuffer = nullptr);
int Strike(int sampleNo, float voiceDetune, float *patchEnvelope);
void HardStop(int voiceIndex);
void RunMultithread();
void Dump(const char* filename);
void Release(int voiceIndex, float *env);

// Public API
int LoadRestAudioB64(const json &sample);
void ProcessMidi(std::vector<unsigned char> *message);
void LoadSoundJSON(const std::string &filename);
void Test();

#endif // SAMPLE_COMPUTE_HPP
