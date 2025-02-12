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
#define POLYPHONY_PER_SHADER 64
#define POLYPHONY 256
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
    float panning;
    float lfoPhase[LFO_COUNT];
    float lfoIncreasePerDispatch[LFO_COUNT];

    float dispatchPhase[POLYPHONY];
    float dispatchPhaseClipped[POLYPHONY];

    float outputPhaseFloor[POLYPHONY][SAMPLES_PER_DISPATCH];

    float samplesNextWeighted[POLYPHONY][SAMPLES_PER_DISPATCH];
    float samples[POLYPHONY][SAMPLES_PER_DISPATCH];
    float mono[SAMPLES_PER_DISPATCH];
    float fadeOut[POLYPHONY][SAMPLES_PER_DISPATCH];

    float xfadeTracknot[POLYPHONY];
    float xfadeTrack[POLYPHONY];

    float loop;
    float loopStart[POLYPHONY];
    float loopEnd[POLYPHONY];
    float loopLength[POLYPHONY];
    float slaveFade[POLYPHONY];
    float sampleLen[POLYPHONY];
    float sampleEnd[POLYPHONY];
    float voiceDetune[POLYPHONY];
    float noLoopFade[POLYPHONY];

    float accumulation[POLYPHONY][SAMPLES_PER_DISPATCH];
    float sampleWithinDispatchPostBend[POLYPHONY][SAMPLES_PER_DISPATCH]; // Adjusted to be a 2D array

    float OVERVOLUME;

    float pitchBend[POLYPHONY];
    float portamento[POLYPHONY];
    float portamentoAlpha[POLYPHONY];
    float portamentoTarget[POLYPHONY];

    float *binaryBlob;    // Dynamically sized array for binary blob data
    float binaryBlobSize; // Total allocated memory size for binaryBlob
    float usedSize;       // Amount of binaryBlob that is currently in use

    float releaseVol[POLYPHONY];
    float combinedEnvelope[POLYPHONY * ENVLENPERPATCH]; // Size needs to be defined
    float velocityVol[POLYPHONY];
    float indexInEnvelope[POLYPHONY];
    float envelopeEnd[POLYPHONY];

    float currEnvelopeVol[POLYPHONY];
    float nextEnvelopeVol[POLYPHONY];

} SampleCompute;

typedef struct ThreadData{
    SampleCompute *sampleCompute;
    int threadNo;
} ThreadData;

// Internal "private" functions
void Init();
void SetPitchBend(float bend, int index);
void UpdateDetune(float detune, int index);
int GetEnvLenPerPatch();
int AdvanceEnvelope();
void ApplyPanning();
int AppendSample(const float* npArray, int npArraySize);
void DeleteMem(int startAddr, int endAddr);
void Run(int threadNo, float* outputBuffer = nullptr);
void Strike(int sampleNo, float voiceDetune, float *patchEnvelope);
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
