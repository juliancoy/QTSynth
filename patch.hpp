#ifndef PATCH_HPP
#define PATCH_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <pthread.h>

#include "sample_compute.hpp"
#include <nlohmann/json.hpp>
#include <cpp-base64/base64.h>
#include <fstream>
#include <iostream>
#include <mutex>


using json = nlohmann::json;

// Cache-friendly sample storage
struct alignas(64) SampleData
{                                                 // Align to 64-byte cache line
    float *samples;                               // Contiguous sample data
    size_t numFrames;                             // Number of frames
    size_t numChannels;                           // Number of channels
    float baseFrequency;                          // Sample frequency
    float sampleRate;                             // Sample rate
    int loopStart;                                // Loop start frame
    int loopLength;                               // Loop length in frames
    int loopEnd;                                  // Loop end frame
    float strikeVolume;                           // Strike volume
    float *envelope;                              // Envelope data
    std::vector<std::vector<float>> volumeMatrix; // Channel volume matrix per sample
};

class Patch
{
    public:
    Patch(const std::string &filename, SampleCompute * compute);
    ~Patch();

    // Envelopes
    int envLen = 512;
    std::vector<float> strikeEnvelope;
    std::vector<float> releaseEnvelope; 

    // Global variable to track current tuning system
    int currentTuningSystem = 0;

    SampleCompute * compute;
    // Tuning system mappings
    std::vector<std::vector<std::vector<int>>> key2sampleIndexAll; // Tuning system, key, samples
    std::vector<std::vector<std::vector<float>>> key2sampleDetuneAll;
    std::vector<std::vector<int>> key2voiceIndex;
    std::vector<std::vector<int>> *key2sampleIndex;
    std::vector<std::vector<float>> *key2sampleDetune;

    bool sustainPedalOn = false;
    float modulationDepth = 0;
    float expression = 0;
    float bendDepth = 2;
    json key2samples12tet;

    std::vector<SampleData> samplesData; // All samples in contiguous blocks

    // Public API
    int LoadRestAudioB64(const json &sample);
    void ProcessMidi(std::vector<unsigned char> *message);
    void SetTuningSystem(int tuningSystem);
    void DeleteSample(int sampleNo);    
    int AppendSample(std::vector<float> samples, float sampleRate, int inchannels, float baseFrequency);
    void DumpSampleInfo(const char *filename);

    void SetPitchBend(float bend, int index);
    void UpdateDetune(float detune, int index);

    // functions that adjust SampleCompute
    int Strike(SampleData *sample, float velocity, float sampleDetune);
    int Release(int midi_key, float velocity);
    void ReleaseAll();
    void GenerateKeymap(double (*tuningSystemFn)(int));

};

void WriteVectorToWav(std::vector<float> outvector, const std::string &filename, int channels, float sampleRate);

#endif 
