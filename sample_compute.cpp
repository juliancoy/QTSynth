// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#include "sample_compute.hpp"
#include <rtaudio/RtAudio.h>
#include <algorithm>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>

const double PI = 3.14159265358979323846;
const int SAMPLE_RATE = 44100;
// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData);

// ThreadData threadData[numThreads];
SampleCompute self;

// Extensable arrays for sample state
std::vector<float> sampleStartPhase;
std::vector<int> sampleLength;
std::vector<int> sampleEnd;
std::vector<int> sampleLoopStart;
std::vector<int> sampleLoopLength;
std::vector<int> sampleLoopEnd;
std::vector<float> voiceStrikeVolume;
std::vector<std::vector<float>> patchEnvelope; // Nested vector for envelopes
std::vector<std::vector<std::vector<float>>> sampleChannelVol;
std::vector<float> sampleChannelCount;

int strikeIndex = 0;

void *threadFunction(void *threadArg)
{
    ThreadData *data = (ThreadData *)threadArg;

    float outputBuffer[self.samplesPerDispatch];
    Run(data->threadNo, data->threadCount, outputBuffer);
    pthread_exit(NULL);
}

#include <fstream>
#include <vector>
#include <mutex>
#include "dr_wav.h"

void WriteVectorToWav(std::vector<float> outvector, const std::string &filename, int channels)
{
    if (outvector.empty())
    {
        std::cerr << "Error: binaryBlob is empty, nothing to write." << std::endl;
        return;
    }

    // WAV file format setup
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT; // 32-bit float PCM
    format.channels = channels;                // Channel Count
    format.sampleRate = SAMPLE_RATE;           // Sample rate
    format.bitsPerSample = 32;                 // 32-bit float

    // Initialize WAV file for writing
    drwav wav;
    if (!drwav_init_file_write(&wav, filename.c_str(), &format, nullptr))
    {
        std::cerr << "Failed to initialize WAV file writing: " << filename << std::endl;
        return;
    }

    // Write binaryBlob as PCM frames
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, outvector.size()/channels, outvector.data());
    drwav_uninit(&wav);

    if (framesWritten == 0)
    {
        std::cerr << "Failed to write audio data to " << filename << std::endl;
    }
    else
    {
        std::cout << "Successfully wrote " << framesWritten << " frames to " << filename << std::endl;
    }
}

#include <vector>
#include <cmath>

void RunMultithread(int numThreads = 1)
{
    int rc;
    long t;
    int voicesPerThread = self.polyphony / numThreads;
    pthread_t threads[numThreads];
    ThreadData threadData[numThreads];

    for (t = 0; t < numThreads; t++)
    {
        threadData[t].sampleCompute = &self;
        threadData[t].threadNo = t;
        threadData[t].threadCount = numThreads;

        rc = pthread_create(&threads[t], NULL, threadFunction, (void *)&threadData[t]);
        if (rc)
        {
            std::cout << "ERROR; return code from pthread_create() is " << rc << std::endl;
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (t = 0; t < numThreads; t++)
    {
        pthread_join(threads[t], NULL);
    }
}

RtAudio dac(RtAudio::LINUX_PULSE);

void DeInitAudio()
{
    if (dac.isStreamOpen())
    {
        try
        {
            if (dac.isStreamRunning())
            {
                std::cout << "Stopping audio stream" << std::endl;
                dac.stopStream();
            }

            std::cout << "Closing audio stream" << std::endl;
            dac.closeStream();
        }
        catch (RtAudioErrorType &e)
        {
            std::cerr << "Error while closing audio: " << e << std::endl;
        }
    }
}

void InitAudio(int buffercount)
{
    // Set up RTMIDI
    unsigned int devices = dac.getDeviceCount();
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
            info = dac.getDeviceInfo(i);
            std::cout << "Device " << i << ": " << info.name << std::endl;
        }
        catch (RtAudioErrorType &error)
        {
            std::cerr << error << std::endl;
        }
    }

    // Set output parameters
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac.getDefaultOutputDevice();
    parameters.nChannels = self.outchannels;
    parameters.firstChannel = 0;

    std::cout << "Opening output stream" << std::endl;
    // Open the stream with minimal buffering for low latency
    try
    {
        RtAudio::StreamOptions options;
        options.numberOfBuffers = buffercount; // Minimum number of buffers for stable playback
        // options.flags = RTAUDIO_MINIMIZE_LATENCY; // Request minimum latency

        dac.openStream(&parameters, nullptr, RTAUDIO_FLOAT32,
                       SAMPLE_RATE, &self.samplesPerDispatch, &audioCallback,
                       nullptr, &options);
        dac.startStream();
    }
    catch (RtAudioErrorType &e)
    {
        std::cerr << "Error: " << e << std::endl;
        return;
    }
}

#define MIDI_KEY_COUNT 128

void Init(int polyphony, int samplesPerDispatch, int lfoCount, int envLenPerPatch, int outchannels)
{
    self.outchannels = outchannels;
    self.samplesPerDispatch = samplesPerDispatch;
    self.envLenPerPatch = envLenPerPatch;
    self.polyphony = polyphony;
    self.rhodesEffectOn = false;
    self.loop = 0;
    self.OVERVOLUME = 1.0 / (1 << 3);
    self.binaryBlob.clear(); // Initialize empty vector

    self.polyphony = polyphony;

    self.rhodesEffect.resize(self.outchannels, 0.0f);

    self.lfoCount = lfoCount;
    self.lfoPhase.resize(lfoCount, 0.0f);
    self.lfoIncreasePerDispatch.resize(lfoCount, 0.0f);

    self.dispatchPhase.resize(polyphony, 0.0f);
    self.dispatchPhaseClipped.resize(polyphony, 0.0f);

    self.outputPhaseFloor.resize(polyphony, std::vector<float>(self.samplesPerDispatch, 0.0f));
    self.samples.resize(2, std::vector<std::vector<float>>(polyphony, std::vector<float>(self.samplesPerDispatch, 0.0f)));
    self.fadeOut.resize(polyphony, std::vector<float>(self.samplesPerDispatch, 0.0f));
    self.accumulation.resize(polyphony, std::vector<float>(self.samplesPerDispatch, 0.0f));
    self.sampleWithinDispatchPostBend.resize(polyphony, std::vector<float>(self.samplesPerDispatch, 0.0f));

    self.envelopeEnd.resize(polyphony, 0.0f);
    for (int voiceNo = 0; voiceNo < self.polyphony; voiceNo++)
    {
        self.envelopeEnd[voiceNo] = (voiceNo + 1) * self.envLenPerPatch - 1;
    }

    self.key2sampleIndex.resize(MIDI_KEY_COUNT);
    self.key2sampleDetune.resize(MIDI_KEY_COUNT);
    self.key2voiceIndex.resize(MIDI_KEY_COUNT);

    self.xfadeTracknot.resize(polyphony, 1.0f);
    self.xfadeTrack.resize(polyphony, 0.0f);

    self.voiceLoopStart.resize(polyphony, 0.0f);
    self.voiceLoopEnd.resize(polyphony, 0.0f);
    self.voiceLoopLength.resize(polyphony, 0.0f);
    self.slaveFade.resize(polyphony, 0.0f);
    self.voiceLen.resize(polyphony, 0.0f);
    self.voiceEnd.resize(polyphony, 0.0f);
    self.voiceDetune.resize(polyphony, 0.0f);
    self.voiceChannelCount.resize(polyphony, 0.0f);
    self.voiceChannelVol.resize(polyphony);
    self.noLoopFade.resize(polyphony, 0.0f);

    self.pitchBend.resize(polyphony, 0.0f);
    self.portamento.resize(polyphony, 1.0f);
    self.portamentoAlpha.resize(polyphony, 1.0f);
    self.portamentoTarget.resize(polyphony, 1.0f);

    self.releaseVol.resize(polyphony, 0.0f);
    self.combinedEnvelope.resize(polyphony * envLenPerPatch, 0.0f);
    self.velocityVol.resize(polyphony, 0.0f);
    self.indexInEnvelope.resize(polyphony, 0.0f);

    self.currEnvelopeVol.resize(polyphony, 0.0f);
    self.nextEnvelopeVol.resize(polyphony, 0.0f);
}

using json = nlohmann::json;

int currSampleIndex = 0;
int currPatchSampleNo = 0;

// Function to set the pitch bend for a specific voice
void SetPitchBend(float bend, int index)
{
    if (index >= 0 && index < self.polyphony)
    {
        self.pitchBend[index] = bend;
    }
    else
    {
        std::cout << "Index out of bounds in SetPitchBend" << std::endl;
    }
}

// Function to update the detune value for a specific voice
void UpdateDetune(float detune, int index)
{
    if (index >= 0 && index < self.polyphony)
    {
        self.voiceDetune[index] = detune;
    }
    else
    {
        std::cout << "Index out of bounds in UpdateDetune" << std::endl;
    }
}

// Function to get the envelope length per patch
int GetEnvLenPerPatch()
{
    return self.envLenPerPatch;
}

// Function to append data to the binaryBlob array
int AppendSample(std::vector<float> sample_array, int channels)
{
    // Transform patchSampleNo to sampleIndex
    for (const auto &keyAction : self.key2samples)
    {
        for (const auto &entry : keyAction) // Iterate over all elements in the mapping
        {
            int keyTrigger = entry["keyTrigger"];
            int patchSampleNo = entry["sampleNo"];
            float pitchBend = entry["pitchBend"];

            if (patchSampleNo == currPatchSampleNo)
            {
                self.key2sampleIndex[keyTrigger].push_back(currSampleIndex);
                self.key2sampleDetune[keyTrigger].push_back(pitchBend);
            }
        }
    }

    // Get the current size as the starting point for this sample
    std::lock_guard<std::mutex> lock(self.blobMutex);
    // Store sample start time before inserting data
    int sample_start = self.binaryBlob.size();

    // Append the new samples to the vector
    self.binaryBlob.insert(self.binaryBlob.end(), sample_array.begin(), sample_array.end());

    // Find the maximum absolute sample
    auto max_it = std::max_element(sample_array.begin(), sample_array.end(),
                                   [](float a, float b)
                                   { return std::abs(a) < std::abs(b); });

    float maxSample = (max_it != sample_array.end()) ? std::abs(*max_it) : 0.0f;

    // Increment the start time to ensure no silence at the beginning
    int i = 0;
    while (i < sample_array.size() && std::abs(sample_array[i++]) < maxSample / 100.0f)
        ;
    i -= i % 2;
    std::cout << "First non-silent sample index: " << i << std::endl;
    sample_start += i;

    // If binaryBlob is empty, write this sample to a WAV file using dr_wav
    if (!sample_start)
    {
        WriteVectorToWav(self.binaryBlob, "binaryBlob.wav", channels);
        WriteVectorToWav(sample_array, "sample.wav", channels);
    }

    // Store the Sample Details
    sampleStartPhase.push_back(sample_start);
    sampleLength.push_back(sample_array.size());
    sampleEnd.push_back(sample_start + sample_array.size());

    // TODO: Implement loop
    sampleLoopStart.push_back(sample_start);
    sampleLoopLength.push_back(sample_array.size());
    sampleLoopEnd.push_back(sample_start + sample_array.size());

    currSampleIndex++;
    return 0;
}

// Function to delete a portion of memory from binaryBlob
void DeleteMem(int startAddr, int endAddr)
{
    if (startAddr >= self.binaryBlob.size() || endAddr > self.binaryBlob.size() || startAddr > endAddr)
    {
        std::cout << "Invalid start or end address in DeleteMem" << std::endl;
        return;
    }

    // Erase the range from the vector
    self.binaryBlob.erase(self.binaryBlob.begin() + startAddr, self.binaryBlob.begin() + endAddr);
}

float thisSample, nextSample;

void Run(int threadNo, int numThreads, float *outputBuffer)
{
    float fadelen = 50000.0f;

    // Update LFO phases
    int lfoStart = threadNo * self.lfoCount / numThreads;
    int lfoEnd = lfoStart + self.lfoCount / numThreads;
    for (int lfoNo = lfoStart; lfoNo < lfoEnd; lfoNo++)
    {
        self.lfoPhase[lfoNo] += self.lfoIncreasePerDispatch[lfoNo];
        // Ensure LFO phase wraps around properly
        while (self.lfoPhase[lfoNo] >= 1.0f)
        {
            self.lfoPhase[lfoNo] -= 1.0f;
        }
    }

    // if rhodes effect is on, sweep the sound from left to right
    float volRight;
    float depth = 0.4f;
    for (int channel = 0; channel < self.outchannels; channel++)
    {
        if (self.rhodesEffectOn)
            self.rhodesEffect[channel] = ((1 - depth) + sinf(2 * M_PI * fmodf(self.lfoPhase[0] + 1.0 / self.outchannels, 1.0f)) * depth);
        else
            self.rhodesEffect[channel] = 1;
    }

    int voiceStart = threadNo * self.polyphony / numThreads;
    int voiceEnd = voiceStart + self.polyphony / numThreads;
    // Process each voice
    for (int voiceNo = voiceStart; voiceNo < voiceEnd; voiceNo++)
    {

        // std::cout << " VoiceNo " << voiceNo << " Voice Detune " << self.voiceDetune[voiceNo] << std::endl;

        float thisEnvelopeVol = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceNo])] * self.releaseVol[voiceNo] * self.velocityVol[voiceNo];

        if (self.indexInEnvelope[voiceNo] < self.envelopeEnd[voiceNo])
        {
            self.indexInEnvelope[voiceNo]++;
        }

        self.nextEnvelopeVol[voiceNo] = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceNo])] * self.releaseVol[voiceNo] * self.velocityVol[voiceNo];

        float difference = self.nextEnvelopeVol[voiceNo] - thisEnvelopeVol;

        // Process each sample within the dispatch
        for (int sampleNo = 0; sampleNo < self.samplesPerDispatch; sampleNo++)
        {
            /*std::cout << "   Sample " << sampleNo << std::endl;*/

            // Update portamento for each voice, and for each sample
            self.portamento[voiceNo] = self.portamentoTarget[voiceNo] * self.portamentoAlpha[voiceNo] + (1.0f - self.portamentoAlpha[voiceNo]) * self.portamento[voiceNo];
            // Update the dispatch phase for the next cycle
            self.dispatchPhase[voiceNo] += self.voiceDetune[voiceNo] * self.portamento[voiceNo];

            float normalizedPosition = (float)sampleNo / (float)self.samplesPerDispatch;
            float multiplier = difference * normalizedPosition + thisEnvelopeVol;

            // Clip the phase to valid sample indices and loop if necessary
            if (self.dispatchPhase[voiceNo] * self.voiceChannelCount[voiceNo] >= self.voiceEnd[voiceNo])
            {
                if (self.loop)
                {
                    self.dispatchPhase[voiceNo] = fmodf(self.dispatchPhase[voiceNo] - self.voiceLoopStart[voiceNo], self.voiceLoopLength[voiceNo]) + self.voiceLoopStart[voiceNo];
                }
                else
                {
                    self.dispatchPhase[voiceNo] = self.voiceEnd[voiceNo] - 1;
                }
            }

            std::lock_guard<std::mutex> lock(self.blobMutex);
            for (int inchannel = 0; inchannel < self.voiceChannelCount[voiceNo]; inchannel++)
            {
                for (int outchannel = 0; outchannel < self.outchannels; outchannel++)
                {

                    // Calculate floor and ceiling indices for interpolation
                    int floorIndex = (int)floorf(self.dispatchPhase[voiceNo]) * self.voiceChannelCount[voiceNo] + inchannel;
                    int ceilIndex = floorIndex + self.voiceChannelCount[voiceNo];
                    if (ceilIndex >= self.voiceEnd[voiceNo])
                    {
                        ceilIndex = self.loop ? self.voiceLoopStart[voiceNo] : self.voiceEnd[voiceNo] - 1;
                    }
                    // Ensure indices are within bounds
                    floorIndex = floorIndex < self.binaryBlob.size() ? floorIndex : self.binaryBlob.size() - 1;
                    ceilIndex = ceilIndex < self.binaryBlob.size() ? ceilIndex : self.binaryBlob.size() - 1;

                    // Perform linear interpolation between the two samples
                    float fraction = self.dispatchPhase[voiceNo] - floorIndex;
                    floorIndex = std::min(floorIndex, static_cast<int>(self.binaryBlob.size() - 1));
                    ceilIndex = std::min(ceilIndex, static_cast<int>(self.binaryBlob.size() - 1));
                    thisSample = self.binaryBlob[floorIndex];
                    nextSample = self.binaryBlob[ceilIndex];

                    self.samples[outchannel][voiceNo][sampleNo] =
                        (thisSample * (1.0f - fraction) + nextSample * fraction) * // Interpolation
                        self.OVERVOLUME *                                          // Overall volume scaling
                        self.voiceChannelVol[voiceNo][inchannel][outchannel] *     // Voice-channel volume
                        self.rhodesEffect[outchannel];                             // Rhodes effect per output channel
                }
            }
            // Apply fade out if needed
            if (0)
            {
                // Fade out around the loop points
                self.fadeOut[voiceNo][sampleNo] = fminf(fabsf(self.outputPhaseFloor[voiceNo][sampleNo] - (self.voiceLoopEnd[voiceNo] - self.voiceLoopLength[voiceNo] + self.slaveFade[voiceNo])) / fadelen, 1.0f);
                self.fadeOut[voiceNo][sampleNo] = fminf(fabsf(self.outputPhaseFloor[voiceNo][sampleNo] - (self.voiceLoopEnd[voiceNo] + self.slaveFade[voiceNo])) / fadelen, self.fadeOut[voiceNo][sampleNo]);

                // Applying fade out logic based on slaveFade
                // Assuming slaveFade is an index, not an array, as it is not clear from the Python code
                self.fadeOut[voiceNo][sampleNo] = self.xfadeTracknot[voiceNo] * self.fadeOut[voiceNo][sampleNo] + self.xfadeTrack[voiceNo] * (1.0f - self.fadeOut[voiceNo][sampleNo]);

                // Ensure fadeOut is not less than noLoopFade
                self.fadeOut[voiceNo][sampleNo] = fmaxf(self.fadeOut[voiceNo][sampleNo], self.noLoopFade[voiceNo]);

                // Apply fadeOut to samples
                self.samples[0][voiceNo][sampleNo] *= self.fadeOut[voiceNo][sampleNo];
            }
        }
        if (self.dispatchPhase[voiceNo] >= self.voiceEnd[voiceNo] && self.loop)
        {
            self.dispatchPhase[voiceNo] = fmodf(self.dispatchPhase[voiceNo] - self.voiceLoopStart[voiceNo], self.voiceLoopLength[voiceNo]) + self.voiceLoopStart[voiceNo];
        }
    }

    int sampleStart = threadNo * self.samplesPerDispatch / numThreads;
    int sampleEnd = sampleStart + self.samplesPerDispatch / numThreads;

    // Sum samples across polyphony and interleave channels
    for (int sampleNo = sampleStart; sampleNo < sampleEnd; sampleNo++)
    {
        for (int voiceNo = 0; voiceNo < self.polyphony; voiceNo++)
        {
            for (int channel = 0; channel < self.outchannels; channel++)
            {
                float thisSample = self.samples[channel][voiceNo][sampleNo];
                outputBuffer[sampleNo * 2 + channel] += thisSample;
            }
        }
    }
}

int Strike(int sampleNo, float velocity, float voiceDetune, float *patchEnvelope)
{
    std::cout << "Striking sample " << sampleNo << " at strike index " << strikeIndex << std::endl;
    self.xfadeTrack[strikeIndex] = 0;
    self.xfadeTracknot[strikeIndex] = 1;
    self.dispatchPhase[strikeIndex] = sampleStartPhase[sampleNo];
    self.slaveFade[strikeIndex] = strikeIndex;
    self.noLoopFade[strikeIndex] = 1;

    // Transfer Sample params to Voice params for sequential access in Run
    self.voiceLen[strikeIndex] = sampleLength[sampleNo];
    self.voiceEnd[strikeIndex] = sampleEnd[sampleNo];
    self.voiceLoopLength[strikeIndex] = sampleLoopLength[sampleNo];
    self.voiceLoopStart[strikeIndex] = sampleLoopStart[sampleNo];
    self.voiceLoopEnd[strikeIndex] = sampleLoopEnd[sampleNo];

    // If no patch envelope is supplied, all 1
    if (patchEnvelope == nullptr)
    {
        for (int envIndex = 0; envIndex < self.envLenPerPatch; envIndex++)
        {
            int envelopeIndex = strikeIndex * self.envLenPerPatch + envIndex;
            self.combinedEnvelope[envelopeIndex] = 1;
        }
    }
    // Otherwise, load the patch env
    else
    {
        // Assuming 'envLenPerPatch' is the length of 'patchEnvelope'
        for (int envIndex = 0; envIndex < self.envLenPerPatch; envIndex++)
        {
            int envelopeIndex = strikeIndex * self.envLenPerPatch + envIndex;
            self.combinedEnvelope[envelopeIndex] = patchEnvelope[envIndex];
        }
    }

    // set additional voice init params
    self.releaseVol[strikeIndex] = 1;
    self.velocityVol[strikeIndex] = velocity / 255.0;
    self.indexInEnvelope[strikeIndex] = strikeIndex * self.envLenPerPatch;
    self.voiceDetune[strikeIndex] = voiceDetune;
    std::cout << "Striking voice " << strikeIndex << " with detune " << self.voiceDetune[strikeIndex] << std::endl;

    self.voiceChannelVol[strikeIndex] = sampleChannelVol[sampleNo];
    self.voiceChannelCount[strikeIndex] = sampleChannelCount[sampleNo];

    self.portamento[strikeIndex] = 1;
    self.portamentoAlpha[strikeIndex] = 1;
    self.portamentoTarget[strikeIndex] = 1;

    // implement Round Robin for simplicity
    strikeIndex = (strikeIndex + 1) % self.polyphony;
    return (strikeIndex + self.polyphony - 1) % self.polyphony;
}

void Release(int voiceIndex, float *env)
{
    self.releaseVol[voiceIndex] = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceIndex])];

    // If no patch envelope is supplied, all 1
    if (env == nullptr)
    {
        for (int envIndex = 0; envIndex < self.envLenPerPatch; envIndex++)
        {
            int envelopeIndex = strikeIndex * self.envLenPerPatch + envIndex;
            self.combinedEnvelope[envelopeIndex] = 1;
        }
    }
    // Otherwise, load the patch env
    else
    {
        for (int envPosition = 0; envPosition < self.envLenPerPatch; envPosition++)
        {
            int index = voiceIndex * self.envLenPerPatch + envPosition;
            self.combinedEnvelope[index] = env[envPosition] * self.releaseVol[voiceIndex];
        }
    }
    // Reset the env to the beginning
    self.indexInEnvelope[voiceIndex] = voiceIndex * self.envLenPerPatch;
}

void HardStop(int strikeIndex)
{
    self.indexInEnvelope[strikeIndex] = strikeIndex * self.envLenPerPatch;
    self.releaseVol[strikeIndex] = 1.0f;
    self.velocityVol[strikeIndex] = 0.0f;
}

#define ELEMENTS_TO_PRINT 16
void Dump(const char *filename)
{
    json output;

    // Store scalar values
    output["rhodesEffect"] = self.rhodesEffect;
    output["loop"] = self.loop;
    output["OVERVOLUME"] = self.OVERVOLUME;

    output["lfoPhase"] = self.lfoPhase;
    output["lfoIncreasePerDispatch"] = self.lfoIncreasePerDispatch;
    output["dispatchPhase"] = self.dispatchPhase;
    output["dispatchPhaseClipped"] = self.dispatchPhaseClipped;

    output["xfadeTracknot"] = self.xfadeTracknot;
    output["xfadeTrack"] = self.xfadeTrack;
    /*
    output["loopStart"] = self.loopStart;
    output["loopEnd"] = self.loopEnd;
    output["loopLength"] = self.loopLength;
    output["slaveFade"] = self.slaveFade;
    */
    output["sampleLen"] = self.voiceLen;
    output["sampleEnd"] = self.voiceEnd;
    output["voiceDetune"] = self.voiceDetune;
    output["noLoopFade"] = self.noLoopFade;
    output["pitchBend"] = self.pitchBend;
    output["portamento"] = self.portamento;
    output["portamentoAlpha"] = self.portamentoAlpha;
    output["portamentoTarget"] = self.portamentoTarget;
    output["releaseVol"] = self.releaseVol;
    output["velocityVol"] = self.velocityVol;
    output["indexInEnvelope"] = self.indexInEnvelope;
    output["envelopeEnd"] = self.envelopeEnd;
    output["currEnvelopeVol"] = self.currEnvelopeVol;
    output["nextEnvelopeVol"] = self.nextEnvelopeVol;
    output["combinedEnvelope"] = self.combinedEnvelope;

    /*{
        std::lock_guard<std::mutex> lock(self.blobMutex);
        if (!self.binaryBlob.empty()) {
            output["binaryBlob"] = self.binaryBlob;
        }
    }*/

    // Write to file
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Use dump with 2-space indentation but compact arrays
    file << output.dump(2); // 2-space indentation
    file.close();
}

void DumpSampleInfo(const char *filename)
{
    json output;
    self.key2voiceIndex.resize(MIDI_KEY_COUNT);

    output["key2sampleIndex"] = self.key2sampleIndex;
    output["key2sampleDetune"] = self.key2sampleDetune;
    output["key2voiceIndex"] = self.key2voiceIndex;

    // Write to file
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Use dump with 2-space indentation but compact arrays
    file << output.dump(2); // 2-space indentation
    file.close();
}

void SelfDestruct()
{
    std::cout << "Clean up audio" << std::endl;
    // Clean up audio
    if (dac.isStreamOpen())
    {
        dac.stopStream();
        dac.closeStream();
    }
}

int LoadRestAudioB64(const json &sample)
{
    // Create result structure
    std::vector<float> samples;
    int sampleRate;
    int inchannels;

    // Decode Base64 to binary
    // Preallocate the binary buffer based on base64 string length
    const std::string &base64Data = sample["audioData"].get<std::string>();
    std::string binaryData;
    binaryData.reserve((base64Data.length() * 3) / 4); // Approximate size
    binaryData = base64_decode(base64Data);
    if (sample["audioFormat"] == "wav")
    {
        drwav wav;
        if (drwav_init_memory(&wav, binaryData.data(), binaryData.size(), nullptr))
        {
            samples.resize(wav.totalPCMFrameCount * wav.channels);
            drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, samples.data());
            sampleRate = wav.sampleRate;
            inchannels = wav.channels;
            drwav_uninit(&wav);
        }
    }
    else if (sample["audioFormat"] == "mp3")
    {
        drmp3 mp3;
        if (drmp3_init_memory(&mp3, binaryData.data(), binaryData.size(), nullptr))
        {
            drmp3_uint64 frameCount = drmp3_get_pcm_frame_count(&mp3);
            samples.resize(frameCount * mp3.channels);
            drmp3_read_pcm_frames_f32(&mp3, frameCount, samples.data());
            sampleRate = mp3.sampleRate;
            inchannels = mp3.channels;
            drmp3_uninit(&mp3);
        }
    }
    else
    {
        std::cout << "Unknown format" << std::endl;
        return -1;
    }

    // Number of frames (samples per channel)
    size_t frameCount = samples.size() / inchannels;

    // Determine the mapping of input channels to output channels
    // For example if the input is Mono, it should distribute even power to both sides
    // the same amound of power that the left channel of a stereo signal puts out on one
    // 2D vector to store gains: gains[inchannel][outchannel]
    std::vector<std::vector<float>> gains(inchannels, std::vector<float>(self.outchannels, 0.0f));

    for (size_t inchannel = 0; inchannel < inchannels; inchannel++)
    {
        // Calculate the angle of the current input channel
        float inChannelAngle = inchannels * M_PI / 2 + inchannel * 2.0f * M_PI / inchannels;

        float totalPower = 0.0f;

        for (size_t outchannel = 0; outchannel < self.outchannels; outchannel++)
        {
            // Calculate the angle of the current output channel
            float outChannelAngle = self.outchannels * M_PI / 2 + outchannel * 2.0f * M_PI / self.outchannels;

            // Calculate angular distance between input and output channels
            float angleDiff = fmodf(std::abs(inChannelAngle - outChannelAngle), 2 * M_PI);
            if (angleDiff > M_PI)
            {
                angleDiff = M_PI - angleDiff;
            }
            // Convert angular distance to unity gain.
            float gain = angleDiff / M_PI;
            gain = std::abs(gain);

            // Store the gain for this input-output channel pair
            gains[inchannel][outchannel] = gain;
            totalPower += gain * gain;
        }

        // Normalize for constant power
        float normalizationFactor = 1.0f / std::sqrt(totalPower);
        for (size_t outchannel = 0; outchannel < self.outchannels; outchannel++)
        {
            gains[inchannel][outchannel] *= normalizationFactor;
        }
    }
    // std::cout << gains[0][0] << " " << gains[0][1] << std::endl;
    sampleChannelVol.push_back(gains); // Add a specific array of 3 elements
    sampleChannelCount.push_back(inchannels);

    AppendSample(samples, inchannels);
    currPatchSampleNo++;

    return 0;
}

// Load samples from JSON file
void LoadSoundJSON(const std::string &filename)
{
    std::cout << "Loading " << filename << std::endl;
    // Use memory mapping for large files
    std::ifstream f(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    f.read(buffer.data(), size);

    json data = json::parse(buffer.begin(), buffer.end());

    int patchNoInFile = 0;
    self.key2samples = data[patchNoInFile]["key2samples"];

    // Load samples
    for (const auto &instrument : data)
    {
        for (const auto &sample : instrument["samples"])
        {
            LoadRestAudioB64(sample);
        }
    }

    std::cout << "Finished loading key mappings" << std::endl;
    std::cout << "Binary Blob size " << self.binaryBlob.size() << std::endl;
}

void ProcessMidi(std::vector<unsigned char> *message)
{
    unsigned int status = message->at(0);
    unsigned int midi_key = message->at(1);
    float velocity = int(message->at(2)) / 127.0;
    std::cout << "status " << status << " midi key " << midi_key << " velocity " << velocity << std::endl;

    // Note On
    if ((status & 0xF0) == 0x90 && velocity > 0)
    {
        const auto &sampleIndex = self.key2sampleIndex[midi_key];
        const auto &sampleDetune = self.key2sampleDetune[midi_key];
        std::cout << "Sample Index: " << sampleIndex[0] << ", " << sampleDetune.size() << std::endl;
        for (size_t i = 0; i < sampleIndex.size() && i < sampleDetune.size(); i++)
        {
            std::cout << " sample Detune " << sampleDetune[i] << std::endl;
            self.key2voiceIndex[midi_key].push_back(Strike(sampleIndex[i], velocity, sampleDetune[i], nullptr));
        }
    }

    // Note Off
    else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0))
    {
        int voicesToRelease = self.key2sampleIndex[midi_key].size();

        for (size_t i = 0; i < voicesToRelease; ++i)
        {
            int frontValue = self.key2voiceIndex[midi_key].front();                     // Get the first element
            self.key2voiceIndex[midi_key].erase(self.key2voiceIndex[midi_key].begin()); // Remove the first element
            Release(frontValue, nullptr);                                               // Pass the value to Release()
        }
    }
}

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData)
{
    auto *buffer = static_cast<float *>(outputBuffer);
    Run(0, 1, buffer);
    return 0;
}

void Test()
{
    std::cout << "Test mode activated" << std::endl;

    int samples_per_dispatch = 128;
    Init(10, samples_per_dispatch, 2, 512, 2);
    std::cout << "Loading JSON" << std::endl;
    LoadSoundJSON("Harp.json");
    DumpSampleInfo("sample_info.json"); // Only dump first buffer for debugging

    std::cout << "Loaded patch" << std::endl;
    std::cout << "Processing MIDI" << std::endl;

    unsigned char penta[] = {
        48, 50, 52, 55, 57};
    std::cout << "Running engine" << std::endl;

    // Calculate buffer sizes for 10 seconds of stereo audio
    const int secondsToGenerate = 10;
    const int samplesPerChannel = SAMPLE_RATE * secondsToGenerate;
    const int totalSamples = samplesPerChannel * 2; // Stereo output

    // Allocate buffer with proper size for stereo output
    std::vector<float> buffer(totalSamples);

    // Generate audio in chunks of samples_per_dispatch
    const int numBuffers = samplesPerChannel / samples_per_dispatch;

    for (int i = 0; i < numBuffers; i++)
    {
        // Compute the starting index in the buffer vector
        int startIndex = i * samples_per_dispatch * self.outchannels;
        Run(0, 1, buffer.data() + startIndex);

        // Send a MIDI note every 100 iterations
        if (i % 100 == 0)
        {
            std::vector<unsigned char> message = {0x90, penta[(i / 100) % 5], 127}; // Note on
            ProcessMidi(&message);
        }

        // Dump first buffer for debugging
        if (i == 0)
        {
            Dump("dump.json");
        }
    }
    WriteVectorToWav(buffer, "Outfile.wav", self.outchannels);
}
