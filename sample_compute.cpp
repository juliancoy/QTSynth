// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#include "sample_compute.hpp"
#include <rtaudio/RtAudio.h>
#include <algorithm>
#include <string>
#include <limits>

#include <fstream>
#include <vector>
#include <mutex>
#include "dr_wav.h"

#include "tuning.hpp"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>

#define MIDI_NOTES 128
#define MIDI_CC_SUSTAIN 64
#define MIDI_CC_VOLUME 7
#define MIDI_CC_MODULATION 1
#define MIDI_CC_EXPRESSION 11

const double PI = 3.14159265358979323846;
// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData);

// Global thread pool variables
static pthread_t *threads = nullptr;
static ThreadData *threadData = nullptr;
static int numThreadsInPool = 0;

SampleCompute self;

// Extensable arrays for each loaded patch
std::vector<int> patchStartSample;
std::vector<int> patchSampleCount;

// Extensable arrays for sample state
std::vector<float> sampleStart;
std::vector<float> sampleFrequency;
std::vector<int> sampleFrameCount;
std::vector<int> sampleLoopStart;
std::vector<int> sampleLoopLength;
std::vector<int> sampleLoopEnd;
std::vector<std::vector<std::vector<float>>> sampleChannelVol;
std::vector<float> sampleChannelCount;
std::vector<float> voiceStrikeVolume;
std::vector<std::vector<float>> patchEnvelope; // Nested vector for envelopes
std::mutex threadVoiceLocks[128];              // Fixed array of locks, one per voice

int strikeIndex = 0;

// Global variable to track current tuning system
static int currentTuningSystem = 0;

std::vector<std::vector<int>> * key2sampleIndex;
std::vector<std::vector<float>> * key2sampleDetune;


void *ProcessVoicesThreadWrapper(void *threadArg)
{
    ProcessVoices(((ThreadData *)threadArg)->threadNo,
                  ((ThreadData *)threadArg)->threadCount,
                  ((ThreadData *)threadArg)->outputBuffer);
    return nullptr;
}

void *SumSamplesThreadWrapper(void *threadArg)
{
    SumSamples(((ThreadData *)threadArg)->threadNo,
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
            ProcessVoices(data->threadNo, data->threadCount, data->outputBuffer);
        }
        else if (workFunction == SumSamplesThreadWrapper)
        {
            SumSamples(data->threadNo, data->threadCount, data->outputBuffer);
        }

        // Mark work as complete
        pthread_mutex_lock(&data->mutex);
        data->hasWork = false;
        pthread_mutex_unlock(&data->mutex);
    }

    return nullptr;
}

void InitThreadPool(int numThreads)
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

void RunMultithread(int numThreads, float *outputBuffer, void *(*threadFunc)(void *))
{
    // Update thread data and signal work
    for (int t = 0; t < numThreads; t++)
    {
        pthread_mutex_lock(&threadData[t].mutex);
        threadData[t].sampleCompute = &self;
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
    format.sampleRate = self.outSampleRate;       // Sample rate
    format.bitsPerSample = 32;                 // 32-bit float

    // Initialize WAV file for writing
    drwav wav;
    if (!drwav_init_file_write(&wav, filename.c_str(), &format, nullptr))
    {
        std::cerr << "Failed to initialize WAV file writing: " << filename << std::endl;
        return;
    }

    // Write binaryBlob as PCM frames
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, outvector.size() / channels, outvector.data());
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

static RtAudio *dac = nullptr;

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

void InitAudio(int buffercount)
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
    parameters.nChannels = self.outchannels;
    parameters.firstChannel = 0;

    std::cout << "Opening output stream" << std::endl;
    // Open the stream with minimal buffering for low latency
    try
    {
        RtAudio::StreamOptions options;
        options.flags = RTAUDIO_SCHEDULE_REALTIME;
        options.numberOfBuffers = buffercount; // Minimum number of buffers for stable playback
        // options.flags = RTAUDIO_MINIMIZE_LATENCY; // Request minimum latency

        dac->openStream(&parameters, nullptr, RTAUDIO_FLOAT32,
                        self.outSampleRate, &self.framesPerDispatch, &audioCallback,
                        nullptr, &options);
        dac->startStream();
    }
    catch (RtAudioError &e)
    {
        std::cerr << "Error: " << e.getMessage() << std::endl;
        return;
    }
}

#define MIDI_KEY_COUNT 128

void Init(int polyphony, int samplesPerDispatch, int lfoCount, int envLenPerPatch, int outchannels, float bendDepth, float outSampleRate, int threadCount)
{
    // Initialize thread pool first
    InitThreadPool(threadCount);

    self.threadCount = threadCount;
    self.outSampleRate = outSampleRate;
    self.bendDepth = bendDepth;
    self.outchannels = outchannels;
    self.framesPerDispatch = samplesPerDispatch;
    self.envLenPerPatch = envLenPerPatch;
    self.polyphony = polyphony;
    self.rhodesEffectOn = false;
    self.loop = 0;
    self.masterVolume = 1.0 / (1 << 3);
    self.binaryBlob.clear(); // Initialize empty vector

    self.polyphony = polyphony;

    self.rhodesEffect.resize(self.outchannels, 0.0f);

    self.lfoCount = lfoCount;
    self.lfoPhase.resize(lfoCount, 0.0f);
    self.lfoIncreasePerDispatch.resize(lfoCount, 0.0f);

    self.dispatchFrameNo.resize(polyphony, 0.0f);

    self.outputPhaseFloor.resize(polyphony, std::vector<float>(self.framesPerDispatch, 0.0f));
    self.samples.resize(2, std::vector<std::vector<float>>(polyphony, std::vector<float>(self.framesPerDispatch, 0.0f)));
    self.fadeOut.resize(polyphony, std::vector<float>(self.framesPerDispatch, 0.0f));
    self.accumulation.resize(polyphony, std::vector<float>(self.framesPerDispatch, 0.0f));
    self.sampleWithinDispatchPostBend.resize(polyphony, std::vector<float>(self.framesPerDispatch, 0.0f));

    self.envelopeEnd.resize(polyphony, 0.0f);
    for (int voiceNo = 0; voiceNo < self.polyphony; voiceNo++)
    {
        self.envelopeEnd[voiceNo] = (voiceNo + 1) * self.envLenPerPatch - 1;
    }

    self.key2voiceIndex.resize(MIDI_KEY_COUNT);

    self.xfadeTracknot.resize(polyphony, 1.0f);
    self.xfadeTrack.resize(polyphony, 0.0f);

    self.voiceLoopStart.resize(polyphony, 0.0f);
    self.voiceLoopEnd.resize(polyphony, 0.0f);
    self.voiceLoopLength.resize(polyphony, 0.0f);
    self.slaveFade.resize(polyphony, 0.0f);
    self.voiceStart.resize(polyphony, 0.0f);
    self.voiceFrameCount.resize(polyphony, 0.0f);
    self.voiceDetune.resize(polyphony, 1.0f);
    self.voiceChannelCount.resize(polyphony, 1.0f);
    self.voiceChannelVol.resize(polyphony);
    self.noLoopFade.resize(polyphony, 0.0f);

    self.pitchWheel.resize(polyphony, 1.0f);
    self.portamento.resize(polyphony, 1.0f);
    self.portamentoAlpha.resize(polyphony, 1.0f);
    self.portamentoTarget.resize(polyphony, 1.0f);

    self.releaseVol.resize(polyphony, 0.0f);
    self.combinedEnvelope.resize(polyphony * envLenPerPatch, 0.0f);
    self.velocityVol.resize(polyphony, 0.0f);
    self.indexInEnvelope.resize(polyphony, 0.0f);

    self.nextEnvelopeVol.resize(polyphony, 0.0f);
    // No need to resize fixed array of mutexes
}

using json = nlohmann::json;

int currSampleIndex = 0;
int currAbsoluteSampleNo = 0;
int startSample = 0;

void SetVolume(float newVol)
{
    self.masterVolume = newVol;
}

// Function to get the envelope length per patch
int GetEnvLenPerPatch()
{
    return self.envLenPerPatch;
}

// Function to append data to the binaryBlob array
int AppendSample(std::vector<float> sample_array, int channels)
{
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
    if (!sample_start && false)
    {
        WriteVectorToWav(self.binaryBlob, "binaryBlob.wav", channels);
        WriteVectorToWav(sample_array, "sample.wav", channels);
    }

    // Store the Sample Details
    sampleStart.push_back(sample_start);
    sampleFrameCount.push_back(sample_array.size() / channels);

    // TODO: Implement loop
    sampleLoopStart.push_back(0);
    sampleLoopLength.push_back(sample_array.size());
    sampleLoopEnd.push_back(sample_array.size());

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

void ProcessVoices(int threadNo, int numThreads, float *outputBuffer)
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

    int firstVoice = threadNo * self.polyphony / numThreads;
    int lastVoice = firstVoice + self.polyphony / numThreads;
    // Process each voice
    for (int voiceNo = firstVoice; voiceNo < lastVoice; voiceNo++)
    {

        // std::cout << " VoiceNo " << voiceNo << " Voice Detune " << self.voiceDetune[voiceNo] << std::endl;
        int voiceChannelCount = self.voiceChannelCount[voiceNo];
        int voiceFrameCount = self.voiceFrameCount[voiceNo];
        float thisEnvelopeVol = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceNo])] * self.releaseVol[voiceNo] * self.velocityVol[voiceNo];

        if (self.indexInEnvelope[voiceNo] < self.envelopeEnd[voiceNo])
        {
            self.indexInEnvelope[voiceNo]++;
        }

        self.nextEnvelopeVol[voiceNo] = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceNo])] * self.releaseVol[voiceNo] * self.velocityVol[voiceNo];

        float difference = self.nextEnvelopeVol[voiceNo] - thisEnvelopeVol;

        // Process each sample within the dispatch
        for (int sampleNo = 0; sampleNo < self.framesPerDispatch; sampleNo++)
        {
            /*std::cout << "   Sample " << sampleNo << std::endl;*/

            // Update portamento for each voice, and for each sample
            self.portamento[voiceNo] = self.portamentoTarget[voiceNo] * self.portamentoAlpha[voiceNo] + (1.0f - self.portamentoAlpha[voiceNo]) * self.portamento[voiceNo];
            // Calculate the next dispatch phase
            float nextDispatchFrameNo = self.dispatchFrameNo[voiceNo] + self.voiceDetune[voiceNo] * self.portamento[voiceNo] * self.pitchWheel[voiceNo];

            // Clip the phase to valid sample indices and loop if necessary before updating
            if (nextDispatchFrameNo < 0)
            {
                // If phase goes negative, clamp to start
                nextDispatchFrameNo = 0;
            }
            else if (nextDispatchFrameNo * voiceChannelCount >= voiceFrameCount)
            {
                if (self.loop)
                {
                    nextDispatchFrameNo = fmodf(nextDispatchFrameNo - self.voiceLoopStart[voiceNo], self.voiceLoopLength[voiceNo]) + self.voiceLoopStart[voiceNo];
                }
                else
                {
                    nextDispatchFrameNo = voiceFrameCount / voiceChannelCount - 1;
                }
            }

            // Update the dispatch phase for the next cycle
            self.dispatchFrameNo[voiceNo] = nextDispatchFrameNo;

            float normalizedPosition = (float)sampleNo / (float)self.framesPerDispatch;
            float multiplier = difference * normalizedPosition + thisEnvelopeVol;

            // Separate the integer frame and the fractional offset.
            int sampleFrame = (int)floorf(self.dispatchFrameNo[voiceNo]);
            float fraction = self.dispatchFrameNo[voiceNo] - sampleFrame;

            // read samples and apply panning
            for (int inchannel = 0; inchannel < voiceChannelCount; inchannel++)
            {
                // Calculate the floor index using the number of channels per frame.
                int floorIndex = sampleFrame * voiceChannelCount + inchannel;
                int ceilIndex = floorIndex + voiceChannelCount;

                // Handle looping or clamping at the end of the sample.
                if (ceilIndex >= voiceFrameCount * voiceChannelCount)
                {
                    ceilIndex = self.loop ? self.voiceLoopStart[voiceNo] : voiceFrameCount * voiceChannelCount - 1;
                    floorIndex = ceilIndex - voiceChannelCount;
                }

                // Ensure indices remain within the binaryBlob bounds
                int startOffset = self.voiceStart[voiceNo];
                
                // Clamp indices to valid range
                int floorIdx = startOffset + floorIndex;
                int ceilIdx = startOffset + ceilIndex;
                
                if (floorIdx < 0 || floorIdx >= voiceFrameCount || ceilIdx < 0 || ceilIdx >= voiceFrameCount) {
                    // If indices are out of bounds, use zero samples
                    thisSample = 0.0f;
                    nextSample = 0.0f;
                } else {
                    thisSample = self.binaryBlob[floorIdx];
                    nextSample = self.binaryBlob[ceilIdx];
                }
                // Denormal prevention.
                if (fabs(thisSample) < 1e-15)
                    thisSample = 0.0f;
                if (fabs(nextSample) < 1e-15)
                    nextSample = 0.0f;

                for (int outchannel = 0; outchannel < self.outchannels; outchannel++)
                {
                    // Linear interpolation using the proper fractional offset.
                    self.samples[outchannel][voiceNo][sampleNo] =
                        (thisSample * (1.0f - fraction) + nextSample * fraction) * // Interpolated sample
                        self.masterVolume *                                        // Overall volume scaling
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
        if (self.dispatchFrameNo[voiceNo] >= voiceFrameCount && self.loop)
        {
            self.dispatchFrameNo[voiceNo] = fmodf(self.dispatchFrameNo[voiceNo] - self.voiceLoopStart[voiceNo], self.voiceLoopLength[voiceNo]) + self.voiceLoopStart[voiceNo];
        }
    }
}

void SumSamples(int threadNo, int numThreads, float *outputBuffer)
{
    int firstFrame = threadNo * self.framesPerDispatch / numThreads;
    int lastFrame = firstFrame + self.framesPerDispatch / numThreads;

    // Sum samples across polyphony and interleave channels
    // Clear the output buffer before processing
    std::fill(outputBuffer + firstFrame * self.outchannels, outputBuffer + lastFrame * self.outchannels, 0.0f);

    for (int sampleNo = firstFrame; sampleNo < lastFrame; sampleNo++)
    {
        for (int voiceNo = 0; voiceNo < self.polyphony; voiceNo++)
        {
            for (int channel = 0; channel < self.outchannels; channel++)
            {
                float thisSample = self.samples[channel][voiceNo][sampleNo];
                outputBuffer[sampleNo * self.outchannels + channel] += thisSample;
            }
        }
    }
}

int Strike(int sampleNo, float velocity, float sampleDetune, float *patchEnvelope)
{
    std::lock_guard<std::mutex> lock(threadVoiceLocks[strikeIndex * self.threadCount / self.polyphony]);

    // std::cout << "Striking sample " << sampleNo << " at strike index " << strikeIndex << std::endl;
    self.xfadeTrack[strikeIndex] = 0;
    self.xfadeTracknot[strikeIndex] = 1;
    self.dispatchFrameNo[strikeIndex] = 0;
    self.slaveFade[strikeIndex] = strikeIndex;
    self.noLoopFade[strikeIndex] = 1;

    // Transfer Sample params to Voice params for sequential access in Run
    self.voiceFrameCount[strikeIndex] = sampleFrameCount[sampleNo];
    self.voiceStart[strikeIndex] = sampleStart[sampleNo];
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

    self.voiceDetune[strikeIndex] = sampleDetune;
    // std::cout << "Striking voice " << strikeIndex << " with detune " << self.voiceDetune[strikeIndex] << std::endl;

    self.voiceChannelVol[strikeIndex] = sampleChannelVol[sampleNo];
    self.voiceChannelCount[strikeIndex] = sampleChannelCount[sampleNo];

    self.portamento[strikeIndex] = 1;
    self.portamentoAlpha[strikeIndex] = 1;
    self.portamentoTarget[strikeIndex] = 1;

    // implement Round Robin for simplicity
    strikeIndex = (strikeIndex + 1) % self.polyphony;
    return (strikeIndex + self.polyphony - 1) % self.polyphony;
}

void ReleaseAll(float *env)
{
    for (int key = 0; key < MIDI_KEY_COUNT; key++)
    {
        while (Release(key, env))
        {
        };
    }
}

int Release(int midi_key, float *env)
{
    int voicesToRelease = self.key2voiceIndex[midi_key].size();
    int voicesReleased = 0;
    for (size_t i = 0; i < voicesToRelease; ++i)
    {
        if (self.key2voiceIndex[midi_key].empty())
        {
            return voicesReleased; // No samples to release
        }
        int releaseIndex = self.key2voiceIndex[midi_key].front();

        std::lock_guard<std::mutex> lock(threadVoiceLocks[releaseIndex * self.threadCount / self.polyphony]);

        self.key2voiceIndex[midi_key].erase(self.key2voiceIndex[midi_key].begin()); // Remove first element

        self.indexInEnvelope[releaseIndex] = releaseIndex * self.envLenPerPatch;
        // Apply envelope release if provided
        if (env != nullptr)
        {
            for (int envPosition = 0; envPosition < self.envLenPerPatch; envPosition++)
            {
                int index = releaseIndex * self.envLenPerPatch + envPosition;
                self.combinedEnvelope[index] = env[envPosition] * self.releaseVol[releaseIndex];
            }
        }
        else
        {
            // Default release (full volume fade out)

            for (int envIndex = 0; envIndex < self.envLenPerPatch; envIndex++)
            {
                int envelopeIndex = releaseIndex * self.envLenPerPatch + envIndex;
                self.combinedEnvelope[envelopeIndex] = 1;
            }
        }
        voicesReleased++;
    }
    return voicesToRelease;
}

void HardStop()
{
    for (int voiceIndex = 0; voiceIndex < self.polyphony; voiceIndex++)
    {
        std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * self.threadCount / self.polyphony]);

        self.indexInEnvelope[strikeIndex] = strikeIndex * self.envLenPerPatch;
        self.releaseVol[strikeIndex] = 1.0f;
        self.velocityVol[strikeIndex] = 0.0f;
    }
}

#define ELEMENTS_TO_PRINT 16
void Dump(const char *filename)
{
    json output;

    // Store scalar values
    output["rhodesEffect"] = self.rhodesEffect;
    output["loop"] = self.loop;
    output["OVERVOLUME"] = self.masterVolume;

    output["lfoPhase"] = self.lfoPhase;
    output["lfoIncreasePerDispatch"] = self.lfoIncreasePerDispatch;
    output["dispatchFrameNo"] = self.dispatchFrameNo;

    output["xfadeTracknot"] = self.xfadeTracknot;
    output["xfadeTrack"] = self.xfadeTrack;
    output["loopStart"] = sampleLoopStart;
    output["loopEnd"] = sampleLoopEnd;
    output["loopLength"] = sampleLoopLength;
    output["slaveFade"] = self.slaveFade;

    output["sampleLen"] = self.voiceFrameCount;
    output["voiceDetune"] = self.voiceDetune;
    output["noLoopFade"] = self.noLoopFade;
    output["pitchWheel"] = self.pitchWheel;
    output["portamento"] = self.portamento;
    output["portamentoAlpha"] = self.portamentoAlpha;
    output["portamentoTarget"] = self.portamentoTarget;
    output["releaseVol"] = self.releaseVol;
    output["velocityVol"] = self.velocityVol;
    output["indexInEnvelope"] = self.indexInEnvelope;
    output["envelopeEnd"] = self.envelopeEnd;
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
    std::cout << "Clean up audio and threads" << std::endl;
    // Clean up audio
    DeInitAudio();
    // Clean up thread pool
    DestroyThreadPool();
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
    sampleFrequency.push_back(midiNoteTo12TETFreq(sample["pitch_keycenter"]));

    // generate the keybindings for 12 tone harmonic

    AppendSample(samples, inchannels);
    currAbsoluteSampleNo++;

    return 0;
}

void GenerateKeymap(double (*tuningSystemFn)(int))
{
    std::vector<std::vector<int>> thisKeyMap;
    std::vector<std::vector<float>> thisDetune;
    thisKeyMap.resize(MIDI_NOTES);
    thisDetune.resize(MIDI_NOTES);

    // find the nearest samples based on closest frequency
    for (int midiNote = 0; midiNote < MIDI_NOTES; midiNote++)
    {
        float desiredFrequency = tuningSystemFn(midiNote);

        // Find the sample with frequency closest to desired frequency
        float minDifference = std::numeric_limits<float>::max();
        int closestSampleIndex = -1;
        float closestRatio = 1.0f;

        // Only look at samples from the current instrument
        for (int i = startSample; i < currAbsoluteSampleNo; i++)
        {
            float difference = std::abs(std::log2(sampleFrequency[i] / desiredFrequency));
            if (difference < minDifference)
            {
                minDifference = difference;
                closestSampleIndex = i;
                closestRatio = desiredFrequency / sampleFrequency[i];
            }
        }

        if (closestSampleIndex >= 0)
        {
            thisKeyMap[midiNote].push_back(closestSampleIndex);
            thisDetune[midiNote].push_back(closestRatio);
        }
    }
    self.key2sampleIndexAll.push_back(thisKeyMap);
    self.key2sampleDetuneAll.push_back(thisDetune);
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
    // Generate the key bindings
    self.key2samples12tet = data[patchNoInFile]["key2samples"];

    bool useExplicitMapping = self.key2samples12tet.size() > 0;

    startSample = currAbsoluteSampleNo;
    patchStartSample.push_back(currAbsoluteSampleNo);

    // Load samples
    for (const auto &instrument : data)
    {
        for (const auto &sample : instrument["samples"])
        {
            LoadRestAudioB64(sample);
        }
    }
    patchSampleCount.push_back(currAbsoluteSampleNo - startSample);

    // If we have explicit key mappings in the JSON, use those
    if (!self.key2samples12tet.empty())
    {
        std::vector<std::vector<int>> thisKeyMap;
        std::vector<std::vector<float>> thisDetune;
        thisKeyMap.resize(MIDI_NOTES);
        thisDetune.resize(MIDI_NOTES);
        for (const auto &keyAction : self.key2samples12tet)
        {
            for (const auto &entry : keyAction)
            {
                int keyTrigger = entry["keyTrigger"].get<int>();
                int patchSampleNo = entry["sampleNo"].get<int>();
                float detuneRatio = entry["pitchBend"].get<float>();

                if (patchSampleNo >= startSample && patchSampleNo < currAbsoluteSampleNo)
                {
                    thisKeyMap[keyTrigger].push_back(patchSampleNo);
                    thisDetune[keyTrigger].push_back(detuneRatio);
                }
            }
        }
        self.key2sampleIndexAll.push_back(thisKeyMap);
        self.key2sampleDetuneAll.push_back(thisDetune);
    }
    else
    {
        // Otherwise generate mappings automatically
        GenerateKeymap(midiNoteTo12TETFreq);
    }

    // Always generate Maqam Rast mappings
    GenerateKeymap(midiNoteToMaqamRastFreq);
    GenerateKeymap(midiNoteToPythagoreanFreq);
    GenerateKeymap(midiNoteToRagaYamanFreq);
    GenerateKeymap(midiNoteToBohlenPierceFreq);
    GenerateKeymap(midiNoteToMaqamBayatiFreq);
    GenerateKeymap(midiNoteToSlendroPelogFreq);
    GenerateKeymap(midiNoteToHarmonicSeriesFreq);

    SetTuningSystem(0);

    std::cout << "Finished loading key mappings" << std::endl;
    std::cout << "Binary Blob size " << self.binaryBlob.size() << std::endl;
}

void SetTuningSystem(int tuningSystem)
{
    currentTuningSystem = tuningSystem;
    key2sampleIndex = &self.key2sampleIndexAll[tuningSystem];
    key2sampleDetune = &self.key2sampleDetuneAll[tuningSystem];
}

void ProcessMidi(std::vector<unsigned char> *message)
{
    unsigned int status = message->at(0);
    unsigned int midi_key = message->at(1);
    float velocity = int(message->at(2)) / 127.0f;

    // std::cout << "MIDI Status: " << status << ", Key: " << midi_key << ", Velocity: " << velocity << std::endl;

    // 🎹 **Note On**
    if ((status & 0xF0) == 0x90 && velocity > 0)
    {
        // Get the vector of sample indices for this MIDI key
        const auto& sampleIndices = (*key2sampleIndex)[midi_key];
        const auto& sampleDetunes = (*key2sampleDetune)[midi_key];
        
        // Iterate through each sample index for this key
        for (size_t i = 0; i < sampleIndices.size(); i++)
        {
            int voiceIndex = Strike(sampleIndices[i], velocity, sampleDetunes[i], nullptr);
            self.key2voiceIndex[midi_key].push_back(voiceIndex);
        }
    }

    // 🎵 **Note Off**
    else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0))
    {
        while (!self.key2voiceIndex[midi_key].empty())
        {
            int voiceIndex = self.key2voiceIndex[midi_key].front();
            self.key2voiceIndex[midi_key].erase(self.key2voiceIndex[midi_key].begin());
            Release(voiceIndex, nullptr);
        }
    }

    // 🎛 **Control Change (CC) - Handling Sustain Pedal & Other Controls**
    else if ((status & 0xF0) == 0xB0)
    {
        unsigned int cc_number = message->at(1);
        float cc_value = message->at(2) / 127.0f;

        switch (cc_number)
        {
        case MIDI_CC_SUSTAIN:
            self.sustainPedalOn = (cc_value > 0.5f);
            std::cout << "Sustain Pedal: " << (self.sustainPedalOn ? "ON" : "OFF") << std::endl;
            if (!self.sustainPedalOn)
            {
                // Release all sustained notes
                for (int note = 0; note < MIDI_NOTES; note++)
                {
                    while (!self.key2voiceIndex[note].empty())
                    {
                        int voiceIndex = self.key2voiceIndex[note].front();
                        self.key2voiceIndex[note].erase(self.key2voiceIndex[note].begin());
                        Release(voiceIndex, nullptr);
                    }
                }
            }
            break;

        case MIDI_CC_VOLUME:
            self.masterVolume = cc_value;
            std::cout << "Master Volume: " << self.masterVolume << std::endl;
            break;

        case MIDI_CC_MODULATION:
            self.modulationDepth = cc_value;
            std::cout << "Modulation Depth: " << self.modulationDepth << std::endl;
            break;

        case MIDI_CC_EXPRESSION:
            self.expression = cc_value;
            std::cout << "Expression: " << self.expression << std::endl;
            break;

        default:
            std::cout << "Unhandled MIDI CC: " << cc_number << " Value: " << cc_value << std::endl;
            break;
        }
    }

    // 🎚 **Pitch Bend Handling**
    else if ((status & 0xF0) == 0xE0)
    {
        int bendLSB = message->at(1);                        // Least Significant Byte
        int bendMSB = message->at(2);                        // Most Significant Byte
        int bendValue = (bendMSB << 7) | bendLSB;            // Combine into 14-bit value
        float normalizedBend = (bendValue - 8192) / 8192.0f; // Convert to -1.0 to +1.0 range
        normalizedBend *= 2;                                 // 2 step bend

        std::cout << "Pitch Bend: " << normalizedBend << std::endl;

        // Apply pitch bend to all active voices
        for (int voiceIndex = 0; voiceIndex < self.polyphony; voiceIndex++)
        {
            std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * self.threadCount / self.polyphony]);
            self.pitchWheel[voiceIndex] = normalizedBend;
        }
    }
}

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData)
{
    auto *buffer = static_cast<float *>(outputBuffer);
    std::lock_guard<std::mutex> lock(self.blobMutex);
    if (self.threadCount < 2)
    {
        ProcessVoices(0, 1, buffer);
        SumSamples(0, 1, buffer);
    }
    else
    {
        RunMultithread(self.threadCount, buffer, ProcessVoicesThreadWrapper);
        RunMultithread(self.threadCount, buffer, SumSamplesThreadWrapper);
    }
    return 0;
}

void Test()
{
    std::cout << "Test mode activated" << std::endl;

    int samples_per_dispatch = 128;
    Init(10, samples_per_dispatch, 2, 512, 2, 12, 48000, 4);
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
    const int samplesPerChannel = self.outSampleRate * secondsToGenerate;
    const int totalSamples = samplesPerChannel * 2; // Stereo output

    // Allocate buffer with proper size for stereo output
    std::vector<float> buffer(totalSamples);

    // Generate audio in chunks of samples_per_dispatch
    const int numBuffers = samplesPerChannel / samples_per_dispatch;

    for (int i = 0; i < numBuffers; i++)
    {
        // Compute the starting index in the buffer vector
        int startIndex = i * samples_per_dispatch * self.outchannels;
        // Send a MIDI note every 100 iterations
        if (i % 100 == 0)
        {
            std::vector<unsigned char> message = {0x90, penta[(i / 100) % 5], 127}; // Note on
            ProcessMidi(&message);
        }
        ProcessVoices(0, 1, buffer.data() + startIndex);

        // Dump first buffer for debugging
        if (i == 0)
            Dump("dump.json");
    }

    WriteVectorToWav(buffer, "Outfile.wav", self.outchannels);
    float env[self.envLenPerPatch];
    std::fill_n(env, self.envLenPerPatch, 0.0f); // ✅ Correct usage of std::fill_n()
    ReleaseAll(env);

    std::vector<unsigned char> message = {0x90, 45, 127}; // Note on
    ProcessMidi(&message);

    for (int i = 0; i < numBuffers; i++)
    {
        // Compute the starting index in the buffer vector
        int startIndex = i * samples_per_dispatch * self.outchannels;
        ProcessVoices(0, 1, buffer.data() + startIndex);

        // increase the pitch
        float bendAmount = float(i) / numBuffers;
        int bendValue = static_cast<int>((bendAmount + 1.0) * 8192.0);
        bendValue = std::clamp(bendValue, 0, 16383);           // Ensure within valid range
        unsigned char lsb = bendValue & 0x7F;                  // Least Significant Byte
        unsigned char msb = (bendValue >> 7) & 0x7F;           // Most Significant Byte
        std::vector<unsigned char> message = {0xE0, lsb, msb}; // Pitch Bend on Channel 1
        // ProcessMidi(&message);
    }

    WriteVectorToWav(buffer, "Bend.wav", self.outchannels);
}
