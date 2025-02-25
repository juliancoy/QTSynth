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
std::mutex threadVoiceLocks[128]; // Fixed array of locks, one per voice

int strikeIndex = 0;

// Global variable to track current tuning system
static int currentTuningSystem = 0;

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
    format.sampleRate = self.outSampleRate;    // Sample rate
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
    self.outchannels = outchannels;
    self.framesPerDispatch = samplesPerDispatch;
    self.envLenPerPatch = envLenPerPatch;
    self.polyphony = polyphony;
    self.rhodesEffectOn = false;
    self.loop = false;
    self.masterVolume = 1.0 / (1 << 3);

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

    self.xfadeTracknot.resize(polyphony, 1.0f);
    self.xfadeTrack.resize(polyphony, 0.0f);

    self.slaveFade.resize(polyphony, 0.0f);
    self.voiceDetune.resize(polyphony, 1.0f);
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

    // Initialize voiceSamplePtr vector
    self.voiceSamplePtr.resize(polyphony, nullptr);

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

// Function to append multichannel audio data
int AppendSample(std::vector<float> samples, float sampleRate, int inchannels, float baseFrequency)
{

    // Number of frames (samples per channel)
    size_t frameCount = samples.size() / inchannels;

    // Determine the mapping of input channels to output channels
    // For example if the input is Mono, it should distribute even power to both sides
    // the same amound of power that the left channel of a stereo signal puts out on one
    // 2D vector to store gains: gains[inchannel][outchannel]
    std::vector<std::vector<float>> volumeMatrix(inchannels, std::vector<float>(self.outchannels, 0.0f));

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
            volumeMatrix[inchannel][outchannel] = gain;
            totalPower += gain * gain;
        }

        // Normalize for constant power
        float normalizationFactor = 1.0f / std::sqrt(totalPower);
        for (size_t outchannel = 0; outchannel < self.outchannels; outchannel++)
        {
            volumeMatrix[inchannel][outchannel] *= normalizationFactor;
        }
    }

    std::lock_guard<std::mutex> lock(self.samplesMutex);

    SampleData newSample;
    newSample.volumeMatrix = volumeMatrix;
    newSample.numChannels = inchannels;
    newSample.numFrames = frameCount;
    newSample.baseFrequency = baseFrequency;
    // Allocate memory for samples and copy the data
    newSample.samples = new float[samples.size()];
    std::copy(samples.begin(), samples.end(), newSample.samples);
    newSample.loopStart = 0;
    newSample.loopLength = newSample.numFrames;
    newSample.loopEnd = newSample.numFrames;
    newSample.strikeVolume = 1.0f;
    newSample.envelope = nullptr;

    return 0;
}

// Function to delete a sample from a patch
void DeleteSample(int sampleNo, Patch *patch)
{
    if (sampleNo >= 0 && sampleNo < patch->samplesData.size())
    {
        patch->samplesData.erase(patch->samplesData.begin() + sampleNo);
    }
}

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
        SampleData *sample = self.voiceSamplePtr[voiceNo];
        float thisEnvelopeVol = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceNo])] * self.releaseVol[voiceNo] * self.velocityVol[voiceNo];

        if (sample == nullptr)
        {
            continue;
        }

        int voiceChannelCount = sample->numChannels;
        int voiceFrameCount = sample->numFrames;
        int voiceLoopEnd = sample->loopEnd;
        int voiceLoopLength = sample->loopLength;
        int voiceLoopStart = sample->loopStart;

        if (self.indexInEnvelope[voiceNo] < self.envelopeEnd[voiceNo])
        {
            self.indexInEnvelope[voiceNo]++;
        }

        self.nextEnvelopeVol[voiceNo] = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceNo])] * self.releaseVol[voiceNo] * self.velocityVol[voiceNo];

        float difference = self.nextEnvelopeVol[voiceNo] - thisEnvelopeVol;
        float dispatchFrameNo = self.dispatchFrameNo[voiceNo];

        // Process each sample within the dispatch
        for (int sampleNo = 0; sampleNo < self.framesPerDispatch; sampleNo++)
        {
            // Update portamento for each voice, and for each sample
            self.portamento[voiceNo] = self.portamentoTarget[voiceNo] * self.portamentoAlpha[voiceNo] + (1.0f - self.portamentoAlpha[voiceNo]) * self.portamento[voiceNo];

            float normalizedPosition = (float)sampleNo / (float)self.framesPerDispatch;
            float multiplier = difference * normalizedPosition + thisEnvelopeVol;

            // Separate the integer frame and the fractional offset.
            int sampleFrame = (int)floorf(dispatchFrameNo);
            float fraction = dispatchFrameNo - sampleFrame;

            // Calculate the floor index using the number of channels per frame.
            int floorIndex = sampleFrame;
            int ceilIndex = floorIndex + 1;

            // read samples and apply panning
            for (int inchannel = 0; inchannel < voiceChannelCount; inchannel++)
            {
                for (int outchannel = 0; outchannel < self.outchannels; outchannel++)
                {
                    // Calculate sample indices for interleaved format
                    size_t baseIndex = floorIndex * voiceChannelCount + inchannel;
                    float thisSample = sample->samples[baseIndex];
                    float nextSample = sample->samples[baseIndex + voiceChannelCount];

                    // Get channel volume from precomputed matrix
                    float channelVol = sample->volumeMatrix[inchannel][outchannel];

                    // Linear interpolation using the proper fractional offset
                    float volumeFx = self.masterVolume * channelVol * self.rhodesEffect[outchannel];
                    float interpolatedSample = (thisSample * (1.0f - fraction) + nextSample * fraction);
                    float thisVoiceContribution = interpolatedSample * volumeFx;

                    self.samples[outchannel][voiceNo][sampleNo] = thisVoiceContribution;
                }
            }

            // Apply fade out if needed
            if (self.loop)
            {
                // Fade out around the loop points
                self.fadeOut[voiceNo][sampleNo] = fminf(fabsf(self.outputPhaseFloor[voiceNo][sampleNo] - (voiceLoopEnd - voiceLoopLength + self.slaveFade[voiceNo])) / fadelen, 1.0f);
                self.fadeOut[voiceNo][sampleNo] = fminf(fabsf(self.outputPhaseFloor[voiceNo][sampleNo] - (voiceLoopEnd + self.slaveFade[voiceNo])) / fadelen, self.fadeOut[voiceNo][sampleNo]);

                // Applying fade out logic based on slaveFade
                // Assuming slaveFade is an index, not an array, as it is not clear from the Python code
                self.fadeOut[voiceNo][sampleNo] = self.xfadeTracknot[voiceNo] * self.fadeOut[voiceNo][sampleNo] + self.xfadeTrack[voiceNo] * (1.0f - self.fadeOut[voiceNo][sampleNo]);

                // Ensure fadeOut is not less than noLoopFade
                self.fadeOut[voiceNo][sampleNo] = fmaxf(self.fadeOut[voiceNo][sampleNo], self.noLoopFade[voiceNo]);

                // Apply fadeOut to samples
                self.samples[0][voiceNo][sampleNo] *= self.fadeOut[voiceNo][sampleNo];
            }

            // Calculate the next dispatch phase
            float increment = self.voiceDetune[voiceNo] * self.portamento[voiceNo] * self.pitchWheel[voiceNo];
            // Remove debug print for performance

            dispatchFrameNo = std::clamp(dispatchFrameNo + increment, 0.0f, voiceFrameCount - 2.0f);

            if (self.loop && self.dispatchFrameNo[voiceNo] >= voiceLoopEnd)
            {
                dispatchFrameNo = voiceLoopStart;
            }
        }
        // Update the dispatch phase for the next cycle
        self.dispatchFrameNo[voiceNo] = dispatchFrameNo;
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

int Strike(SampleData *sample, float velocity, float sampleDetune, float *patchEnvelope)
{
    std::lock_guard<std::mutex> lock(threadVoiceLocks[strikeIndex * self.threadCount / self.polyphony]);

    // std::cout << "Striking sample " << sampleNo << " at strike index " << strikeIndex << std::endl;
    self.xfadeTrack[strikeIndex] = 0;
    self.xfadeTracknot[strikeIndex] = 1;
    self.dispatchFrameNo[strikeIndex] = 0;
    self.slaveFade[strikeIndex] = strikeIndex;
    self.noLoopFade[strikeIndex] = 1;

    // Transfer Sample params to Voice params for sequential access in Run
    self.voiceSamplePtr[strikeIndex] = sample;

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

int Release(int midi_key, float *env, Patch *patch)
{
    int voicesToRelease = patch->key2voiceIndex[midi_key].size();
    int voicesReleased = 0;
    for (size_t i = 0; i < voicesToRelease; ++i)
    {
        if (patch->key2voiceIndex[midi_key].empty())
        {
            return voicesReleased; // No samples to release
        }
        int releaseIndex = patch->key2voiceIndex[midi_key].front();

        std::lock_guard<std::mutex> lock(threadVoiceLocks[releaseIndex * self.threadCount / self.polyphony]);

        patch->key2voiceIndex[midi_key].erase(patch->key2voiceIndex[midi_key].begin()); // Remove first element

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
    output["slaveFade"] = self.slaveFade;

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
    int sampleRate = 0;
    int inchannels = 0;

    // Decode Base64 to binary
    const std::string &base64Data = sample["audioData"].get<std::string>();
    std::string binaryData = base64_decode(base64Data);

    if (sample["audioFormat"] == "wav")
    {
        drwav wav;
        if (drwav_init_memory(&wav, binaryData.data(), binaryData.size(), nullptr))
        {
            size_t totalFrames = wav.totalPCMFrameCount * wav.channels;
            samples.resize(totalFrames); // Allocate memory for PCM data

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
            size_t totalFrames = frameCount * mp3.channels;
            samples.resize(totalFrames); // Allocate memory for PCM data

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

    float baseFrequency = midiNoteTo12TETFreq(sample["pitch_keycenter"]);
    // Append unaltered interleaved samples
    AppendSample(samples, sampleRate, inchannels, baseFrequency);
    currAbsoluteSampleNo++;

    return 0;
}

void GenerateKeymap(double (*tuningSystemFn)(int), Patch *patch)
{
    // Create vectors to store sample indices and detune values
    std::vector<std::vector<int>> thisKeyMap(MIDI_NOTES);
    std::vector<std::vector<float>> thisDetune(MIDI_NOTES);

    // Create a temporary map to store pointers for quick lookup
    std::vector<SampleData *> samplePtrs;
    for (SampleData &sample : patch->samplesData)
    {
        samplePtrs.push_back(&sample);
    }

    // Find the nearest samples based on closest frequency
    for (int midiNote = 0; midiNote < MIDI_NOTES; midiNote++)
    {
        float desiredFrequency = tuningSystemFn(midiNote);

        // Find the sample with frequency closest to desired frequency
        float minDifference = std::numeric_limits<float>::max();
        SampleData *closestSample = nullptr;
        float closestRatio = 1.0f;

        // Only look at samples from the current instrument
        for (size_t i = 0; i < patch->samplesData.size(); i++)
        {
            SampleData &sample = patch->samplesData[i];
            float difference = std::abs(std::log2(sample.baseFrequency / desiredFrequency));
            if (difference < minDifference)
            {
                minDifference = difference;
                closestSample = &sample;
                closestRatio = desiredFrequency / sample.baseFrequency;
            }
        }

        if (closestSample != nullptr)
        {
            // Find the index of the closest sample in the samplesData vector
            auto it = std::find(samplePtrs.begin(), samplePtrs.end(), closestSample);
            if (it != samplePtrs.end())
            {
                int sampleIndex = std::distance(samplePtrs.begin(), it);
                thisKeyMap[midiNote].push_back(sampleIndex);
                thisDetune[midiNote].push_back(closestRatio);
            }
        }
    }

    // Add the new keymap to the patch
    patch->key2sampleIndexAll.push_back(thisKeyMap);
    patch->key2sampleDetuneAll.push_back(thisDetune);
}

// Load samples from JSON file
void LoadSoundJSON(const std::string &filename)
{
    // Create a new patch
    Patch newPatch;
    newPatch.bendDepth = 2;
    newPatch.key2voiceIndex.resize(MIDI_KEY_COUNT);

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
    newPatch.key2samples12tet = data[patchNoInFile]["key2samples"];

    bool useExplicitMapping = newPatch.key2samples12tet.size() > 0;

    startSample = currAbsoluteSampleNo;

    // Load samples
    for (const auto &instrument : data)
    {
        for (const auto &sample : instrument["samples"])
        {
            LoadRestAudioB64(sample);
        }
    }

    // If we have explicit key mappings in the JSON, use those
    if (!newPatch.key2samples12tet.empty())
    {
        std::vector<std::vector<int>> thisKeyMap;
        std::vector<std::vector<float>> thisDetune;
        thisKeyMap.resize(MIDI_NOTES);
        thisDetune.resize(MIDI_NOTES);
        for (const auto &keyAction : newPatch.key2samples12tet)
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
        newPatch.key2sampleIndexAll.push_back(thisKeyMap);
        newPatch.key2sampleDetuneAll.push_back(thisDetune);
    }
    else
    {
        // Otherwise generate mappings automatically
        GenerateKeymap(midiNoteTo12TETFreq, &newPatch);
    }

    // Always generate Maqam Rast mappings
    GenerateKeymap(midiNoteToMaqamRastFreq, &newPatch);
    GenerateKeymap(midiNoteToPythagoreanFreq, &newPatch);
    GenerateKeymap(midiNoteToRagaYamanFreq, &newPatch);
    GenerateKeymap(midiNoteToBohlenPierceFreq, &newPatch);
    GenerateKeymap(midiNoteToMaqamBayatiFreq, &newPatch);
    GenerateKeymap(midiNoteToSlendroPelogFreq, &newPatch);
    GenerateKeymap(midiNoteToHarmonicSeriesFreq, &newPatch);

    SetTuningSystem(0);

    std::cout << "Finished loading key mappings" << std::endl;
}

void SetTuningSystem(int tuningSystem)
{
    for (Patch &patch : patches)
    {
        currentTuningSystem = tuningSystem;
        patch.key2sampleIndex = &patch.key2sampleIndexAll[tuningSystem];
        patch.key2sampleDetune = &patch.key2sampleDetuneAll[tuningSystem];
    }
}

void PatchProcessMidi(std::vector<unsigned char> *message, Patch *patch)
{
    unsigned int status = message->at(0);
    unsigned int midi_key = message->at(1);
    float velocity = int(message->at(2)) / 127.0f;

    // std::cout << "MIDI Status: " << status << ", Key: " << midi_key << ", Velocity: " << velocity << std::endl;

    // 🎹 **Note On**
    if ((status & 0xF0) == 0x90 && velocity > 0)
    {
        // Get the vector of sample indices for this MIDI key
        const auto &sampleIndices = (*patch->key2sampleIndex)[midi_key];
        const auto &sampleDetunes = (*patch->key2sampleDetune)[midi_key];

        // Iterate through each sample index for this key
        for (int sampleNo = 0; sampleNo < sampleIndices.size(); sampleNo++)
        {
            int voiceIndex = Strike(&patch->samplesData[sampleIndices[sampleNo]], velocity, sampleDetunes[sampleNo], nullptr);
            patch->key2voiceIndex[midi_key].push_back(voiceIndex);
        }
    }

    // 🎵 **Note Off**
    else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0))
    {
        while (!patch->key2voiceIndex[midi_key].empty())
        {
            int voiceIndex = patch->key2voiceIndex[midi_key].front();
            patch->key2voiceIndex[midi_key].erase(patch->key2voiceIndex[midi_key].begin());
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
            patch->sustainPedalOn = (cc_value > 0.5f);
            std::cout << "Sustain Pedal: " << (patch->sustainPedalOn ? "ON" : "OFF") << std::endl;
            if (!patch->sustainPedalOn)
            {
                // Release all sustained notes
                for (int note = 0; note < MIDI_NOTES; note++)
                {
                    while (!patch->key2voiceIndex[note].empty())
                    {
                        int voiceIndex = patch->key2voiceIndex[note].front();
                        patch->key2voiceIndex[note].erase(patch->key2voiceIndex[note].begin());
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
            patch->modulationDepth = cc_value;
            std::cout << "Modulation Depth: " << patch->modulationDepth << std::endl;
            break;

        case MIDI_CC_EXPRESSION:
            patch->expression = cc_value;
            std::cout << "Expression: " << patch->expression << std::endl;
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

        // Define the pitch bend range in semitones (e.g., ±2 semitones)
        const float pitchBendRange = 2.0f;

        // Calculate the frequency multiplier
        float frequencyMultiplier = std::pow(2.0f, (normalizedBend * pitchBendRange) / 12.0f);

        std::cout << "Pitch Bend: " << frequencyMultiplier << std::endl;

        // Apply pitch bend to all active voices
        for (int voiceIndex = 0; voiceIndex < self.polyphony; voiceIndex++)
        {
            std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * self.threadCount / self.polyphony]);
            self.pitchWheel[voiceIndex] = frequencyMultiplier;
        }
    }
}

void ProcessMidi(std::vector<unsigned char> *message)
{
    for (Patch patch : patches)
    {
        PatchProcessMidi(message, &patch);
    }
}

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData)
{
    auto *buffer = static_cast<float *>(outputBuffer);
    std::lock_guard<std::mutex> lock(self.samplesMutex);
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
