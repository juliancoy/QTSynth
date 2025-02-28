// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#include "sample_compute.hpp"
#include <algorithm>
#include <string>
#include <limits>

#include <fstream>
#include <vector>
#include <mutex>
#include "dr_wav.h"

#include "tuning.hpp"
#include "patch.hpp"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>

#define MIDI_NOTES 128
#define MIDI_CC_SUSTAIN 64
#define MIDI_CC_VOLUME 7
#define MIDI_CC_MODULATION 1
#define MIDI_CC_EXPRESSION 11

#define MIDI_KEY_COUNT 128

SampleCompute::SampleCompute(int polyphony, int samplesPerDispatch, int lfoCount, int outchannels, float bendDepth, float outSampleRate, int threadCount)
{
    // Initialize thread pool first
    InitThreadPool(threadCount);

    threadCount = threadCount;
    outSampleRate = outSampleRate;
    outchannels = outchannels;
    framesPerDispatch = samplesPerDispatch;
    polyphony = polyphony;
    rhodesEffectOn = false;
    loop = false;
    masterVolume = 1.0 / (1 << 3);

    polyphony = polyphony;

    rhodesEffect.resize(outchannels, 0.0f);

    lfoCount = lfoCount;
    lfoPhase.resize(lfoCount, 0.0f);
    lfoIncreasePerDispatch.resize(lfoCount, 0.0f);

    voiceDispatchFrameNo.resize(polyphony, 0.0f);

    outputPhaseFloor.resize(polyphony, std::vector<float>(framesPerDispatch, 0.0f));
    samples.resize(2, std::vector<std::vector<float>>(polyphony, std::vector<float>(framesPerDispatch, 0.0f)));
    fadeOut.resize(polyphony, std::vector<float>(framesPerDispatch, 0.0f));
    accumulation.resize(polyphony, std::vector<float>(framesPerDispatch, 0.0f));
    sampleWithinDispatchPostBend.resize(polyphony, std::vector<float>(framesPerDispatch, 0.0f));

    xfadeTracknot.resize(polyphony, 1.0f);
    xfadeTrack.resize(polyphony, 0.0f);

    slaveFade.resize(polyphony, 0.0f);
    voiceDetune.resize(polyphony, 1.0f);
    noLoopFade.resize(polyphony, 0.0f);

    pitchWheel.resize(polyphony, 1.0f);
    portamento.resize(polyphony, 1.0f);
    portamentoAlpha.resize(polyphony, 1.0f);
    portamentoTarget.resize(polyphony, 1.0f);

    releaseVol.resize(polyphony, 0.0f);
    voiceEnvelope.resize(polyphony);
    velocityVol.resize(polyphony, 0.0f);
    indexInEnvelope.resize(polyphony, 0.0f);

    nextEnvelopeVol.resize(polyphony, 0.0f);

    // Initialize voiceSamplePtr vector
    voiceSamplePtr.resize(polyphony, nullptr);

    // No need to resize fixed array of mutexes
}

int currSampleIndex = 0;
int currAbsoluteSampleNo = 0;
int startSample = 0;

#define ELEMENTS_TO_PRINT 16
void SampleCompute::Dump(const char *filename)
{
    json output;

    // Store scalar values
    output["rhodesEffect"] = rhodesEffect;
    output["loop"] = loop;
    output["OVERVOLUME"] = masterVolume;

    output["lfoPhase"] = lfoPhase;
    output["lfoIncreasePerDispatch"] = lfoIncreasePerDispatch;
    output["dispatchFrameNo"] = voiceDispatchFrameNo;

    output["xfadeTracknot"] = xfadeTracknot;
    output["xfadeTrack"] = xfadeTrack;
    output["slaveFade"] = slaveFade;

    output["voiceDetune"] = voiceDetune;
    output["noLoopFade"] = noLoopFade;
    output["pitchWheel"] = pitchWheel;
    output["portamento"] = portamento;
    output["portamentoAlpha"] = portamentoAlpha;
    output["portamentoTarget"] = portamentoTarget;
    output["releaseVol"] = releaseVol;
    output["velocityVol"] = velocityVol;
    output["indexInEnvelope"] = indexInEnvelope;
    output["nextEnvelopeVol"] = nextEnvelopeVol;

    /*{
        std::lock_guard<std::mutex> lock(blobMutex);
        if (!binaryBlob.empty()) {
            output["binaryBlob"] = binaryBlob;
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

void SampleCompute::ProcessVoices(int threadNo, int numThreads, float *outputBuffer)
{
    float fadelen = 50000.0f;

    // Update LFO phases
    int lfoStart = threadNo * lfoCount / numThreads;
    int lfoEnd = lfoStart + lfoCount / numThreads;
    for (int lfoNo = lfoStart; lfoNo < lfoEnd; lfoNo++)
    {
        lfoPhase[lfoNo] += lfoIncreasePerDispatch[lfoNo];
        // Ensure LFO phase wraps around properly
        while (lfoPhase[lfoNo] >= 1.0f)
        {
            lfoPhase[lfoNo] -= 1.0f;
        }
    }

    // if rhodes effect is on, sweep the sound from left to right
    float volRight;
    float depth = 0.4f;
    for (int channel = 0; channel < outchannels; channel++)
    {
        if (rhodesEffectOn)
            rhodesEffect[channel] = ((1 - depth) + sinf(2 * M_PI * fmodf(lfoPhase[0] + 1.0 / outchannels, 1.0f)) * depth);
        else
            rhodesEffect[channel] = 1;
    }

    int firstVoice = threadNo * polyphony / numThreads;
    int lastVoice = firstVoice + polyphony / numThreads;
    // Process each voice
    for (int voiceNo = firstVoice; voiceNo < lastVoice; voiceNo++)
    {
        SampleData *sample = voiceSamplePtr[voiceNo];
        std::vector<float> thisEnv = *voiceEnvelope[voiceNo];
        float thisEnvelopeVol = thisEnv[(int)(indexInEnvelope[voiceNo])] * releaseVol[voiceNo] * velocityVol[voiceNo];

        if (sample == nullptr)
        {
            continue;
        }

        int voiceChannelCount = sample->numChannels;
        int voiceFrameCount = sample->numFrames;
        int voiceLoopEnd = sample->loopEnd;
        int voiceLoopLength = sample->loopLength;
        int voiceLoopStart = sample->loopStart;


        indexInEnvelope[voiceNo] = std::clamp(indexInEnvelope[voiceNo]+1, 0.0f, (float)thisEnv.size());

        nextEnvelopeVol[voiceNo] = thisEnv[(int)(indexInEnvelope[voiceNo])] * releaseVol[voiceNo] * velocityVol[voiceNo];

        float difference = nextEnvelopeVol[voiceNo] - thisEnvelopeVol;
        float dispatchFrameNo = voiceDispatchFrameNo[voiceNo];

        // Process each sample within the dispatch
        for (int sampleNo = 0; sampleNo < framesPerDispatch; sampleNo++)
        {
            // Update portamento for each voice, and for each sample
            portamento[voiceNo] = portamentoTarget[voiceNo] * portamentoAlpha[voiceNo] + (1.0f - portamentoAlpha[voiceNo]) * portamento[voiceNo];

            float normalizedPosition = (float)sampleNo / (float)framesPerDispatch;
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
                for (int outchannel = 0; outchannel < outchannels; outchannel++)
                {
                    // Calculate sample indices for interleaved format
                    size_t baseIndex = floorIndex * voiceChannelCount + inchannel;
                    float thisSample = sample->samples[baseIndex];
                    float nextSample = sample->samples[baseIndex + voiceChannelCount];

                    // Get channel volume from precomputed matrix
                    float channelVol = sample->volumeMatrix[inchannel][outchannel];

                    // Linear interpolation using the proper fractional offset
                    float volumeFx = masterVolume * channelVol * rhodesEffect[outchannel];
                    float interpolatedSample = (thisSample * (1.0f - fraction) + nextSample * fraction);
                    float thisVoiceContribution = interpolatedSample * volumeFx;

                    samples[outchannel][voiceNo][sampleNo] = thisVoiceContribution;
                }
            }

            // Apply fade out if needed
            if (loop)
            {
                // Fade out around the loop points
                fadeOut[voiceNo][sampleNo] = fminf(fabsf(outputPhaseFloor[voiceNo][sampleNo] - (voiceLoopEnd - voiceLoopLength + slaveFade[voiceNo])) / fadelen, 1.0f);
                fadeOut[voiceNo][sampleNo] = fminf(fabsf(outputPhaseFloor[voiceNo][sampleNo] - (voiceLoopEnd + slaveFade[voiceNo])) / fadelen, fadeOut[voiceNo][sampleNo]);

                // Applying fade out logic based on slaveFade
                // Assuming slaveFade is an index, not an array, as it is not clear from the Python code
                fadeOut[voiceNo][sampleNo] = xfadeTracknot[voiceNo] * fadeOut[voiceNo][sampleNo] + xfadeTrack[voiceNo] * (1.0f - fadeOut[voiceNo][sampleNo]);

                // Ensure fadeOut is not less than noLoopFade
                fadeOut[voiceNo][sampleNo] = fmaxf(fadeOut[voiceNo][sampleNo], noLoopFade[voiceNo]);

                // Apply fadeOut to samples
                samples[0][voiceNo][sampleNo] *= fadeOut[voiceNo][sampleNo];
            }

            // Calculate the next dispatch phase
            float increment = voiceDetune[voiceNo] * portamento[voiceNo] * pitchWheel[voiceNo];
            // Remove debug print for performance

            dispatchFrameNo = std::clamp(dispatchFrameNo + increment, 0.0f, voiceFrameCount - 2.0f);

            if (loop && dispatchFrameNo >= voiceLoopEnd)
            {
                dispatchFrameNo = voiceLoopStart;
            }
        }
        // Update the dispatch phase for the next cycle
        voiceDispatchFrameNo[voiceNo] = dispatchFrameNo;
    }
}

void SampleCompute::SumSamples(int threadNo, int numThreads, float *outputBuffer)
{
    int firstFrame = threadNo * framesPerDispatch / numThreads;
    int lastFrame = firstFrame + framesPerDispatch / numThreads;

    // Sum samples across polyphony and interleave channels
    // Clear the output buffer before processing
    std::fill(outputBuffer + firstFrame * outchannels, outputBuffer + lastFrame * outchannels, 0.0f);

    for (int sampleNo = firstFrame; sampleNo < lastFrame; sampleNo++)
    {
        for (int voiceNo = 0; voiceNo < polyphony; voiceNo++)
        {
            for (int channel = 0; channel < outchannels; channel++)
            {
                float thisSample = samples[channel][voiceNo][sampleNo];
                outputBuffer[sampleNo * outchannels + channel] += thisSample;
            }
        }
    }
}

void SampleCompute::HardStop()
{
    for (int voiceIndex = 0; voiceIndex < polyphony; voiceIndex++)
    {
        std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * threadCount / polyphony]);

        indexInEnvelope[strikeIndex] = 0;
        releaseVol[strikeIndex] = 1.0f;
        velocityVol[strikeIndex] = 0.0f;
    }
}

#define ELEMENTS_TO_PRINT 16
void SampleCompute::Dump(const char *filename)
{
    json output;

    // Store scalar values
    output["rhodesEffect"] = rhodesEffect;
    output["loop"] = loop;
    output["OVERVOLUME"] = masterVolume;

    output["lfoPhase"] = lfoPhase;
    output["lfoIncreasePerDispatch"] = lfoIncreasePerDispatch;
    output["dispatchFrameNo"] = voiceDispatchFrameNo;

    output["xfadeTracknot"] = xfadeTracknot;
    output["xfadeTrack"] = xfadeTrack;
    output["slaveFade"] = slaveFade;

    output["voiceDetune"] = voiceDetune;
    output["noLoopFade"] = noLoopFade;
    output["pitchWheel"] = pitchWheel;
    output["portamento"] = portamento;
    output["portamentoAlpha"] = portamentoAlpha;
    output["portamentoTarget"] = portamentoTarget;
    output["releaseVol"] = releaseVol;
    output["velocityVol"] = velocityVol;
    output["indexInEnvelope"] = indexInEnvelope;
    output["nextEnvelopeVol"] = nextEnvelopeVol;

    /*{
        std::lock_guard<std::mutex> lock(blobMutex);
        if (!binaryBlob.empty()) {
            output["binaryBlob"] = binaryBlob;
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

void SampleCompute::HardStop()
{
    for (int voiceIndex = 0; voiceIndex < polyphony; voiceIndex++)
    {
        std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * threadCount / polyphony]);

        indexInEnvelope[strikeIndex] = 0;
        releaseVol[strikeIndex] = 1.0f;
        velocityVol[strikeIndex] = 0.0f;
    }
}


// Correct destructor syntax
SampleCompute::~SampleCompute()
{
    std::cout << "Clean up audio and threads" << std::endl;

    // Clean up thread pool
    DestroyThreadPool();
}

void Test()
{
    std::cout << "Test mode activated" << std::endl;

    int samples_per_dispatch = 128;
    int outchannels = 2;
    int outSampleRate = 48000;
    SampleCompute * compute = new SampleCompute(10, samples_per_dispatch, 2, outchannels, 12, outSampleRate, 4);
    std::cout << "Loading JSON" << std::endl;
    Patch * patch = new Patch("Harp.json", compute);
    patch->DumpSampleInfo("sample_info.json"); // Only dump first buffer for debugging

    std::cout << "Loaded patch" << std::endl;
    std::cout << "Processing MIDI" << std::endl;

    unsigned char penta[] = {
        48, 50, 52, 55, 57};
    std::cout << "Running engine" << std::endl;

    // Calculate buffer sizes for 10 seconds of stereo audio
    const int secondsToGenerate = 10;
    const int samplesPerChannel = outSampleRate * secondsToGenerate;
    const int totalSamples = samplesPerChannel * 2; // Stereo output

    // Allocate buffer with proper size for stereo output
    std::vector<float> buffer(totalSamples);

    // Generate audio in chunks of samples_per_dispatch
    const int numBuffers = samplesPerChannel / samples_per_dispatch;

    for (int i = 0; i < numBuffers; i++)
    {
        // Compute the starting index in the buffer vector
        int startIndex = i * samples_per_dispatch * outchannels;
        // Send a MIDI note every 100 iterations
        if (i % 100 == 0)
        {
            std::vector<unsigned char> message = {0x90, penta[(i / 100) % 5], 127}; // Note on
            patch->ProcessMidi(&message);
        }
        compute->ProcessVoices(0, 1, buffer.data() + startIndex);
        compute->SumSamples(0, 1, buffer.data() + startIndex);

        // Dump first buffer for debugging
        if (i == 0)
            compute->Dump("dump.json");
    }

    WriteVectorToWav(buffer, "Outfile.wav", outchannels, outSampleRate);
    patch->ReleaseAll();

    std::vector<unsigned char> message = {0x90, 45, 127}; // Note on
    patch->ProcessMidi(&message);

    for (int i = 0; i < numBuffers; i++)
    {
        // Compute the starting index in the buffer vector
        int startIndex = i * samples_per_dispatch * outchannels;
        compute->ProcessVoices(0, 1, buffer.data() + startIndex);

        // increase the pitch
        float bendAmount = float(i) / numBuffers;
        int bendValue = static_cast<int>((bendAmount + 1.0) * 8192.0);
        bendValue = std::clamp(bendValue, 0, 16383);           // Ensure within valid range
        unsigned char lsb = bendValue & 0x7F;                  // Least Significant Byte
        unsigned char msb = (bendValue >> 7) & 0x7F;           // Most Significant Byte
        std::vector<unsigned char> message = {0xE0, lsb, msb}; // Pitch Bend on Channel 1
        // ProcessMidi(&message);
    }

    WriteVectorToWav(buffer, "Bend.wav", outchannels, outSampleRate);
}
