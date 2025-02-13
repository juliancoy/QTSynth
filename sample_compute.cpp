// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#include "sample_compute.hpp"
#include <rtaudio/RtAudio.h>

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>

const double PI = 3.14159265358979323846;
const int SAMPLE_RATE = 44100;
// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency
const int NUM_CHANNELS = 1;

unsigned int FRAMES_PER_BUFFER = 64;

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData);

// ThreadData threadData[numThreads];
SampleCompute self;

// Extensable arrays for sample state
std::vector<float> sampleStartPhase;
std::vector<int> sampleLength;
std::vector<int> sampleEnd;
std::vector<int> loopStart;
std::vector<int> loopLength;
std::vector<int> loopEnd;
std::vector<float> voiceStrikeVolume;
std::vector<std::vector<float>> patchEnvelope; // Nested vector for envelopes

int strikeIndex = 0;

void *threadFunction(void *threadArg)
{
    ThreadData *data = (ThreadData *)threadArg;

    float outputBuffer[MAX_SAMPLES_PER_DISPATCH];
    Run(data->threadNo, data->threadCount, outputBuffer);
    pthread_exit(NULL);
}

#include <fstream>
#include <vector>
#include <mutex>
#include "dr_wav.h"

void WriteVectorToWav(std::vector<float> outvector, const std::string &filename)
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
    format.channels = NUM_CHANNELS;            // Mono or stereo
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
    drwav_uint64 framesWritten = drwav_write_pcm_frames(&wav, outvector.size(), outvector.data());
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

void InitAudio()
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
    parameters.nChannels = NUM_CHANNELS;
    parameters.firstChannel = 0;

    std::cout << "Opening output stream" << std::endl;
    // Open the stream with minimal buffering for low latency
    try
    {
        RtAudio::StreamOptions options;
        options.numberOfBuffers = 2;              // Minimum number of buffers for stable playback
        options.flags = RTAUDIO_MINIMIZE_LATENCY; // Request minimum latency

        dac.openStream(&parameters, nullptr, RTAUDIO_FLOAT32,
                       SAMPLE_RATE, &FRAMES_PER_BUFFER, &audioCallback,
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

void Init(int polyphony, int samplesPerDispatch, int lfoCount, int envLenPerPatch)
{
    self.samplesPerDispatch = samplesPerDispatch;
    self.polyphony = polyphony;
    self.rhodesEffect = 0;
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
        self.envelopeEnd[voiceNo] = (voiceNo + 1) * ENVELOPE_LENGTH - 1;
    }

    self.key2sampleIndex.resize(MIDI_KEY_COUNT);
    self.key2sampleDetune.resize(MIDI_KEY_COUNT);
    self.key2voiceIndex.resize(MIDI_KEY_COUNT);
    self.sampleIndex2ChannelVol.resize();

    self.xfadeTracknot.resize(polyphony, 1.0f);
    self.xfadeTrack.resize(polyphony, 0.0f);

    self.loopStart.resize(polyphony, 0.0f);
    self.loopEnd.resize(polyphony, 0.0f);
    self.loopLength.resize(polyphony, 0.0f);
    self.slaveFade.resize(polyphony, 0.0f);
    self.sampleLen.resize(polyphony, 0.0f);
    self.sampleEnd.resize(polyphony, 0.0f);
    self.voiceDetune.resize(polyphony, 0.0f);
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
    return ENVELOPE_LENGTH;
}

// Function to append data to the binaryBlob array
int AppendSample(std::vector<float> sample_array)
{
    // Transform patchSampleNo to sampleIndex
    // As some samples are 2 channels (or more!) wide
    std::cout << " Appending Sample " << std::endl;
    for (const auto &keyAction : self.key2samples)
    {
        std::cout << keyAction << std::endl;

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
    int sample_start = self.binaryBlob.size();

    // Append the new samples to the vector
    self.binaryBlob.insert(self.binaryBlob.end(), sample_array.begin(), sample_array.end());

    // If binaryBlob is empty, write this sample to a WAV file using dr_wav
    /*if (self.binaryBlob.empty())
    {
        WriteVectorToWav(self.binaryBlob, "binaryBlob.wav");
        WriteVectorToWav(sample_array, "sample.wav");
    }*/

    // Store the Sample Details
    sampleStartPhase.push_back(sample_start);
    sampleLength.push_back(sample_array.size());
    sampleEnd.push_back(sample_start + sample_array.size());

    // TODO: Implement loop
    loopStart.push_back(sample_start);
    loopLength.push_back(sample_array.size());
    loopEnd.push_back(sample_start + sample_array.size());

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
    for (int channel = 0; channel < self.outchannels; channel++){
        if (self.rhodesEffectOn)
            self.rhodesEffect[channel] = ((1 - depth) + sinf(2 * M_PI * fmodf(self.lfoPhase[0] + 1.0/self.outchannels, 1.0f)) * depth);
        else
            self.rhodesEffect[channel] = 1;
    }

    int voiceStart = threadNo * self.polyphony / numThreads;
    int voiceEnd = voiceStart + self.polyphony / numThreads;
    // Process each voice
    for (int voiceNo = voiceStart; voiceNo < voiceEnd; voiceNo++)
    {

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
            /*std::cout << "   Sample " << sampleNo << std::endl;
            std::cout << "   Voice Detune " << self.voiceDetune[voiceNo] << std::endl;
            std::cout << "   portamento " << self.portamento[voiceNo] << std::endl;*/

            // Update portamento for each voice, and for each sample
            self.portamento[voiceNo] = self.portamentoTarget[voiceNo] * self.portamentoAlpha[voiceNo] + (1.0f - self.portamentoAlpha[voiceNo]) * self.portamento[voiceNo];
            // Update the dispatch phase for the next cycle
            self.dispatchPhase[voiceNo] += self.voiceDetune[voiceNo] * self.portamento[voiceNo];

            float normalizedPosition = (float)sampleNo / (float)self.samplesPerDispatch;
            float multiplier = difference * normalizedPosition + thisEnvelopeVol;

            // Clip the phase to valid sample indices and loop if necessary
            if (self.dispatchPhase[voiceNo] >= self.sampleEnd[voiceNo])
            {
                if (self.loop)
                {
                    self.dispatchPhase[voiceNo] = fmodf(self.dispatchPhase[voiceNo] - self.loopStart[voiceNo], self.loopLength[voiceNo]) + self.loopStart[voiceNo];
                }
                else
                {
                    self.dispatchPhase[voiceNo] = self.sampleEnd[voiceNo] - 1;
                }
            }

            // Calculate floor and ceiling indices for interpolation
            int floorIndex = (int)floorf(self.dispatchPhase[voiceNo]);
            int ceilIndex = floorIndex + 1;
            if (ceilIndex >= self.sampleEnd[voiceNo])
            {
                ceilIndex = self.loop ? self.loopStart[voiceNo] : self.sampleEnd[voiceNo] - 1;
            }

            // Ensure indices are within bounds
            floorIndex = floorIndex < self.binaryBlob.size() ? floorIndex : self.binaryBlob.size() - 1;
            ceilIndex = ceilIndex < self.binaryBlob.size() ? ceilIndex : self.binaryBlob.size() - 1;

            // Perform linear interpolation between the two samples
            float fraction = self.dispatchPhase[voiceNo] - floorIndex;
            std::lock_guard<std::mutex> lock(self.blobMutex);
            floorIndex = std::min(floorIndex, static_cast<int>(self.binaryBlob.size() - 1));
            ceilIndex = std::min(ceilIndex, static_cast<int>(self.binaryBlob.size() - 1));
            thisSample = self.binaryBlob[floorIndex];
            nextSample = self.binaryBlob[ceilIndex];

            for (int channel = 0; channel < self.outchannels; channel++)
            {
                self.samples[channel][voiceNo][sampleNo] = (thisSample * (1.0f - fraction) + nextSample * fraction) * self.OVERVOLUME * self.channelVol[voiceNo][channel] * self.rhodesEffect[channel];
            }
            // Apply fade out if needed
            if (0)
            {
                // Fade out around the loop points
                self.fadeOut[voiceNo][sampleNo] = fminf(fabsf(self.outputPhaseFloor[voiceNo][sampleNo] - (self.loopEnd[voiceNo] - self.loopLength[voiceNo] + self.slaveFade[voiceNo])) / fadelen, 1.0f);
                self.fadeOut[voiceNo][sampleNo] = fminf(fabsf(self.outputPhaseFloor[voiceNo][sampleNo] - (self.loopEnd[voiceNo] + self.slaveFade[voiceNo])) / fadelen, self.fadeOut[voiceNo][sampleNo]);

                // Applying fade out logic based on slaveFade
                // Assuming slaveFade is an index, not an array, as it is not clear from the Python code
                self.fadeOut[voiceNo][sampleNo] = self.xfadeTracknot[voiceNo] * self.fadeOut[voiceNo][sampleNo] + self.xfadeTrack[voiceNo] * (1.0f - self.fadeOut[voiceNo][sampleNo]);

                // Ensure fadeOut is not less than noLoopFade
                self.fadeOut[voiceNo][sampleNo] = fmaxf(self.fadeOut[voiceNo][sampleNo], self.noLoopFade[voiceNo]);

                // Apply fadeOut to samples
                self.samples[0][voiceNo][sampleNo] *= self.fadeOut[voiceNo][sampleNo];
            }
        }
        if (self.dispatchPhase[voiceNo] >= self.sampleEnd[voiceNo] && self.loop)
        {
            self.dispatchPhase[voiceNo] = fmodf(self.dispatchPhase[voiceNo] - self.loopStart[voiceNo], self.loopLength[voiceNo]) + self.loopStart[voiceNo];
        }
    }

    int sampleStart = threadNo * self.samplesPerDispatch / numThreads;
    int sampleEnd = sampleStart + self.samplesPerDispatch / numThreads;

    // Sum samples across polyphony and interleave channels
    for (int sampleNo = sampleStart; sampleNo < sampleEnd; sampleNo++)
    {
        for (int voiceNo = 0; voiceNo < self.polyphony; voiceNo++)
        {
            for (int channel = 0; channel < self.outchannels; channel++){
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

    self.sampleLen[strikeIndex] = sampleLength[sampleNo];
    self.sampleEnd[strikeIndex] = sampleEnd[sampleNo];
    self.loopLength[strikeIndex] = loopLength[sampleNo];
    self.loopStart[strikeIndex] = loopStart[sampleNo];
    self.loopEnd[strikeIndex] = loopEnd[sampleNo];

    // If no patch envelope is supplied, all 1
    if (patchEnvelope == nullptr)
    {
        for (int envIndex = 0; envIndex < ENVELOPE_LENGTH; envIndex++)
        {
            int envelopeIndex = strikeIndex * ENVELOPE_LENGTH + envIndex;
            self.combinedEnvelope[envelopeIndex] = 1;
        }
    }
    // Otherwise, load the patch env
    else
    {
        // Assuming 'envLenPerPatch' is the length of 'patchEnvelope'
        for (int envIndex = 0; envIndex < ENVELOPE_LENGTH; envIndex++)
        {
            int envelopeIndex = strikeIndex * ENVELOPE_LENGTH + envIndex;
            self.combinedEnvelope[envelopeIndex] = patchEnvelope[envIndex];
        }
    }

    self.releaseVol[strikeIndex] = 1;
    self.velocityVol[strikeIndex] = velocity / 255.0;
    self.indexInEnvelope[strikeIndex] = strikeIndex * ENVELOPE_LENGTH;
    self.voiceDetune[strikeIndex] = voiceDetune;

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

    for (int envPosition = 0; envPosition < ENVELOPE_LENGTH; envPosition++)
    {
        int index = voiceIndex * ENVELOPE_LENGTH + envPosition;
        self.combinedEnvelope[voiceIndex] = env[envPosition] * self.releaseVol[voiceIndex];
    }
    self.indexInEnvelope[voiceIndex] = voiceIndex * ENVELOPE_LENGTH;
}

void HardStop(int strikeIndex)
{
    self.indexInEnvelope[strikeIndex] = strikeIndex * ENVELOPE_LENGTH;
    self.releaseVol[strikeIndex] = 1.0f;
    self.velocityVol[strikeIndex] = 0.0f;
}

#define ELEMENTS_TO_PRINT 16
void Dump(const char *filename)
{
    json output;

    // Store scalar values
    output["panning"] = self.rhodesEffect;
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
    output["sampleLen"] = self.sampleLen;
    output["sampleEnd"] = self.sampleEnd;
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
    SampleData result;

    // Decode Base64 to binary
    std::string binaryData = base64_decode(sample["audioData"].get<std::string>());

    if (sample["audioFormat"] == "wav")
    {
        drwav wav;
        if (drwav_init_memory(&wav, binaryData.data(), binaryData.size(), nullptr))
        {
            result.samples.resize(wav.totalPCMFrameCount * wav.channels);
            drwav_read_pcm_frames_f32(&wav, wav.totalPCMFrameCount, result.samples.data());
            result.sampleRate = wav.sampleRate;
            result.channels = wav.channels;
            drwav_uninit(&wav);
        }
    }
    else if (sample["audioFormat"] == "mp3")
    {
        drmp3 mp3;
        if (drmp3_init_memory(&mp3, binaryData.data(), binaryData.size(), nullptr))
        {
            drmp3_uint64 frameCount = drmp3_get_pcm_frame_count(&mp3);
            result.samples.resize(frameCount * mp3.channels);
            drmp3_read_pcm_frames_f32(&mp3, frameCount, result.samples.data());
            result.sampleRate = mp3.sampleRate;
            result.channels = mp3.channels;
            drmp3_uninit(&mp3);
        }
    }
    else
    {
        std::cout << "Unknown format" << std::endl;
        return -1;
    }

    // Number of frames (samples per channel)
    size_t frameCount = result.samples.size() / result.channels;

    // Separate buffers for each channel
    std::vector<std::vector<float>> channelBuffers(result.channels, std::vector<float>(frameCount));

    // Deinterleave the samples
    for (size_t i = 0; i < frameCount; i++)
    {
        for (int ch = 0; ch < result.channels; ch++)
        {
            channelBuffers[ch][i] = result.samples[i * result.channels + ch];
        }
    }

    // Append each channel separately
    for (int i = 0; i < result.channels; i++)
    {
        AppendSample(channelBuffers[i]);
    }
    currPatchSampleNo++;

    return 0;
}

void ProcessMidi(std::vector<unsigned char> *message)
{
    unsigned char status = message->at(0);
    unsigned char midi_key = message->at(1);
    unsigned char velocity = message->at(2);

    // Note On
    if ((status & 0xF0) == 0x90 && velocity > 0)
    {
        const auto &sampleIndex = self.key2sampleIndex[midi_key];
        const auto &sampleDetune = self.key2sampleDetune[midi_key];
        std::cout << "Sample Index" << sampleIndex[0] << ", " << sampleDetune.size() << std::endl;
        for (size_t i = 0; i < sampleIndex.size() && i < sampleDetune.size(); i++)
        {
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

// Load samples from JSON file
void LoadSoundJSON(const std::string &filename)
{
    std::cout << "Loading " << filename << std::endl;

    std::ifstream f(filename);
    json data = json::parse(f);

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
    std::cout << "Initializing with 4 polyphony" << std::endl;

    int samples_per_dispatch = FRAMES_PER_BUFFER;
    Init(2, samples_per_dispatch, 2, 512);
    std::cout << "Loading JSON" << std::endl;
    LoadSoundJSON("Harp.json");
    DumpSampleInfo("sample_info.json"); // Only dump first buffer for debugging

    std::cout << "Loaded patch" << std::endl;
    std::cout << "Processing MIDI" << std::endl;

    std::vector<unsigned char> message = {0x90, 45, 127}; // Note on, middle A, velocity 127
    ProcessMidi(&message);
    std::cout << "Processed MIDI" << std::endl;
    std::cout << "Running engine" << std::endl;

    // Calculate buffer sizes for 10 seconds of stereo audio
    const int secondsToGenerate = 5;
    const int samplesPerChannel = SAMPLE_RATE * secondsToGenerate;
    const int totalSamples = samplesPerChannel * 2; // Stereo output

    // Allocate buffer with proper size for stereo output
    float *buffer = new float[totalSamples]();

    // Generate audio in chunks of FRAMES_PER_BUFFER
    const int numBuffers = samplesPerChannel / samples_per_dispatch;
    float *currentBufferPos = buffer;

    for (int i = 0; i < numBuffers; i++)
    {
        Run(0, 1, currentBufferPos);
        currentBufferPos += samples_per_dispatch * 2; // Move pointer by frames * channels

        if (i == 0)
        {
            // Dump("dump.json"); // Only dump first buffer for debugging
        }
    }

    std::cout << "Generated Buffer" << std::endl;
    std::cout << "Writing Buffer to file" << std::endl;

    // Write output to WAV file
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = 2; // Stereo output
    format.sampleRate = SAMPLE_RATE;
    format.bitsPerSample = 32;

    drwav wav;
    if (drwav_init_file_write(&wav, "Outfile.wav", &format, nullptr))
    {
        drwav_write_pcm_frames(&wav, samplesPerChannel, buffer);
        drwav_uninit(&wav);
        std::cout << "Wrote output to Outfile.wav" << std::endl;
    }
    else
    {
        std::cerr << "Failed to write WAV file" << std::endl;
    }
    delete[] buffer;
}