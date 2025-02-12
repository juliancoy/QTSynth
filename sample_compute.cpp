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

ThreadData threadData[NUM_THREADS];
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

pthread_t threads[NUM_THREADS];
void *threadFunction(void *threadArg)
{
    ThreadData *data = (ThreadData *)threadArg;
    Run(data->threadNo);
    pthread_exit(NULL);
}

void RunMultithread()
{
    int rc;
    long t;
    int voicesPerThread = self.POLYPHONY / NUM_THREADS;

    for (t = 0; t < NUM_THREADS; t++)
    {
        threadData[t].sampleCompute = &self;
        threadData[t].threadNo = t;

        rc = pthread_create(&threads[t], NULL, threadFunction, (void *)&threadData[t]);
        if (rc)
        {
            std::cout << "ERROR; return code from pthread_create() is " << rc << std::endl;
            exit(-1);
        }
    }

    // Wait for all threads to complete
    for (t = 0; t < NUM_THREADS; t++)
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

void Init(int POLYPHONY)
{
    self.POLYPHONY = POLYPHONY;
    self.panning = 0;
    self.loop = 0;
    self.OVERVOLUME = 1.0 / (1 << 3);
    self.binaryBlob.clear(); // Initialize empty vector

    for (int voiceNo = 0; voiceNo < self.POLYPHONY; voiceNo++)
    {
        self.xfadeTracknot[voiceNo] = 1;
        self.portamento[voiceNo] = 1;
        self.portamentoAlpha[voiceNo] = 1;
        self.portamentoTarget[voiceNo] = 1;
        self.envelopeEnd[voiceNo] = (voiceNo + 1) * ENVLENPERPATCH - 1;
    }
}

using json = nlohmann::json;

std::vector<std::vector<int>> g_key2sampleNo(128);
std::vector<std::vector<float>> g_key2pitchBend(128);
std::vector<std::vector<int>> g_key2voiceIndex(128);

int currSample = 0;

// Function to set the pitch bend for a specific voice
void SetPitchBend(float bend, int index)
{
    if (index >= 0 && index < self.POLYPHONY)
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
    if (index >= 0 && index < self.POLYPHONY)
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
    return ENVLENPERPATCH;
}

// Function to append data to the binaryBlob array
int AppendSample(const float *npArray, int npArraySize)
{
    std::cout << "appending sample length " << npArraySize << std::endl;
    if (npArray == nullptr || npArraySize <= 0)
    {
        std::cout << "Invalid input array in AppendSample" << std::endl;
        return -1;
    }

    // Get the current size as the starting point for this sample
    int sample_start = self.binaryBlob.size();

    // Append the new samples to the vector
    self.binaryBlob.insert(self.binaryBlob.end(), npArray, npArray + npArraySize);

    // Store the Sample Details
    sampleStartPhase.push_back(sample_start);
    sampleLength.push_back(npArraySize);
    sampleEnd.push_back(sample_start + npArraySize);

    // TODO: Implement loop
    loopStart.push_back(sample_start);
    loopLength.push_back(npArraySize);
    loopEnd.push_back(sample_start + npArraySize);

    return (sample_start);
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

void Run(int threadNo, float *outputBuffer)
{
    float fadelen = 50000.0f;

    // Update LFO phases
    int lfoStart = threadNo * LFO_COUNT / NUM_THREADS;
    int lfoEnd = lfoStart + LFO_COUNT / NUM_THREADS;
    for (int lfoNo = lfoStart; lfoNo < lfoEnd; lfoNo++)
    {
        self.lfoPhase[lfoNo] += self.lfoIncreasePerDispatch[lfoNo];
        // Ensure LFO phase wraps around properly
        while (self.lfoPhase[lfoNo] >= 1.0f)
        {
            self.lfoPhase[lfoNo] -= 1.0f;
        }
    }

    int voiceStart = threadNo * self.POLYPHONY / NUM_THREADS;
    int voiceEnd = voiceStart + self.POLYPHONY / NUM_THREADS;
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
        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {

            // Update portamento for each voice, and for each sample
            self.portamento[voiceNo] = self.portamentoTarget[voiceNo] * self.portamentoAlpha[voiceNo] + (1.0f - self.portamentoAlpha[voiceNo]) * self.portamento[voiceNo];
            // Update the dispatch phase for the next cycle
            self.dispatchPhase[voiceNo] += self.voiceDetune[voiceNo] * self.portamento[voiceNo];

            float normalizedPosition = (float)sampleNo / (float)SAMPLES_PER_DISPATCH;
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
            float thisSample = self.binaryBlob[floorIndex];
            float nextSample = self.binaryBlob[ceilIndex];
            self.samples[voiceNo][sampleNo] = thisSample * (1.0f - fraction) + nextSample * fraction;
            self.samples[voiceNo][sampleNo] *= multiplier;

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
                self.samples[voiceNo][sampleNo] *= self.fadeOut[voiceNo][sampleNo];
            }
        }
        if (self.dispatchPhase[voiceNo] >= self.sampleEnd[voiceNo] && self.loop)
        {
            self.dispatchPhase[voiceNo] = fmodf(self.dispatchPhase[voiceNo] - self.loopStart[voiceNo], self.loopLength[voiceNo]) + self.loopStart[voiceNo];
        }
    }

    int sampleStart = threadNo * SAMPLES_PER_DISPATCH / NUM_THREADS;
    int sampleEnd = sampleStart + SAMPLES_PER_DISPATCH / NUM_THREADS;
    // Sum samples across self.POLYPHONY and apply panning
    for (int sampleNo = sampleStart; sampleNo < sampleEnd; sampleNo++)
    {
        float sum = 0.0f;
        for (int voiceNo = 0; voiceNo < self.POLYPHONY; voiceNo++)
        {
            sum += self.samples[voiceNo][sampleNo];
        }
        self.mono[sampleNo] = sampleNo;
        // self.mono[sampleNo] = sum * self.OVERVOLUME;
    }

    // Apply Panning
    if (self.panning)
    {
        float depth = 0.4f;
        float rhodes = sinf(2 * M_PI * fmodf(self.lfoPhase[0], 1.0f)) * depth;

        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {
            if (outputBuffer)
            {
                outputBuffer[sampleNo * 2] = self.mono[sampleNo] * ((1 - depth) + rhodes);
                outputBuffer[sampleNo * 2 + 1] = self.mono[sampleNo] * ((1 - depth) - rhodes);
            }
        }
    }
    else
    {
        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {
            if (outputBuffer)
            {
                outputBuffer[sampleNo * 2] = self.mono[sampleNo];
                outputBuffer[sampleNo * 2 + 1] = self.mono[sampleNo];
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

    self.sampleLen[strikeIndex] = self.sampleLen[sampleNo];
    self.sampleEnd[strikeIndex] = self.sampleEnd[sampleNo];
    self.loopLength[strikeIndex] = self.loopLength[sampleNo];
    self.loopStart[strikeIndex] = self.loopStart[sampleNo];
    self.loopEnd[strikeIndex] = self.loopEnd[sampleNo];

    // If no patch envelope is supplied, all 1
    if (patchEnvelope == nullptr)
    {
        for (int envIndex = 0; envIndex < ENVLENPERPATCH; envIndex++)
        {
            int envelopeIndex = strikeIndex * ENVLENPERPATCH + envIndex;
            self.combinedEnvelope[envelopeIndex] = 1;
        }
    }
    // Otherwise, load the patch env
    else
    {
        // Assuming 'envLenPerPatch' is the length of 'patchEnvelope'
        for (int envIndex = 0; envIndex < ENVLENPERPATCH; envIndex++)
        {
            int envelopeIndex = strikeIndex * ENVLENPERPATCH + envIndex;
            self.combinedEnvelope[envelopeIndex] = patchEnvelope[envIndex];
        }
    }

    self.releaseVol[strikeIndex] = 1;
    self.velocityVol[strikeIndex] = velocity;
    self.indexInEnvelope[strikeIndex] = strikeIndex * ENVLENPERPATCH;
    self.voiceDetune[strikeIndex] = voiceDetune;

    self.portamento[strikeIndex] = 1;
    self.portamentoAlpha[strikeIndex] = 1;
    self.portamentoTarget[strikeIndex] = 1;

    // implement Round Robin for simplicity
    strikeIndex = (strikeIndex + 1) % self.POLYPHONY;
    return (strikeIndex + self.POLYPHONY - 1) % self.POLYPHONY;
}

void Release(int voiceIndex, float *env)
{
    self.releaseVol[voiceIndex] = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceIndex])];

    for (int envPosition = 0; envPosition < ENVLENPERPATCH; envPosition++)
    {
        int index = voiceIndex * ENVLENPERPATCH + envPosition;
        self.combinedEnvelope[voiceIndex] = env[envPosition] * self.releaseVol[voiceIndex];
    }
    self.indexInEnvelope[voiceIndex] = voiceIndex * ENVLENPERPATCH;
}

void HardStop(int strikeIndex)
{
    self.indexInEnvelope[strikeIndex] = strikeIndex * ENVLENPERPATCH;
    self.releaseVol[strikeIndex] = 1.0f;
    self.velocityVol[strikeIndex] = 0.0f;
}

#define ELEMENTS_TO_PRINT 16
void Dump(const char *filename)
{
    json output;

    // Store scalar values
    output["panning"] = self.panning;
    output["loop"] = self.loop;
    output["OVERVOLUME"] = self.OVERVOLUME;

    // Helper function to truncate array to 128 elements
    auto truncateArray = [](const float *arr, size_t size)
    {
        size_t truncSize = std::min(size_t(128), size);
        return std::vector<float>(arr, arr + truncSize);
    };

    // Store array data (truncated to 128 elements)
    output["lfoPhase"] = truncateArray(self.lfoPhase, LFO_COUNT);
    output["lfoIncreasePerDispatch"] = truncateArray(self.lfoIncreasePerDispatch, LFO_COUNT);
    output["dispatchPhase"] = truncateArray(self.dispatchPhase, ELEMENTS_TO_PRINT);
    output["dispatchPhaseClipped"] = truncateArray(self.dispatchPhaseClipped, ELEMENTS_TO_PRINT);

    // Store voice-specific arrays (truncated to 128 elements)
    output["xfadeTracknot"] = truncateArray(self.xfadeTracknot, ELEMENTS_TO_PRINT);
    output["xfadeTrack"] = truncateArray(self.xfadeTrack, ELEMENTS_TO_PRINT);
    output["loopStart"] = truncateArray(self.loopStart, ELEMENTS_TO_PRINT);
    output["loopEnd"] = truncateArray(self.loopEnd, ELEMENTS_TO_PRINT);
    output["loopLength"] = truncateArray(self.loopLength, ELEMENTS_TO_PRINT);
    output["slaveFade"] = truncateArray(self.slaveFade, ELEMENTS_TO_PRINT);
    output["sampleLen"] = truncateArray(self.sampleLen, ELEMENTS_TO_PRINT);
    output["sampleEnd"] = truncateArray(self.sampleEnd, ELEMENTS_TO_PRINT);
    output["voiceDetune"] = truncateArray(self.voiceDetune, ELEMENTS_TO_PRINT);
    output["noLoopFade"] = truncateArray(self.noLoopFade, ELEMENTS_TO_PRINT);
    output["pitchBend"] = truncateArray(self.pitchBend, ELEMENTS_TO_PRINT);
    output["portamento"] = truncateArray(self.portamento, ELEMENTS_TO_PRINT);
    output["portamentoAlpha"] = truncateArray(self.portamentoAlpha, ELEMENTS_TO_PRINT);
    output["portamentoTarget"] = truncateArray(self.portamentoTarget, ELEMENTS_TO_PRINT);
    output["releaseVol"] = truncateArray(self.releaseVol, ELEMENTS_TO_PRINT);
    output["velocityVol"] = truncateArray(self.velocityVol, ELEMENTS_TO_PRINT);
    output["indexInEnvelope"] = truncateArray(self.indexInEnvelope, ELEMENTS_TO_PRINT);
    output["envelopeEnd"] = truncateArray(self.envelopeEnd, ELEMENTS_TO_PRINT);
    output["currEnvelopeVol"] = truncateArray(self.currEnvelopeVol, ELEMENTS_TO_PRINT);
    output["nextEnvelopeVol"] = truncateArray(self.nextEnvelopeVol, ELEMENTS_TO_PRINT);

    // Store combinedEnvelope (truncated to 128 elements)
    output["combinedEnvelope"] = truncateArray(self.combinedEnvelope, self.POLYPHONY * ENVLENPERPATCH);

    // Store binaryBlob data (truncated to 128 elements)
    if (!self.binaryBlob.empty())
    {
        output["binaryBlob"] = truncateArray(self.binaryBlob.data(), self.binaryBlob.size());
    }

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

// Decode a Base64 encoded audio sample. Load it to the enging. return its start address in the Binary Blob
int LoadRestAudioB64(const json &sample)
{
    // Create result structure
    SampleData result;

    // Decode Base64 to binary
    std::string binaryData = base64_decode(sample["audioData"].get<std::string>());

    // std::cout << "Format: " << sample["audioFormat"] << std::endl;
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
    }
    int sampleAddr = AppendSample(result.samples.data(), result.samples.size());

    currSample++;
    return currSample - 1;
}

void ProcessMidi(std::vector<unsigned char> *message)
{
    unsigned char status = message->at(0);
    unsigned char note = message->at(1);
    unsigned char velocity = message->at(2);

    // Note On
    if ((status & 0xF0) == 0x90 && velocity > 0)
    {
        const auto &sampleIndex = g_key2sampleNo[note];
        const auto &pitchBends = g_key2pitchBend[note];

        for (size_t i = 0; i < sampleIndex.size() && i < pitchBends.size(); i++)
        {
            g_key2voiceIndex[note].push_back(Strike(sampleIndex[i], velocity, pitchBends[i], nullptr));
        }
    }

    // Note Off
    else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0))
    {        const auto &sampleIndex = g_key2sampleNo[note];
        const auto &pitchBends = g_key2pitchBend[note];

        for (size_t i = 0; i < sampleIndex.size() && i < pitchBends.size(); ++i)
        {
            Release(sampleIndex[i], nullptr);
        }
    }
}

// Load samples from JSON file
void LoadSoundJSON(const std::string &filename)
{
    std::cout << "Loading " << filename << std::endl;

    std::ifstream f(filename);
    json data = json::parse(f);

    // Load samplesV
    for (const auto &instrument : data)
    {
        for (const auto &sample : instrument["samples"])
        {
            LoadRestAudioB64(sample);
        }
    }

    // Load key mappings
    for (const auto &mapping : data[0]["key2samples"])
    {
        int keyTrigger = mapping[0]["keyTrigger"];
        int sampleNo = mapping[0]["sampleNo"];
        float pitchBend = mapping[0]["pitchBend"];

        g_key2sampleNo[keyTrigger].push_back(sampleNo);
        g_key2pitchBend[keyTrigger].push_back(pitchBend);
    }
}

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData)
{
    auto *buffer = static_cast<float *>(outputBuffer);
    Run(0, buffer);
    return 0;
}

void Test()
{
    std::cout << "Test mode activated" << std::endl;
    std::cout << "Initializing with 4 polyphony" << std::endl;
    Init(4);
    std::cout << "Loading JSON" << std::endl;
    LoadSoundJSON("Harp.json");
    std::cout << "Loaded patch" << std::endl;
    std::cout << "Processing MIDI" << std::endl;

    std::vector<unsigned char> message = {0x90, 45, 127}; // Note on, middle A, velocity 127
    // Allocate buffer large enough for 10 seconds of audio
    int totalSamples = SAMPLE_RATE * 10 * NUM_CHANNELS;
    ProcessMidi(&message);
    std::cout << "Processed MIDI" << std::endl;
    std::cout << "Running engine" << std::endl;

    // Generate 10 seconds of audio
    int numBuffers = (SAMPLE_RATE * 10) / FRAMES_PER_BUFFER;
    float *buffer = new float[totalSamples]();
    for(int i = 0; i < numBuffers; i++){
    {
        Run(0, buffer + FRAMES_PER_BUFFER);
        Dump("dump.json");
        break;
    }

    std::cout << "Generated Buffer" << std::endl;
    std::cout << "Writing Buffer to file" << std::endl;

    // Write output to WAV file
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = NUM_CHANNELS;
    format.sampleRate = SAMPLE_RATE;
    format.bitsPerSample = 32;

    drwav wav;
    if (drwav_init_file_write(&wav, "Outfile.wav", &format, nullptr))
    {
        drwav_write_pcm_frames(&wav, totalSamples / NUM_CHANNELS, buffer);
        drwav_uninit(&wav);
        std::cout << "Wrote output to Outfile.wav" << std::endl;
    }
    else
    {
        std::cerr << "Failed to write WAV file" << std::endl;
    }

    // Clean up the buffer
    delete[] buffer;
}
