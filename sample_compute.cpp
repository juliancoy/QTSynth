// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#include "sample_compute.hpp"

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>

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
    int voicesPerThread = POLYPHONY / NUM_THREADS;

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

void Init()
{
    self.panning = 0;
    self.loop = 0;
    self.OVERVOLUME = 1.0 / (1 << 3);
    self.binaryBlobSize = 0;
    self.binaryBlob = NULL;  // Initialize to NULL
    self.usedSize = 0;      // Initialize usedSize

    for (int voiceNo = 0; voiceNo < POLYPHONY; voiceNo++)
    {
        self.xfadeTracknot[voiceNo] = 1;
        self.portamento[voiceNo] = 1;
        self.portamentoAlpha[voiceNo] = 1;
        self.portamentoTarget[voiceNo] = 1;
        self.envelopeEnd[voiceNo] = (voiceNo + 1) * ENVLENPERPATCH - 1;
    }
}

using json = nlohmann::json;

// Global map to store decoded samples
std::map<int, std::pair<int, float>> g_key2samples; // keyTrigger -> (sampleNo, pitchBend)
int currSample = 0;


// Function to set the pitch bend for a specific voice
void SetPitchBend(float bend, int index)
{
    if (index >= 0 && index < POLYPHONY)
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
    if (index >= 0 && index < POLYPHONY)
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
    if (npArray == NULL || npArraySize <= 0) {
        std::cout << "Invalid input array in AppendSample" << std::endl;
        return -1;
    }

    // Check if there is enough space in the current allocation; if not, reallocate
    if (self.usedSize + npArraySize > self.binaryBlobSize)
    {
        int newSize = self.usedSize + npArraySize; // Calculate new size needed
        float *newBlob;
        
        if (self.binaryBlob == NULL) {
            newBlob = static_cast<float *>(malloc(newSize * sizeof(float)));
        } else {
            newBlob = static_cast<float *>(realloc(self.binaryBlob, newSize * sizeof(float)));
        }
        
        if (newBlob == NULL)
        {
            std::cout << "Failed to allocate memory in AppendSample" << std::endl;
            return -1;
        }
        self.binaryBlob = newBlob;
        self.binaryBlobSize = newSize; // Update the total allocated size
    }

    // Copy new data to the end of the used portion of binaryBlob
    memcpy(self.binaryBlob + (int)self.usedSize, npArray, npArraySize * sizeof(float));
    // Update the used size
    int sample_start = self.usedSize;
    self.usedSize += npArraySize;

    // Store the Sample Details
    sampleStartPhase.push_back(sample_start);
    sampleLength.push_back(npArraySize);
    sampleEnd.push_back(sample_start + npArraySize);

    // TODO: Implement loop
    loopStart.push_back(sample_start);
    loopLength.push_back(npArraySize);
    loopEnd.push_back(sample_start + npArraySize);
    
    return(sample_start);
}

// Function to delete a portion of memory from binaryBlob
void DeleteMem(int startAddr, int endAddr)
{
    if (startAddr >= self.usedSize || endAddr > self.usedSize || startAddr > endAddr)
    {
        std::cout << "Invalid start or end address in DeleteMem" << std::endl;
        return;
    }

    int deleteLength = endAddr - startAddr;
    // Move the part after endAddr forward to startAddr
    memmove(self.binaryBlob + startAddr, self.binaryBlob + endAddr, (self.usedSize - endAddr) * sizeof(float));
    // Update the used size
    self.usedSize -= deleteLength;
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

    int voiceStart = threadNo * POLYPHONY / NUM_THREADS;
    int voiceEnd = voiceStart + POLYPHONY / NUM_THREADS;
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
            floorIndex = floorIndex < self.binaryBlobSize ? floorIndex : self.binaryBlobSize - 1;
            ceilIndex = ceilIndex < self.binaryBlobSize ? ceilIndex : self.binaryBlobSize - 1;

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
    // Sum samples across polyphony and apply panning
    for (int sampleNo = sampleStart; sampleNo < sampleEnd; sampleNo++)
    {
        float sum = 0.0f;
        for (int voiceNo = 0; voiceNo < POLYPHONY; voiceNo++)
        {
            sum += self.samples[voiceNo][sampleNo];
        }
        self.mono[sampleNo] = sum * self.OVERVOLUME;
    }

    // Apply Panning
    if (self.panning)
    {
        float depth = 0.4f;
        float rhodes = sinf(2 * M_PI * fmodf(self.lfoPhase[0], 1.0f)) * depth;

        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {
            if (outputBuffer) {
                outputBuffer[sampleNo * 2] = self.mono[sampleNo] * ((1 - depth) + rhodes);
                outputBuffer[sampleNo * 2 + 1] = self.mono[sampleNo] * ((1 - depth) - rhodes);
            }
        }
    }
    else
    {
        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {
            if (outputBuffer) {
                outputBuffer[sampleNo * 2] = self.mono[sampleNo];
                outputBuffer[sampleNo * 2 + 1] = self.mono[sampleNo];
            }
        }
    }
}

void Strike(int sampleNo, float voiceDetune, float *patchEnvelope)
{
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
    if(patchEnvelope == nullptr){
        for (int envIndex = 0; envIndex < ENVLENPERPATCH; envIndex++)
        {
            int envelopeIndex = strikeIndex * ENVLENPERPATCH + envIndex;
            self.combinedEnvelope[envelopeIndex] = 1;
        }
    }
    // Otherwise, load the patch env
    else{
        // Assuming 'envLenPerPatch' is the length of 'patchEnvelope'
        for (int envIndex = 0; envIndex < ENVLENPERPATCH; envIndex++)
        {
            int envelopeIndex = strikeIndex * ENVLENPERPATCH + envIndex;
            self.combinedEnvelope[envelopeIndex] = patchEnvelope[envIndex];
        }
    }

    self.releaseVol[strikeIndex] = 1;
    self.velocityVol[strikeIndex] = voiceStrikeVolume[sampleNo];
    self.indexInEnvelope[strikeIndex] = strikeIndex * ENVLENPERPATCH;
    self.voiceDetune[strikeIndex] = voiceDetune;

    self.portamento[strikeIndex] = 1;
    self.portamentoAlpha[strikeIndex] = 1;
    self.portamentoTarget[strikeIndex] = 1;
    
    // implement Round Robin for simplicity
    strikeIndex = (strikeIndex+1)%POLYPHONY;
}

void Release(int noteNo, float *env)
{
    int releaseIndex = 0;
    self.releaseVol[releaseIndex] = self.combinedEnvelope[(int)(self.indexInEnvelope[releaseIndex])];

    for (int envPosition = 0; envPosition < ENVLENPERPATCH; envPosition++)
    {
        int index = releaseIndex * ENVLENPERPATCH + envPosition;
        self.combinedEnvelope[releaseIndex] = env[envPosition] * self.releaseVol[releaseIndex];
    }
    self.indexInEnvelope[releaseIndex] = releaseIndex * ENVLENPERPATCH;
}

void HardStop(int strikeIndex)
{
    self.indexInEnvelope[strikeIndex] = strikeIndex * ENVLENPERPATCH;
    self.releaseVol[strikeIndex] = 1.0f;
    self.velocityVol[strikeIndex] = 0.0f;
}

void Dump(const char *filename)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        perror("Failed to open file");
        return;
    }

    // Write the struct except for the binaryBlob pointer
    size_t sizeWithoutBlob = sizeof(SampleCompute) - sizeof(float *) - sizeof(float) - sizeof(float);
    fwrite(&self, sizeWithoutBlob, 1, file);

    // Write the binaryBlobSize, usedSize and binaryBlob data
    fwrite(&self.binaryBlobSize, sizeof(float), 1, file);
    fwrite(&self.usedSize, sizeof(float), 1, file);
    if (self.binaryBlobSize > 0)
    {
        fwrite(self.binaryBlob, self.binaryBlobSize, 1, file);
    }

    fclose(file);
}


// Decode a Base64 encoded audio sample. Load it to the enging. return its start address in the Binary Blob
int LoadRestAudioB64(const json &sample)
{
    // Create result structure
    SampleData result;

    // Decode Base64 to binary
    std::string binaryData = base64_decode(sample["audioData"].get<std::string>());

    std::cout << "Format: " << sample["audioFormat"] << std::endl;
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

void ProcessMidi(int midiNote)
{
    // Find the mapping for this key
    auto it = g_key2samples.find(midiNote);
    if (it != g_key2samples.end()) {
        Strike(it->second.first, it->second.second, nullptr);
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

        g_key2samples[keyTrigger] = std::make_pair(sampleNo, pitchBend);
    }
}
