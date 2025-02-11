// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#define M_PI 3.1415926

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sample_compute.hpp"

ThreadData threadData[NUM_THREADS];
SampleCompute self;

#include <pthread.h>
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
            printf("ERROR; return code from pthread_create() is %d\n", rc);
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

    for (int voiceNo = 0; voiceNo < POLYPHONY; voiceNo++)
    {
        self.xfadeTracknot[voiceNo] = 1;
        self.portamento[voiceNo] = 1;
        self.portamentoAlpha[voiceNo] = 1;
        self.portamentoTarget[voiceNo] = 1;
        self.envelopeEnd[voiceNo] = (voiceNo + 1) * ENVLENPERPATCH - 1;
    }
}

// Function to set the pitch bend for a specific voice
void SetPitchBend(float bend, int index)
{
    if (index >= 0 && index < POLYPHONY)
    {
        self.pitchBend[index] = bend;
    }
    else
    {
        printf("Index out of bounds in SetPitchBend\n");
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
        printf("Index out of bounds in UpdateDetune\n");
    }
}

// Function to get the envelope length per patch
int GetEnvLenPerPatch()
{
    return ENVLENPERPATCH;
}

// Function to append data to the binaryBlob array
int AppendSample(float *npArray, int npArraySize)
{
    // Check if there is enough space in the current allocation; if not, reallocate
    if (self.usedSize + npArraySize > self.binaryBlobSize)
    {
        int newSize = self.usedSize + npArraySize; // Calculate new size needed
        float *newBlob = static_cast<float *>(realloc(self.binaryBlob, newSize * sizeof(float)));
        if (newBlob == NULL)
        {
            printf("Failed to allocate memory in AppendSample\n");
            return;
        }
        self.binaryBlob = newBlob;
        self.binaryBlobSize = newSize; // Update the total allocated size
    }

    // Copy new data to the end of the used portion of binaryBlob
    memcpy(self.binaryBlob + (int)self.usedSize, npArray, npArraySize * sizeof(float));
    // Update the used size
    int sample_start = self.usedSize;
    self.usedSize += npArraySize;
    return(sample_start);
}

// Function to delete a portion of memory from binaryBlob
void DeleteMem(int startAddr, int endAddr)
{
    if (startAddr >= self.usedSize || endAddr > self.usedSize || startAddr > endAddr)
    {
        printf("Invalid start or end address in DeleteMem\n");
        return;
    }

    int deleteLength = endAddr - startAddr;
    // Move the part after endAddr forward to startAddr
    memmove(self.binaryBlob + startAddr, self.binaryBlob + endAddr, (self.usedSize - endAddr) * sizeof(float));
    // Update the used size
    self.usedSize -= deleteLength;
}

void ApplyPanning()
{
    if (self.panning)
    {
        float depth = 0.4f;
        float rhodes = sinf(2 * M_PI * fmodf(self.lfoPhase[0], 1.0f)) * depth;

        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {
            self.pa[sampleNo][0] = self.mono[sampleNo] * ((1 - depth) + rhodes);
            self.pa[sampleNo][1] = self.mono[sampleNo] * ((1 - depth) - rhodes);
        }
    }
    else
    {
        for (int sampleNo = 0; sampleNo < SAMPLES_PER_DISPATCH; sampleNo++)
        {
            self.pa[sampleNo][0] = self.mono[sampleNo];
            self.pa[sampleNo][1] = self.mono[sampleNo];
        }
    }
}

void Run(int threadNo)
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

    ApplyPanning();
}

void Strike(float sampleStartPhase, int sampleLength, int sampleEnd, int loopStart, int loopLength, int loopEnd, int voiceIndex, float voiceStrikeVolume, float voiceDetune, float patchPortamentoStart, float patchPortamentoAlpha, float patchPitchwheelReal, float *patchEnvelope)
{
    self.xfadeTrack[voiceIndex] = 0;
    self.xfadeTracknot[voiceIndex] = 1;
    self.dispatchPhase[voiceIndex] = sampleStartPhase;
    self.slaveFade[voiceIndex] = voiceIndex;
    self.noLoopFade[voiceIndex] = 1;

    self.sampleLen[voiceIndex] = sampleLength;
    self.sampleEnd[voiceIndex] = sampleEnd;
    self.loopLength[voiceIndex] = loopLength;
    self.loopStart[voiceIndex] = loopStart;
    self.loopEnd[voiceIndex] = loopEnd;

    self.portamento[voiceIndex] = patchPortamentoStart;
    // Assuming the 'msg.velocity' logic is handled elsewhere and 'portamento' logic is provided

    self.portamentoAlpha[voiceIndex] = patchPortamentoAlpha;
    self.pitchBend[voiceIndex] = patchPitchwheelReal;

    // Assuming 'envLenPerPatch' is the length of 'patchEnvelope'
    for (int envIndex = 0; envIndex < ENVLENPERPATCH; envIndex++)
    {
        int envelopeIndex = voiceIndex * ENVLENPERPATCH + envIndex;
        self.combinedEnvelope[envelopeIndex] = patchEnvelope[envIndex];
    }

    self.releaseVol[voiceIndex] = 1;
    self.velocityVol[voiceIndex] = voiceStrikeVolume;
    self.indexInEnvelope[voiceIndex] = voiceIndex * ENVLENPERPATCH;
    self.voiceDetune[voiceIndex] = voiceDetune;
}

void Release(int voiceIndex, float *env)
{
    self.releaseVol[voiceIndex] = self.combinedEnvelope[(int)(self.indexInEnvelope[voiceIndex])];

    for (int envPosition = 0; envPosition < ENVLENPERPATCH; envPosition++)
    {
        int index = voiceIndex * ENVLENPERPATCH + envPosition;
        self.combinedEnvelope[index] = env[envPosition] * self.releaseVol[voiceIndex];
    }

    self.indexInEnvelope[voiceIndex] = voiceIndex * ENVLENPERPATCH;
}

void HardStop(int voiceIndex)
{
    self.indexInEnvelope[voiceIndex] = voiceIndex * ENVLENPERPATCH;
    self.releaseVol[voiceIndex] = 1.0f;
    self.velocityVol[voiceIndex] = 0.0f;
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
