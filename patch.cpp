// gcc -shared -fPIC -o libsample_compute.so sample_compute.c

#include <algorithm>
#include <string>
#include <limits>

#include <fstream>
#include <vector>
#include <mutex>
#include "dr_wav.h"

#include "tuning.hpp"
#include "patch.hpp"
#define MIDI_KEY_COUNT 128

#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>

#define MIDI_NOTES 128
#define MIDI_CC_SUSTAIN 64
#define MIDI_CC_VOLUME 7
#define MIDI_CC_MODULATION 1
#define MIDI_CC_EXPRESSION 11

void WriteVectorToWav(std::vector<float> outvector, const std::string &filename, int channels, float sampleRate)
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
    format.sampleRate = sampleRate;    // Sample rate
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

using json = nlohmann::json;

int currSampleIndex = 0;
int currAbsoluteSampleNo = 0;
int startSample = 0;

// Function to append multichannel audio data
int Patch::AppendSample(std::vector<float> samples, float sampleRate, int inchannels, float baseFrequency)
{

    // Number of frames (samples per channel)
    size_t frameCount = samples.size() / inchannels;

    // Determine the mapping of input channels to output channels
    // For example if the input is Mono, it should distribute even power to both sides
    // the same amound of power that the left channel of a stereo signal puts out on one
    // 2D vector to store gains: gains[inchannel][outchannel]
    std::vector<std::vector<float>> volumeMatrix(inchannels, std::vector<float>(outchannels, 0.0f));

    for (size_t inchannel = 0; inchannel < inchannels; inchannel++)
    {
        // Calculate the angle of the current input channel
        float inChannelAngle = inchannels * M_PI / 2 + inchannel * 2.0f * M_PI / inchannels;

        float totalPower = 0.0f;

        for (size_t outchannel = 0; outchannel < outchannels; outchannel++)
        {
            // Calculate the angle of the current output channel
            float outChannelAngle = outchannels * M_PI / 2 + outchannel * 2.0f * M_PI / outchannels;

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
        for (size_t outchannel = 0; outchannel < outchannels; outchannel++)
        {
            volumeMatrix[inchannel][outchannel] *= normalizationFactor;
        }
    }

    std::lock_guard<std::mutex> lock(samplesMutex);

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
void Patch::DeleteSample(int sampleNo)
{
    if (sampleNo >= 0 && sampleNo < samplesData.size())
    {
        samplesData.erase(samplesData.begin() + sampleNo);
    }
}

int Patch::Strike(SampleData *sample, float velocity, float sampleDetune)
{
    std::lock_guard<std::mutex> lock(threadVoiceLocks[strikeIndex * threadCount / polyphony]);

    SampleCompute sc = *compute;
    // std::cout << "Striking sample " << sampleNo << " at strike index " << strikeIndex << std::endl;
    compute->xfadeTrack[strikeIndex] = 0;
    compute->xfadeTracknot[strikeIndex] = 1;
    compute->dispatchFrameNo[strikeIndex] = 0;
    compute->slaveFade[strikeIndex] = strikeIndex;
    compute->noLoopFade[strikeIndex] = 1;

    // Transfer Sample params to Voice params for sequential access in Run
    compute->voiceSamplePtr[strikeIndex] = sample;

    // If no patch envelope is supplied, all 1
    if (env == nullptr)
    {
        for (int envIndex = 0; envIndex < compute->envLenPerPatch; envIndex++)
        {
            int envelopeIndex = strikeIndex * compute->envLenPerPatch + envIndex;
            compute->combinedEnvelope[envelopeIndex] = 1;
        }
    }
    // Otherwise, load the patch env
    else
    {
        // Assuming 'envLenPerPatch' is the length of 'patchEnvelope'
        for (int envIndex = 0; envIndex < compute->envLenPerPatch; envIndex++)
        {
            int envelopeIndex = strikeIndex * compute->envLenPerPatch + envIndex;
            compute->combinedEnvelope[envelopeIndex] = compute->patchEnvelope[envIndex];
        }
    }

    // set additional voice init params
    releaseVol[strikeIndex] = 1;
    velocityVol[strikeIndex] = velocity / 255.0;
    indexInEnvelope[strikeIndex] = strikeIndex * envLenPerPatch;

    voiceDetune[strikeIndex] = sampleDetune;
    // std::cout << "Striking voice " << strikeIndex << " with detune " << voiceDetune[strikeIndex] << std::endl;

    portamento[strikeIndex] = 1;
    portamentoAlpha[strikeIndex] = 1;
    portamentoTarget[strikeIndex] = 1;

    // implement Round Robin for simplicity
    strikeIndex = (strikeIndex + 1) % polyphony;
    return (strikeIndex + polyphony - 1) % polyphony;
}

void Patch::ReleaseAll()
{
    for (int key = 0; key < MIDI_KEY_COUNT; key++)
        while (Release(key, env, nullptr));
}

int Patch::Release(int midi_key, float *env, Patch *patch)
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

        std::lock_guard<std::mutex> lock(threadVoiceLocks[releaseIndex * threadCount / polyphony]);

        patch->key2voiceIndex[midi_key].erase(patch->key2voiceIndex[midi_key].begin()); // Remove first element

        indexInEnvelope[releaseIndex] = releaseIndex * envLenPerPatch;
        // Apply envelope release if provided
        if (env != nullptr)
        {
            for (int envPosition = 0; envPosition < envLenPerPatch; envPosition++)
            {
                int index = releaseIndex * envLenPerPatch + envPosition;
                combinedEnvelope[index] = env[envPosition] * releaseVol[releaseIndex];
            }
        }
        else
        {
            // Default release (full volume fade out)

            for (int envIndex = 0; envIndex < envLenPerPatch; envIndex++)
            {
                int envelopeIndex = releaseIndex * envLenPerPatch + envIndex;
                combinedEnvelope[envelopeIndex] = 1;
            }
        }
        voicesReleased++;
    }
    return voicesToRelease;
}

void Patch::HardStop()
{
    for (int voiceIndex = 0; voiceIndex < polyphony; voiceIndex++)
    {
        std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * threadCount / polyphony]);

        indexInEnvelope[strikeIndex] = strikeIndex * envLenPerPatch;
        releaseVol[strikeIndex] = 1.0f;
        velocityVol[strikeIndex] = 0.0f;
    }
}

void Patch::DumpSampleInfo(const char *filename)
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

void SampleCompute()
{
    std::cout << "Clean up audio and threads" << std::endl;

    // Clean up thread pool
    DestroyThreadPool();
}

int Patch::LoadRestAudioB64(const json &sample)
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

void Patch::GenerateKeymap(double (*tuningSystemFn)(int), Patch *patch)
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
Patch::Patch(const std::string &filename, SampleCompute * compute)
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

void Patch::ProcessMidi(std::vector<unsigned char> *message, Patch *patch)
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
            Release(voiceIndex, nullptr, patch);
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
                        Release(voiceIndex, nullptr, patch);
                    }
                }
            }
            break;

        case MIDI_CC_VOLUME:
            masterVolume = cc_value;
            std::cout << "Master Volume: " << masterVolume << std::endl;
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
        for (int voiceIndex = 0; voiceIndex < polyphony; voiceIndex++)
        {
            std::lock_guard<std::mutex> lock(threadVoiceLocks[voiceIndex * threadCount / polyphony]);
            pitchWheel[voiceIndex] = frequencyMultiplier;
        }
    }
}