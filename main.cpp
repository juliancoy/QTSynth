#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>
#include <rtaudio/RtAudio.h>
#include <rtmidi/RtMidi.h>
#include "sample_compute.hpp"
#include "PianoKeyboard.hpp"
#include <nlohmann/json.hpp>
#include <cpp-base64/base64.h>
#define DR_WAV_IMPLEMENTATION
#include <dr_wav.h>
#define DR_MP3_IMPLEMENTATION
#include <dr_mp3.h>
#include <cmath>
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <fstream>

using json = nlohmann::json;

// Structure to hold decoded sample data
struct SampleData
{
    std::vector<float> samples;
    int sampleRate;
    int channels;
};

// Global map to store decoded samples
std::map<int, SampleData> sampleNo2addy;
std::vector<std::tuple<int, int, float>> g_key2samples; // keyTrigger, sampleNo, pitchBend
int currSample = 0;

// Decode a Base64 encoded audio sample. Load it to the enging. return its start address in the Binary Blob
int loadRestAudioB64(const json &sample)
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
    sampleNo2addy[currSample] = result;
    currSample++;
    return currSample - 1;
}

// Load samples from JSON file
void loadSoundJSON(const std::string &filename)
{
    std::cout << "Loading " << filename << std::endl;

    std::ifstream f(filename);
    json data = json::parse(f);

    // Load samplesV
    for (const auto &instrument : data)
    {
        for (const auto &sample : instrument["samples"])
        {
            loadRestAudioB64(sample);
        }
    }

    // Load key mappings
    for (const auto &mapping : data[0]["key2samples"])
    {
        int keyTrigger = mapping[0]["keyTrigger"];
        int sampleNo = mapping[0]["sampleNo"];
        float pitchBend = mapping[0]["pitchBend"];

        g_key2samples.emplace_back(keyTrigger, sampleNo, pitchBend);
    }
}

void KeyStrike(int midiNote)
{
    if (midiNote >= 0 && midiNote < 128)
    {
        // Find the mapping for this key
        for (const auto &[keyTrigger, sampleNo, pitchBend] : g_key2samples)
        {
            if (keyTrigger == midiNote && sampleNo2addy.count(sampleNo) > 0)
            {
                const auto &sampleAddy = sampleNo2addy[sampleNo];
                int sampleLength = sample.samples.size() / sample.channels;

                Strike(0.0f, sampleLength, sampleLength, 0, 0, 0,
                       sampleAddy, 1.0f, pitchBend, 0.0f, 0.0f, 1.0f, nullptr);
                break;
            }
        }
    }
}

const double PI = 3.14159265358979323846;
const int SAMPLE_RATE = 44100;
// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency
unsigned int FRAMES_PER_BUFFER = 64;
const int NUM_CHANNELS = 1;

// Convert MIDI note to frequency
double midiNoteToFreq(int note)
{
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}


class SynthWindow : public QMainWindow
{
public:
    SynthWindow()
    {
        setWindowTitle("Qt Sine Synth");
        resize(600, 200);

        auto centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        auto layout = new QVBoxLayout(centralWidget);

        keyboard_ = new PianoKeyboard(this);
        layout->addWidget(keyboard_);

        // Enable keyboard focus
        setFocusPolicy(Qt::StrongFocus);

        loadSoundJSON("Harp.json");

        // Initialize key mapping (QWERTY layout mapped to musical notes)
        // Starting from middle C (60) on the home row
        keyToNote_ = {
            // Numbers row (higher octave)
            {Qt::Key_1, 72},     // C5
            {Qt::Key_2, 74},     // D5
            {Qt::Key_3, 76},     // E5
            {Qt::Key_4, 77},     // F5
            {Qt::Key_5, 79},     // G5
            {Qt::Key_6, 81},     // A5
            {Qt::Key_7, 83},     // B5
            {Qt::Key_8, 84},     // C6
            {Qt::Key_9, 86},     // D6
            {Qt::Key_0, 88},     // E6
            {Qt::Key_Minus, 89}, // F6
            {Qt::Key_Equal, 91}, // G6

            // Top letter row
            {Qt::Key_Q, 67},            // G4
            {Qt::Key_W, 69},            // A4
            {Qt::Key_E, 71},            // B4
            {Qt::Key_R, 72},            // C5
            {Qt::Key_T, 74},            // D5
            {Qt::Key_Y, 76},            // E5
            {Qt::Key_U, 77},            // F5
            {Qt::Key_I, 79},            // G5
            {Qt::Key_O, 81},            // A5
            {Qt::Key_P, 83},            // B5
            {Qt::Key_BracketLeft, 84},  // C6
            {Qt::Key_BracketRight, 86}, // D6

            // Home row (middle)
            {Qt::Key_A, 60},          // Middle C (C4)
            {Qt::Key_S, 62},          // D4
            {Qt::Key_D, 64},          // E4
            {Qt::Key_F, 65},          // F4
            {Qt::Key_G, 67},          // G4
            {Qt::Key_H, 69},          // A4
            {Qt::Key_J, 71},          // B4
            {Qt::Key_K, 72},          // C5
            {Qt::Key_L, 74},          // D5
            {Qt::Key_Semicolon, 76},  // E5
            {Qt::Key_Apostrophe, 77}, // F5

            // Bottom letter row
            {Qt::Key_Z, 48},      // C3
            {Qt::Key_X, 50},      // D3
            {Qt::Key_C, 52},      // E3
            {Qt::Key_V, 53},      // F3
            {Qt::Key_B, 55},      // G3
            {Qt::Key_N, 57},      // A3
            {Qt::Key_M, 59},      // B3
            {Qt::Key_Comma, 60},  // C4
            {Qt::Key_Period, 62}, // D4
            {Qt::Key_Slash, 64},  // E4
        };

        // Initialize MIDI
        try
        {
            midiIn_ = std::make_unique<RtMidiIn>();

            // Open all available ports
            unsigned int nPorts = midiIn_->getPortCount();
            for (unsigned int i = 0; i < nPorts; i++)
            {
                try
                {
                    // Create a new MIDI input for each port
                    auto midiIn = std::make_unique<RtMidiIn>();
                    midiIn->openPort(i);
                    midiIn->setCallback(&SynthWindow::midiCallback, this);
                    midiIn->ignoreTypes(false, false, false);
                    midiInputs_.push_back(std::move(midiIn));

                    std::cout << "Opened MIDI port " << i << ": "
                              << midiIn_->getPortName(i) << std::endl;
                }
                catch (RtMidiError &error)
                {
                    error.printMessage();
                }
            }
        }
        catch (RtMidiError &error)
        {
            error.printMessage();
        }
    }

    ~SynthWindow()
    {
        for (auto &midiIn : midiInputs_)
        {
            midiIn->closePort();
        }
    }

protected:
    void keyPressEvent(QKeyEvent *event) override
    {
        if (event->isAutoRepeat())
            return;

        if (auto it = keyToNote_.find(event->key()); it != keyToNote_.end())
        {
            keyboard_->keyPressed(it->second);
            KeyStrike(it->second);
        }
    }

    void keyReleaseEvent(QKeyEvent *event) override
    {
        if (event->isAutoRepeat())
            return;

        if (auto it = keyToNote_.find(event->key()); it != keyToNote_.end())
        {
            Release(event->key(), nullptr);
            keyboard_->keyReleased(it->second);
        }
    }

private:
    static void midiCallback(double timeStamp, std::vector<unsigned char> *message, void *userData)
    {
        auto *window = static_cast<SynthWindow *>(userData);
        if (message->size() < 3)
            return;

        unsigned char status = message->at(0);
        unsigned char note = message->at(1);
        unsigned char velocity = message->at(2);

        if ((status & 0xF0) == 0x90 && velocity > 0)
        { // Note On
            KeyStrike(note);
            window->keyboard_->keyPressed(note);
        }
        else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0))
        { // Note Off
            Release(note, nullptr);
            window->keyboard_->keyReleased(note);
        }
    }

    PianoKeyboard *keyboard_;
    std::unique_ptr<RtMidiIn> midiIn_;
    std::vector<std::unique_ptr<RtMidiIn>> midiInputs_;
    std::map<int, int> keyToNote_; // Maps Qt key codes to MIDI note numbers
};

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData)
{
    auto *buffer = static_cast<float *>(outputBuffer);
    Run(0);
    return 0;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    RtAudio dac(RtAudio::LINUX_PULSE);
    unsigned int devices = dac.getDeviceCount();
    if (devices < 1)
    {
        std::cerr << "No audio devices found!" << std::endl;
        return -1;
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
        return -1;
    }

    // Create and show the window
    SynthWindow window;
    window.show();

    // Run the application
    int result = app.exec();

    // Clean up audio
    if (dac.isStreamOpen())
    {
        dac.stopStream();
        dac.closeStream();
    }
}
