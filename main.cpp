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
#include "key2note.hpp"
#include "PianoKeyboard.hpp"
#include <cmath>
#include <memory>
#include <map>
#include <vector>
#include <mutex>
#include <fstream>

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

        LoadSoundJSON("Harp.json");

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
            ProcessMidi(it->second);
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
            ProcessMidi(note);
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
};

int audioCallback(void *outputBuffer, void * /*inputBuffer*/, unsigned int nBufferFrames,
                  double /*streamTime*/, RtAudioStreamStatus /*status*/, void *userData)
{
    auto *buffer = static_cast<float *>(outputBuffer);
    Run(0, buffer);
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
