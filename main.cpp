#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QSlider>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>
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
#include <iostream>

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
        std::cout << "Initializing window" << std::endl;
        setWindowTitle("Qt Sine Synth");
        resize(600, 200);

        auto centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        auto layout = new QHBoxLayout(centralWidget);

        keyboard_ = new PianoKeyboard(this);
        layout->addWidget(keyboard_);

        // Create volume slider
        volumeSlider_ = new QSlider(Qt::Vertical, this);
        volumeSlider_->setMinimum(0);
        volumeSlider_->setMaximum(100);
        volumeSlider_->setValue(50); // Default volume at 50%
        volumeSlider_->setTickPosition(QSlider::TicksBothSides);
        volumeSlider_->setTickInterval(10);
        connect(volumeSlider_, &QSlider::valueChanged, this, &SynthWindow::onVolumeChanged);
        layout->addWidget(volumeSlider_);

        // Enable keyboard focus
        setFocusPolicy(Qt::StrongFocus);

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

        auto it = keyToNote_.find(event->key());
        if (it != keyToNote_.end())
        {
            keyboard_->showKeyDepressed(it->second);
            std::vector<unsigned char> message = {0x90, static_cast<unsigned char>(it->second), 127}; // Note on, note, velocity 127
            ProcessMidi(&message);
        }
    }

    void keyReleaseEvent(QKeyEvent *event) override
    {
        if (event->isAutoRepeat())
            return;

        auto it = keyToNote_.find(event->key());
        if (it != keyToNote_.end())
        {
            std::vector<unsigned char> message = {0x80, static_cast<unsigned char>(it->second), 0}; // Note off
            ProcessMidi(&message);
            keyboard_->showKeyReleased(it->second);
        }
    }

private:
    static void midiCallback(double timeStamp, std::vector<unsigned char> *message, void *userData)
    {
        auto *window = static_cast<SynthWindow *>(userData);
        // send it to the synth
        ProcessMidi(message);

        // update the window appearance as well
        unsigned char status = message->at(0);
        unsigned char note = message->at(1);
        unsigned char velocity = message->at(2);

        if ((status & 0xF0) == 0x90 && velocity > 0)
        { // Note On
            window->keyboard_->keyPressed(note);
        }
        else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0))
        { // Note Off
            window->keyboard_->keyReleased(note);
        }
    }

    void onVolumeChanged(int value) {
        float normalizedVolume = value / 100.0f;
        SetVolume(normalizedVolume);
    }

    PianoKeyboard *keyboard_;
    QSlider *volumeSlider_;
    std::unique_ptr<RtMidiIn> midiIn_;
    std::vector<std::unique_ptr<RtMidiIn>> midiInputs_;
    float currentVolume_ = 0.5f; // Track current volume
};

// Handle command line arguments
int polyphony = 64;
int samplesPerDispatch = 128;
int sampleRate = 44100;
int lfoCount = 16;
int envLenPerPatch = 512;
int outchannels = 2;
float bendDepth = 2.0f;
int bufferCount = 4;
int threadCount = 4;

// Print help information
void printHelp()
{
    std::cout << "Qt Sine Synth - Command Line Options:\n"
              << "  --test       Run in test mode\n"
              << "  --help, -h   Show this help message\n"
              << "  --polyphony <n>     Set polyphony (default: " << polyphony << ")\n"
              << "  --samples <n>       Set samples per dispatch (default: " << samplesPerDispatch << ")\n"
              << "  --samplerate <n>    Set samples rate (default: " << sampleRate << ")\n"
              << "  --lfo <n>           Set LFO count (default: " << lfoCount << ")\n"
              << "  --env <n>           Set envelope length per patch (default: " << envLenPerPatch << ")\n"
              << "  --channels <n>      Set output channels (default: " << outchannels << ")\n"
              << "  --bend <n>          Set pitch bend depth (default: " << bendDepth << ")\n"
              << "  --buffers <n>       Set audio buffer count (default: " << bufferCount << ")\n";
}

int main(int argc, char *argv[])
{

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--test")
        {
            Test();
            return 0;
        }
        else if (arg == "-h" || arg == "--help")
        {
            printHelp();
            return 0;
        }
        else if (arg == "--polyphony" && i + 1 < argc)
            polyphony = std::stoi(argv[++i]);
        else if (arg == "--samples" && i + 1 < argc)
            samplesPerDispatch = std::stoi(argv[++i]);
        else if (arg == "--samplerate" && i + 1 < argc)
            sampleRate = std::stoi(argv[++i]);
        else if (arg == "--lfo" && i + 1 < argc)
            lfoCount = std::stoi(argv[++i]);
        else if (arg == "--env" && i + 1 < argc)
            envLenPerPatch = std::stoi(argv[++i]);
        else if (arg == "--channels" && i + 1 < argc)
            outchannels = std::stoi(argv[++i]);
        else if (arg == "--bend" && i + 1 < argc)
            bendDepth = std::stof(argv[++i]);
        else if (arg == "--buffers" && i + 1 < argc)
            bufferCount = std::stoi(argv[++i]);
        else if (arg == "--threadcount" && i + 1 < argc)
            threadCount = std::stoi(argv[++i]);
    }

    QApplication app(argc, argv);
    Init(polyphony, samplesPerDispatch, lfoCount, envLenPerPatch, outchannels, bendDepth, sampleRate, threadCount);
    LoadSoundJSON("Harp.json");
    InitAudio(bufferCount);
    
    std::cout << "Creating window" << std::endl;
    // Create and show the window
    SynthWindow window;
    window.show();

    std::cout << "Run the application" << std::endl;
    // Run the application
    int result = app.exec();
    DeInitAudio();
}
