#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
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

// Print help information
void printHelp()
{
    std::cout << "Qt Sine Synth - Command Line Options:\n"
              << "  --test       Run in test mode\n"
              << "  --help, -h   Show this help message\n";
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

        auto layout = new QVBoxLayout(centralWidget);

        keyboard_ = new PianoKeyboard(this);
        layout->addWidget(keyboard_);

        // Enable keyboard focus
        setFocusPolicy(Qt::StrongFocus);

        std::cout << "Loading JSON" << std::endl;
        
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

        auto it = keyToNote_.find(event->key());
        if (it != keyToNote_.end())
        {
            keyboard_->keyPressed(it->second);
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
            Release(event->key(), nullptr);
            keyboard_->keyReleased(it->second);
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

    PianoKeyboard *keyboard_;
    std::unique_ptr<RtMidiIn> midiIn_;
    std::vector<std::unique_ptr<RtMidiIn>> midiInputs_;
};

int main(int argc, char *argv[])
{
    // Handle command line arguments
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
    }

    QApplication app(argc, argv);
    Init(128, 128, 16, 2056);
    InitAudio();
    
    std::cout << "Creating window" << std::endl;
    // Create and show the window
    SynthWindow window;
    window.show();

    std::cout << "Run the application" << std::endl;
    // Run the application
    int result = app.exec();
}
