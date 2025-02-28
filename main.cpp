#include "main.hpp"

SynthWindow::SynthWindow()
{
    std::cout << "Initializing window" << std::endl;
    setWindowTitle("Qt Sine Synth");
    resize(600, 200);

    midiIn_ = new RtMidiIn();

    auto centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    auto mainLayout = new QVBoxLayout(centralWidget);
    auto controlLayout = new QHBoxLayout();
    auto keyboardLayout = new QHBoxLayout();

    // Create tuning system selector
    tuningSelector_ = new QComboBox(this);
    tuningSelector_->addItem("12-TET", 0);
    tuningSelector_->addItem("Rast راست", 1);
    tuningSelector_->addItem("Pythagorean", 2);
    tuningSelector_->addItem("Raga Yaman", 3);
    tuningSelector_->addItem("Bohlen-Pierce", 4);
    tuningSelector_->addItem("Bayati بياتي", 5);
    tuningSelector_->addItem("Slendro-Pelog", 6);
    tuningSelector_->addItem("Harmonic Series", 7);
    connect(tuningSelector_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &SynthWindow::onTuningChanged);
    controlLayout->addWidget(tuningSelector_);
    controlLayout->addStretch();

    keyboard_ = new PianoKeyboard(this);
    keyboardLayout->addWidget(keyboard_);

    // Create volume slider
    volumeSlider_ = new QSlider(Qt::Vertical, this);
    volumeSlider_->setMinimum(0);
    volumeSlider_->setMaximum(100);
    volumeSlider_->setValue(50); // Default volume at 50%
    volumeSlider_->setTickPosition(QSlider::TicksBothSides);
    volumeSlider_->setTickInterval(10);
    connect(volumeSlider_, &QSlider::valueChanged, this, &SynthWindow::onVolumeChanged);
    keyboardLayout->addWidget(volumeSlider_);

    mainLayout->addLayout(controlLayout);
    mainLayout->addLayout(keyboardLayout);

    // Enable keyboard focus
    setFocusPolicy(Qt::StrongFocus);

    updateDevices();
}

SynthWindow::~SynthWindow()
{
    for (auto &device : MIDIDevices)
    {
        device.midiIn->closePort();
    }
}

void SynthWindow::updateDevices()
{
    // Initialize MIDI
    try
    {
        // Open all available ports
        unsigned int nPorts = midiIn_->getPortCount();
        for (unsigned int i = 0; i < nPorts; i++)
        {
            try
            {
                // Create a new MIDI input for each port
                auto midiIn = RtMidiIn();
                midiIn.openPort(i);
                midiIn.setCallback(&SynthWindow::midiCallback, this);
                midiIn.ignoreTypes(false, false, false);
                MIDIDevice newDevice;
                newDevice.midiIn = &midiIn;
                newDevice.name = midiIn_->getPortName(i);
                MIDIDevices.push_back(newDevice);

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

void SynthWindow::keyPressEvent(QKeyEvent *event)
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

void SynthWindow::keyReleaseEvent(QKeyEvent *event)
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

void SynthWindow::midiCallback(double timeStamp, std::vector<unsigned char> *message, void *userData)
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

void SynthWindow::onTuningChanged(int index)
{
    for (auto &device : MIDIDevices)
    {
        for (auto &patch : device.patches)
        {
            patch.SetTuningSystem(index); // Pass the tuning system index directly
        }
    }
}

void SynthWindow::onVolumeChanged(int value)
{
    float normalizedVolume = value / 100.0f;
    compute->masterVolume = normalizedVolume;
}


#include <string>

// Handle command line arguments
std::string sampleFilename = "Harp.json";
int polyphony = 64;
int framesPerDispatch = 128;
unsigned int sampleRate; // this gets set to native sample rate of output device
int lfoCount = 16;
int envLenPerPatch = 512;
int outchannels = 2;
float bendDepth = 2.0f;
int bufferCount = 2;
int threadCount = 4;

// Print help information
void printHelp()
{
    std::cout << "Qt Sine Synth - Command Line Options:\n"
              << "  --test       Run in test mode\n"
              << "  --help, -h   Show this help message\n"
              << "  --polyphony <n>     Set polyphony (default: " << polyphony << ")\n"
              << "  --samples <n>       Set samples per dispatch (default: " << framesPerDispatch << ")\n"
              << "  --lfo <n>           Set LFO count (default: " << lfoCount << ")\n"
              << "  --env <n>           Set envelope length per patch (default: " << envLenPerPatch << ")\n"
              << "  --channels <n>      Set output channels (default: " << outchannels << ")\n"
              << "  --bend <n>          Set pitch bend depth (default: " << bendDepth << ")\n"
              << "  --buffers <n>       Set audio buffer count (default: " << bufferCount << ")\n"
              << "  --threadcount <n>   Set audio thread count (default: " << threadCount << ")\n"
              << "  --sampleFilename <file>  Set sample file path (default: " << sampleFilename << ")\n";
}

SampleCompute *compute;

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
            framesPerDispatch = std::stoi(argv[++i]);
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
        else if (arg == "--sampleFilename" && i + 1 < argc)
            sampleFilename = argv[++i];
    }

    InitAudio(bufferCount, framesPerDispatch, &outchannels, &sampleRate);

    QApplication app(argc, argv);
    compute = new SampleCompute(polyphony, framesPerDispatch, lfoCount, outchannels, bendDepth, sampleRate, threadCount);
    Patch *DefaultPatch = new Patch(sampleFilename, compute);

    std::cout << "Creating window" << std::endl;
    // Create and show the window
    SynthWindow window;
    window.show();

    std::cout << "Run the application" << std::endl;
    // Run the application
    int result = app.exec();
    DeInitAudio();
}
