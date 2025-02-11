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
#include <cmath>
#include <memory>
#include <map>
#include <vector>
#include <mutex>

const double PI = 3.14159265358979323846;
const int SAMPLE_RATE = 44100;
// Buffer size of 64 frames @ 44100Hz = ~1.45ms latency
unsigned int FRAMES_PER_BUFFER = 64;
const int NUM_CHANNELS = 1;

// Convert MIDI note to frequency
double midiNoteToFreq(int note) {
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

class Voice {
public:
    Voice(double frequency = 440.0)
        : frequency_(frequency), phase_(0.0), isActive_(false) {}

    void start(double frequency) {
        frequency_ = frequency;
        isActive_ = true;
    }

    void stop() {
        isActive_ = false;
    }

    bool isActive() const { return isActive_; }

    double generateSample() {
        if (!isActive_) return 0.0;
        
        double sample = std::sin(phase_);
        phase_ += 2.0 * PI * frequency_ / SAMPLE_RATE;
        
        if (phase_ >= 2.0 * PI) {
            phase_ -= 2.0 * PI;
        }
        
        return sample * 0.5;
    }

private:
    double frequency_;
    double phase_;
    bool isActive_;
};

class SineWaveGenerator {
public:
    SineWaveGenerator() {
        voices_.resize(128); // One voice per MIDI note
        for (int i = 0; i < 128; i++) {
            voices_[i] = std::make_unique<Voice>(midiNoteToFreq(i));
        }
    }

    void noteOn(int note, int velocity) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (note >= 0 && note < 128) {
            voices_[note]->start(midiNoteToFreq(note));
        }
    }

    void noteOff(int note) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (note >= 0 && note < 128) {
            voices_[note]->stop();
        }
    }

    double generateSample() {
        std::lock_guard<std::mutex> lock(mutex_);
        double sample = 0.0;
        int activeVoices = 0;
        
        for (auto& voice : voices_) {
            if (voice->isActive()) {
                sample += voice->generateSample();
                activeVoices++;
            }
        }
        
        // Normalize output based on number of active voices
        return activeVoices > 0 ? sample / activeVoices : 0.0;
    }

private:
    std::vector<std::unique_ptr<Voice>> voices_;
    std::mutex mutex_;
};

class PianoKey : public QWidget {
public:
    PianoKey(int note, bool isBlack, QWidget* parent = nullptr)
        : QWidget(parent), note_(note), isBlack_(isBlack), isPressed_(false) {
        setFixedSize(isBlack ? 24 : 36, isBlack ? 100 : 150);
    }

    void setPressed(bool pressed) {
        isPressed_ = pressed;
        update();
    }

    int note() const { return note_; }
    bool isBlack() const { return isBlack_; }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter painter(this);
        painter.setRenderHint(QPainter::Antialiasing);

        QColor keyColor = isBlack_ 
            ? (isPressed_ ? Qt::darkGray : Qt::black)
            : (isPressed_ ? Qt::lightGray : Qt::white);
        
        painter.fillRect(rect(), keyColor);
        painter.setPen(Qt::black);
        painter.drawRect(rect().adjusted(0, 0, -1, -1));
    }

private:
    int note_;
    bool isBlack_;
    bool isPressed_;
};

class PianoKeyboard : public QWidget {
public:
    PianoKeyboard(std::shared_ptr<SineWaveGenerator> generator, QWidget* parent = nullptr)
        : QWidget(parent), generator_(generator), activeKey_(nullptr) {
        setFixedHeight(160);
        setMinimumWidth(500);

        // Create piano keys (2 octaves starting from middle C)
        const int startNote = 60; // Middle C
        const bool isBlackKey[] = {0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0}; // Pattern for one octave
        
        int x = 0;
        for (int i = 0; i < 24; ++i) {
            int note = startNote + i;
            bool isBlack = isBlackKey[i % 12];
            auto key = new PianoKey(note, isBlack, this);
            
            if (!isBlack) {
                key->move(x, 0);
                x += key->width();
            } else {
                key->move(x - key->width()/2, 0);
            }
            
            keys_[note] = key;
        }
    }

    void keyPressed(int note) {
        if (auto it = keys_.find(note); it != keys_.end()) {
            it->second->setPressed(true);
        }
    }

    void keyReleased(int note) {
        if (auto it = keys_.find(note); it != keys_.end()) {
            it->second->setPressed(false);
        }
    }

protected:
    void mousePressEvent(QMouseEvent* event) override {
        auto key = getKeyAtPosition(event->pos());
        if (key) {
            generator_->noteOn(key->note(), 64);
            key->setPressed(true);
            activeKey_ = key;
        }
    }

    void mouseReleaseEvent(QMouseEvent*) override {
        if (activeKey_) {
            generator_->noteOff(activeKey_->note());
            activeKey_->setPressed(false);
            activeKey_ = nullptr;
        }
    }

private:
    PianoKey* getKeyAtPosition(const QPoint& pos) {
        // Check black keys first (they're on top)
        for (auto& [note, key] : keys_) {
            if (key->isBlack() && key->geometry().contains(pos)) {
                return key;
            }
        }
        // Then check white keys
        for (auto& [note, key] : keys_) {
            if (!key->isBlack() && key->geometry().contains(pos)) {
                return key;
            }
        }
        return nullptr;
    }

    std::shared_ptr<SineWaveGenerator> generator_;
    std::map<int, PianoKey*> keys_;
    PianoKey* activeKey_; // Track currently pressed key
};

class SynthWindow : public QMainWindow {
public:
    SynthWindow(std::shared_ptr<SineWaveGenerator> generator)
        : generator_(generator) {
        setWindowTitle("Qt Sine Synth");
        resize(600, 200);

        auto centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        auto layout = new QVBoxLayout(centralWidget);
        
        keyboard_ = new PianoKeyboard(generator_, this);
        layout->addWidget(keyboard_);

        // Enable keyboard focus
        setFocusPolicy(Qt::StrongFocus);
        
        // Initialize key mapping (QWERTY layout mapped to musical notes)
        // Starting from middle C (60) on the home row
        keyToNote_ = {
            // Numbers row (higher octave)
            {Qt::Key_1, 72}, // C5
            {Qt::Key_2, 74}, // D5
            {Qt::Key_3, 76}, // E5
            {Qt::Key_4, 77}, // F5
            {Qt::Key_5, 79}, // G5
            {Qt::Key_6, 81}, // A5
            {Qt::Key_7, 83}, // B5
            {Qt::Key_8, 84}, // C6
            {Qt::Key_9, 86}, // D6
            {Qt::Key_0, 88}, // E6
            {Qt::Key_Minus, 89}, // F6
            {Qt::Key_Equal, 91}, // G6
            
            // Top letter row
            {Qt::Key_Q, 67}, // G4
            {Qt::Key_W, 69}, // A4
            {Qt::Key_E, 71}, // B4
            {Qt::Key_R, 72}, // C5
            {Qt::Key_T, 74}, // D5
            {Qt::Key_Y, 76}, // E5
            {Qt::Key_U, 77}, // F5
            {Qt::Key_I, 79}, // G5
            {Qt::Key_O, 81}, // A5
            {Qt::Key_P, 83}, // B5
            {Qt::Key_BracketLeft, 84}, // C6
            {Qt::Key_BracketRight, 86}, // D6
            
            // Home row (middle)
            {Qt::Key_A, 60}, // Middle C (C4)
            {Qt::Key_S, 62}, // D4
            {Qt::Key_D, 64}, // E4
            {Qt::Key_F, 65}, // F4
            {Qt::Key_G, 67}, // G4
            {Qt::Key_H, 69}, // A4
            {Qt::Key_J, 71}, // B4
            {Qt::Key_K, 72}, // C5
            {Qt::Key_L, 74}, // D5
            {Qt::Key_Semicolon, 76}, // E5
            {Qt::Key_Apostrophe, 77}, // F5
            
            // Bottom letter row
            {Qt::Key_Z, 48}, // C3
            {Qt::Key_X, 50}, // D3
            {Qt::Key_C, 52}, // E3
            {Qt::Key_V, 53}, // F3
            {Qt::Key_B, 55}, // G3
            {Qt::Key_N, 57}, // A3
            {Qt::Key_M, 59}, // B3
            {Qt::Key_Comma, 60}, // C4
            {Qt::Key_Period, 62}, // D4
            {Qt::Key_Slash, 64}, // E4
        };

        // Initialize MIDI
        try {
            midiIn_ = std::make_unique<RtMidiIn>();
            
            // Open all available ports
            unsigned int nPorts = midiIn_->getPortCount();
            for (unsigned int i = 0; i < nPorts; i++) {
                try {
                    // Create a new MIDI input for each port
                    auto midiIn = std::make_unique<RtMidiIn>();
                    midiIn->openPort(i);
                    midiIn->setCallback(&SynthWindow::midiCallback, this);
                    midiIn->ignoreTypes(false, false, false);
                    midiInputs_.push_back(std::move(midiIn));
                    
                    std::cout << "Opened MIDI port " << i << ": " 
                              << midiIn_->getPortName(i) << std::endl;
                } catch (RtMidiError &error) {
                    error.printMessage();
                }
            }
        } catch (RtMidiError &error) {
            error.printMessage();
        }
    }

    ~SynthWindow() {
        for (auto& midiIn : midiInputs_) {
            midiIn->closePort();
        }
    }

protected:
    void keyPressEvent(QKeyEvent* event) override {
        if (event->isAutoRepeat()) return;
        
        if (auto it = keyToNote_.find(event->key()); it != keyToNote_.end()) {
            generator_->noteOn(it->second, 100);
            keyboard_->keyPressed(it->second);
        }
    }

    void keyReleaseEvent(QKeyEvent* event) override {
        if (event->isAutoRepeat()) return;
        
        if (auto it = keyToNote_.find(event->key()); it != keyToNote_.end()) {
            generator_->noteOff(it->second);
            keyboard_->keyReleased(it->second);
        }
    }

private:
    static void midiCallback(double timeStamp, std::vector<unsigned char>* message, void* userData) {
        auto* window = static_cast<SynthWindow*>(userData);
        if (message->size() < 3) return;

        unsigned char status = message->at(0);
        unsigned char note = message->at(1);
        unsigned char velocity = message->at(2);

        if ((status & 0xF0) == 0x90 && velocity > 0) {  // Note On
            window->generator_->noteOn(note, velocity);
            window->keyboard_->keyPressed(note);
        } else if ((status & 0xF0) == 0x80 || ((status & 0xF0) == 0x90 && velocity == 0)) {  // Note Off
            window->generator_->noteOff(note);
            window->keyboard_->keyReleased(note);
        }
    }

    std::shared_ptr<SineWaveGenerator> generator_;
    PianoKeyboard* keyboard_;
    std::unique_ptr<RtMidiIn> midiIn_;
    std::vector<std::unique_ptr<RtMidiIn>> midiInputs_;
    std::map<int, int> keyToNote_; // Maps Qt key codes to MIDI note numbers
};

int audioCallback(void* outputBuffer, void* /*inputBuffer*/, unsigned int nBufferFrames,
                 double /*streamTime*/, RtAudioStreamStatus /*status*/, void* userData) {
    auto* generator = static_cast<SineWaveGenerator*>(userData);
    auto* buffer = static_cast<float*>(outputBuffer);

    for (unsigned int i = 0; i < nBufferFrames; i++) {
        buffer[i] = static_cast<float>(generator->generateSample());
    }

    return 0;
}

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // Create audio generator
    auto generator = std::make_shared<SineWaveGenerator>();

    RtAudio dac(RtAudio::LINUX_PULSE);
    unsigned int devices = dac.getDeviceCount();
    if (devices < 1) {
        std::cerr << "No audio devices found!" << std::endl;
        return -1;
    }

    std::cout << "Available audio devices:" << std::endl;
    RtAudio::DeviceInfo info;
    for (unsigned int i = 0; i < devices; i++) {
        try {
            info = dac.getDeviceInfo(i);
            std::cout << "Device " << i << ": " << info.name << std::endl;
        } catch (RtAudioErrorType &error) {
            std::cerr << error << std::endl;
        }
    }


    // Set output parameters
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac.getDefaultOutputDevice();
    parameters.nChannels = NUM_CHANNELS;
    parameters.firstChannel = 0;

    // Open the stream with minimal buffering for low latency
    try {
        RtAudio::StreamOptions options;
        options.numberOfBuffers = 2; // Minimum number of buffers for stable playback
        options.flags = RTAUDIO_MINIMIZE_LATENCY; // Request minimum latency
        
        dac.openStream(&parameters, nullptr, RTAUDIO_FLOAT32,
                      SAMPLE_RATE, &FRAMES_PER_BUFFER, &audioCallback,
                      generator.get(), &options);
        dac.startStream();
    } catch (RtAudioErrorType& e) {
        std::cerr << "Error: " << e << std::endl;
        return -1;
    }

    // Create and show the window
    SynthWindow window(generator);
    window.show();

    // Run the application
    int result = app.exec();

    // Clean up audio
    if (dac.isStreamOpen()) {
        dac.stopStream();
        dac.closeStream();
    }

