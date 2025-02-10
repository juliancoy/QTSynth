#include <QApplication>
#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>
#include <RtAudio.h>
#include <cmath>
#include <memory>

const double PI = 3.14159265358979323846;
const int SAMPLE_RATE = 44100;
unsigned int FRAMES_PER_BUFFER = 256;
const int NUM_CHANNELS = 1;

class SineWaveGenerator {
public:
    SineWaveGenerator(double frequency = 440.0)
        : frequency_(frequency), phase_(0.0), isPlaying_(false) {}

    void setPlaying(bool playing) { isPlaying_ = playing; }
    bool isPlaying() const { return isPlaying_; }

    double generateSample() {
        if (!isPlaying_) return 0.0;
        
        double sample = std::sin(phase_);
        phase_ += 2.0 * PI * frequency_ / SAMPLE_RATE;
        
        if (phase_ >= 2.0 * PI) {
            phase_ -= 2.0 * PI;
        }
        
        return sample * 0.5; // Reduce amplitude to 0.5
    }

private:
    double frequency_;
    double phase_;
    bool isPlaying_;
};

class SynthWindow : public QMainWindow {
public:
    SynthWindow(std::shared_ptr<SineWaveGenerator> generator)
        : generator_(generator) {
        setWindowTitle("Qt Sine Synth");
        resize(300, 200);

        auto centralWidget = new QWidget(this);
        setCentralWidget(centralWidget);

        auto layout = new QVBoxLayout(centralWidget);
        
        toggleButton_ = new QPushButton("Start", this);
        layout->addWidget(toggleButton_);

        connect(toggleButton_, &QPushButton::clicked, this, &SynthWindow::toggleSound);
    }

private slots:
    void toggleSound() {
        bool newState = !generator_->isPlaying();
        generator_->setPlaying(newState);
        toggleButton_->setText(newState ? "Stop" : "Start");
    }

private:
    std::shared_ptr<SineWaveGenerator> generator_;
    QPushButton* toggleButton_;
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

    // Initialize RtAudio
    RtAudio dac;
    if (dac.getDeviceCount() < 1) {
        std::cerr << "No audio devices found!" << std::endl;
        return -1;
    }

    // Set output parameters
    RtAudio::StreamParameters parameters;
    parameters.deviceId = dac.getDefaultOutputDevice();
    parameters.nChannels = NUM_CHANNELS;
    parameters.firstChannel = 0;

    // Open the stream
    try {
        dac.openStream(&parameters, nullptr, RTAUDIO_FLOAT32,
                      SAMPLE_RATE, &FRAMES_PER_BUFFER, &audioCallback,
                      generator.get());
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

    return result;
}
