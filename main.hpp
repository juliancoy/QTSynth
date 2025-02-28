#pragma once

#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtWidgets/QSlider>
#include <QtWidgets/QComboBox>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>
#include <rtmidi/RtMidi.h>
#include "sample_compute.hpp"
#include "patch.hpp"
#include "key2note.hpp"
#include "PianoKeyboard.hpp"
#include "threads.hpp"
#include <cmath>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <fstream>
#include <iostream>

typedef struct MIDIDevice
{
    std::vector<Patch> patches;
    RtMidiIn *midiIn;
    std::string name;
} MIDIDevice;


class SynthWindow : public QMainWindow
{
    Q_OBJECT

public:
    SynthWindow();
    ~SynthWindow();

    void updateDevices();

    struct MIDIDevice
    {
        RtMidiIn *midiIn;
        std::string name;
    };
    RtMidiIn *midiIn_;

    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;

    QComboBox *tuningSelector_;
    QSlider *volumeSlider_;
    PianoKeyboard *keyboard_;


    std::vector<MIDIDevice> MIDIDevices;
    std::map<int, int> keyToNote_;

    static void midiCallback(double timeStamp, std::vector<unsigned char> *message, void *userData);
    void onTuningChanged(int index);
    void onVolumeChanged(int value);
    static void ProcessMidi(std::vector<unsigned char> *message);


    PianoKeyboard *keyboard_;
    QComboBox *tuningSelector_;
    QSlider *volumeSlider_;
    std::vector<MIDIDevice> MIDIDevices;
    float currentVolume_ = 0.5f; // Track current volume

};
