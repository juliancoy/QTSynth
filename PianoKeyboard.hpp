#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QKeyEvent>
#include <rtmidi/RtMidi.h>
#include <map>
#include <memory>
#include <QDebug>
#include "sample_compute.hpp"

class PianoKey : public QWidget
{
public:
    PianoKey(int note, bool isBlack, QWidget *parent = nullptr)
        : QWidget(parent), note_(note), isBlack_(isBlack), isPressed_(false)
    {
        setFixedSize(isBlack ? 24 : 36, isBlack ? 100 : 150);
    }

    void setPressed(bool pressed)
    {
        isPressed_ = pressed;
        update();
    }

    int note() const { return note_; }
    bool isBlack() const { return isBlack_; }

protected:
    void paintEvent(QPaintEvent *) override
    {
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

class PianoKeyboard : public QWidget
{
public:
    PianoKeyboard(QWidget *parent = nullptr)
        : QWidget(parent), activeKey_(nullptr)
    {
        setFixedHeight(160);
        setMinimumWidth(500);

        try
        {
            midiOut = std::make_unique<RtMidiOut>();

            if (midiOut->getPortCount() > 0)
            {
                midiOut->openPort(0); // Open the first available MIDI output port
            }
            else
            {
                qDebug() << "No MIDI output ports available.";
            }
        }
        catch (RtMidiError &error)
        {
            qDebug() << "MIDI Error: " << error.getMessage().c_str();
        }

        // Create piano keys (2 octaves starting from middle C)
        const int startNote = 60;                                       // Middle C
        const bool isBlackKey[] = {0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0}; // Pattern for one octave

        int x = 0;
        for (int i = 0; i < 24; ++i)
        {
            int note = startNote + i;
            bool isBlack = isBlackKey[i % 12];
            auto key = new PianoKey(note, isBlack, this);

            if (!isBlack)
            {
                key->move(x, 0);
                x += key->width();
            }
            else
            {
                key->move(x - key->width() / 2, 0);
            }

            keys_[note] = key;
        }
    }

    ~PianoKeyboard()
    {
        midiOut.reset(); // Ensure MIDI output is properly closed
    }

    void showKeyDepressed(int note)
    {
        auto it = keys_.find(note);
        if (it != keys_.end())
        {
            it->second->setPressed(true);
            activeKey_ = it->second;
        }
    }

    void keyPressed(int note)
    {
        auto it = keys_.find(note);
        if (it != keys_.end())
        {

            std::vector<unsigned char> message = {0x90, static_cast<unsigned char>(it->first), 127}; // Note on, note, velocity 127
            ProcessMidi(&message);
            it->second->setPressed(true);
            activeKey_ = it->second;
        }
    }

    void keyReleased(int note)
    {
        auto it = keys_.find(note);
        if (it != keys_.end())
        {
            std::vector<unsigned char> message = {0x80, static_cast<unsigned char>(it->first), 127}; // Note off, note, velocity 127
            ProcessMidi(&message);
            it->second->setPressed(false);
            if (activeKey_ == it->second)
            {
                activeKey_ = nullptr;
            }
        }
    }

    void showKeyReleased(int note)
    {
        auto it = keys_.find(note);
        if (it != keys_.end())
        {
            it->second->setPressed(false);
            if (activeKey_ == it->second)
            {
                activeKey_ = nullptr;
            }
        }
    }

protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        auto key = getKeyAtPosition(event->pos());
        if (key)
        {
            std::vector<unsigned char> message = {0x90, static_cast<unsigned char>(key->note()), 127}; // Note on, note, velocity 127
            ProcessMidi(&message);
            key->setPressed(true);
            activeKey_ = key;
        }
    }

    void mouseReleaseEvent(QMouseEvent *) override
    {
        if (activeKey_)
        {
            Release(activeKey_->note(), nullptr);
            activeKey_->setPressed(false);
            activeKey_ = nullptr;
        }
    }

private:
    PianoKey *getKeyAtPosition(const QPoint &pos)
    {
        // Check black keys first (they're on top)
        for (auto &[note, key] : keys_)
        {
            if (key->isBlack() && key->geometry().contains(pos))
            {
                return key;
            }
        }
        // Then check white keys
        for (auto &[note, key] : keys_)
        {
            if (!key->isBlack() && key->geometry().contains(pos))
            {
                return key;
            }
        }
        return nullptr;
    }

    std::map<int, PianoKey *> keys_;
    PianoKey *activeKey_; // Track currently pressed key
    std::unique_ptr<RtMidiOut> midiOut;
};

class MainWindow : public QMainWindow
{
public:
    MainWindow()
    {
        auto *keyboard = new PianoKeyboard(this);
        setCentralWidget(keyboard);
        setWindowTitle("Qt Piano with RtMIDI");
        resize(600, 200);
    }
};
