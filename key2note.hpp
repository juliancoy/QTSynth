
#include <QtGui/QKeyEvent>

std::map<int, int> keyToNote_ = {
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