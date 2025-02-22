#include <cmath>
#include <unordered_map>

// Convert MIDI note to frequency
double midiNoteTo12TETFreq(int note)
{
    return 440.0 * std::pow(2.0, (note - 69) / 12.0);
}

// Convert MIDI note to frequency based on Maqam Rast
double midiNoteToMaqamRastFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C)
    const double baseFreq = 261.63; // Frequency of Middle C in Hz

    // Define the Maqam Rast scale degrees with exact harmonic ratios
    // These are not cent deviations but precise frequency multipliers in Just Intonation
    std::unordered_map<int, double> rastScale = {
        {0, 1.0},        // C (Rast) - Tonic
        {1, 17.0/16.0},  // C# (Intermediate) - Between C & D
        {2, 9.0/8.0},    // D (Dugah) - Major second
        {3, 11.0/9.0},   // E-half-flat (Sikah) - Neutral third
        {4, 13.0/10.0},  // E (Intermediate) - Between E-half-flat & F
        {5, 4.0/3.0},    // F (Jaharkah) - Perfect fourth
        {6, 19.0/14.0},  // F# (Intermediate) - Between F & G
        {7, 3.0/2.0},    // G (Nawa) - Perfect fifth
        {8, 27.0/17.0},  // G# (Intermediate) - Between G & A
        {9, 5.0/3.0},    // A (Husseini) - Major sixth
        {10, 7.0/4.0},   // B-half-flat (Awj) - Neutral seventh
        {11, 15.0/8.0},  // B (Intermediate) - Between B-half-flat & C
        {12, 2.0/1.0}    // C (Gerdaniye) - Octave
    };

    // Find the reference octave for the input MIDI note
    int octaveOffset = (midiNote / 12) - 5; // MIDI 60 (C4) is the reference octave
    int scaleDegree = midiNote % 12; // Get the note within the octave

    // Ensure we have a valid interval from Maqam Rast
    if (rastScale.find(scaleDegree) == rastScale.end()) {
        return -1.0; // Error case: MIDI note does not correspond to a valid scale degree
    }

    // Calculate the frequency by applying the Just Intonation ratio and octave transposition
    double frequency = baseFreq * rastScale[scaleDegree] * std::pow(2.0, octaveOffset);

    return frequency;
}