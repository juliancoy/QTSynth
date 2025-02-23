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
        {0, 1.0},         // C (Rast) - Tonic
        {1, 17.0 / 16.0}, // C# (Intermediate) - Between C & D
        {2, 9.0 / 8.0},   // D (Dugah) - Major second
        {3, 12.0 / 9.0}, // E (Intermediate) - Between D & E-half-flat
        {4, 11.0 / 9.0},  // E-half-flat (Sikah) - Neutral third
        {5, 4.0 / 3.0},   // F (Jaharkah) - Perfect fourth
        {6, 19.0 / 14.0}, // F# (Intermediate) - Between F & G
        {7, 3.0 / 2.0},   // G (Nawa) - Perfect fifth
        {8, 27.0 / 17.0}, // G# (Intermediate) - Between G & A
        {9, 5.0 / 3.0},   // A (Husseini) - Major sixth
        {10, 9.0 / 5.0}, // Intermediate
        {11, 7.0 / 4.0},  // B-half-flat (Awj) - Neutral seventh
        {12, 2.0 / 1.0}   // C (Gerdaniye) - Octave
    };

    // Find the reference octave for the input MIDI note
    int octaveOffset = (midiNote / 12) - 5; // MIDI 60 (C4) is the reference octave
    int scaleDegree = midiNote % 12;        // Get the note within the octave

    // Ensure we have a valid interval from Maqam Rast
    if (rastScale.find(scaleDegree) == rastScale.end())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid scale degree
    }

    // Calculate the frequency by applying the Just Intonation ratio and octave transposition
    double frequency = baseFreq * rastScale[scaleDegree] * std::pow(2.0, octaveOffset);

    return frequency;
}

// Convert MIDI note to frequency based on Pythagorean tuning
double midiNoteToPythagoreanFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C)
    const double baseFreq = 261.63; // Frequency of Middle C in Hz

    // Define the Pythagorean tuning scale degrees with frequency ratios
    std::unordered_map<int, double> pythagoreanScale = {
        {0, 1.0},         // C - Tonic
        {1, 256.0 / 243.0}, // C# - Pythagorean minor second
        {2, 9.0 / 8.0},   // D - Major second (whole tone)
        {3, 32.0 / 27.0}, // D# - Pythagorean minor third
        {4, 81.0 / 64.0}, // E - Major third
        {5, 4.0 / 3.0},   // F - Perfect fourth
        {6, 729.0 / 512.0}, // F# - Augmented fourth / Diminished fifth (Pythagorean tritone)
        {7, 3.0 / 2.0},   // G - Perfect fifth
        {8, 128.0 / 81.0}, // G# - Pythagorean minor sixth
        {9, 27.0 / 16.0}, // A - Major sixth
        {10, 16.0 / 9.0}, // A# - Pythagorean minor seventh
        {11, 243.0 / 128.0}, // B - Major seventh
        {12, 2.0 / 1.0}   // C - Octave
    };

    // Find the reference octave for the input MIDI note
    int octaveOffset = (midiNote / 12) - 5; // MIDI 60 (C4) is the reference octave
    int scaleDegree = midiNote % 12;        // Get the note within the octave

    // Ensure we have a valid interval from Pythagorean tuning
    if (pythagoreanScale.find(scaleDegree) == pythagoreanScale.end())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid scale degree
    }

    // Calculate the frequency by applying the Just Intonation ratio and octave transposition
    double frequency = baseFreq * pythagoreanScale[scaleDegree] * std::pow(2.0, octaveOffset);

    return frequency;
}


// Convert MIDI note to frequency based on Raga Yaman (Just Intonation)
double midiNoteToRagaYamanFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C)
    const double baseFreq = 261.63; // Frequency of Middle C in Hz

    // Define the Raga Yaman scale degrees with Just Intonation ratios
    std::unordered_map<int, double> yamanScale = {
        {0, 1.0},         // C (Sa) - Tonic
        {1, 16.0 / 15.0}, // C# (Komal Re) - Minor second (used ornamentally)
        {2, 9.0 / 8.0},   // D (Shuddh Re) - Major second
        {3, 6.0 / 5.0},   // D# (Komal Ga) - Minor third (not in Yaman, but available)
        {4, 5.0 / 4.0},   // E (Shuddh Ga) - Major third
        {5, 4.0 / 3.0},   // F (Ma) - Perfect fourth
        {6, 45.0 / 32.0}, // F# (Tivra Ma) - Augmented fourth (Yaman’s special note)
        {7, 3.0 / 2.0},   // G (Pa) - Perfect fifth
        {8, 8.0 / 5.0},   // G# (Komal Dha) - Minor sixth (not in Yaman, but available)
        {9, 5.0 / 3.0},   // A (Shuddh Dha) - Major sixth
        {10, 9.0 / 5.0},  // A# (Komal Ni) - Minor seventh (not in Yaman)
        {11, 15.0 / 8.0}, // B (Shuddh Ni) - Major seventh
        {12, 2.0 / 1.0}   // C (Sa) - Octave
    };

    // Find the reference octave for the input MIDI note
    int octaveOffset = (midiNote / 12) - 5; // MIDI 60 (C4) is the reference octave
    int scaleDegree = midiNote % 12;        // Get the note within the octave

    // Ensure we have a valid interval from Raga Yaman
    if (yamanScale.find(scaleDegree) == yamanScale.end())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid scale degree
    }

    // Calculate the frequency by applying the Just Intonation ratio and octave transposition
    double frequency = baseFreq * yamanScale[scaleDegree] * std::pow(2.0, octaveOffset);

    return frequency;
}

// Convert MIDI note to frequency based on the Bohlen-Pierce scale
double midiNoteToBohlenPierceFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C in ET, "Gamma" in BP)
    const double baseFreq = 261.63; // Frequency of Gamma in Hz (Middle C equivalent)

    // Define the Bohlen-Pierce scale degrees with Just Intonation frequency ratios
    std::unordered_map<int, double> bohlenPierceScale = {
        {0, 1.0},           // Gamma (Tonic)
        {1, 27.0 / 25.0},   // Delta (Small Second)
        {2, 25.0 / 21.0},   // Epsilon (Large Second)
        {3, 9.0 / 7.0},     // Zeta (Small Third)
        {4, 7.0 / 5.0},     // Eta (Large Third)
        {5, 75.0 / 49.0},   // Theta (Fourth)
        {6, 5.0 / 3.0},     // Iota (Augmented Fourth)
        {7, 9.0 / 5.0},     // Kappa (Fifth)
        {8, 49.0 / 25.0},   // Lambda (Small Sixth)
        {9, 15.0 / 7.0},    // Mu (Large Sixth)
        {10, 7.0 / 3.0},    // Nu (Small Seventh)
        {11, 63.0 / 25.0},  // Xi (Large Seventh)
        {12, 3.0 / 1.0}     // Tritave (3:1, like an octave but larger)
    };

    // Find the reference tritave for the input MIDI note
    int tritaveOffset = (midiNote / 13) - 4; // MIDI 60 (Gamma) is the reference
    int scaleDegree = midiNote % 13;         // Get the note within the 13-tone BP scale

    // Ensure we have a valid interval from the Bohlen-Pierce scale
    if (bohlenPierceScale.find(scaleDegree) == bohlenPierceScale.end())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid BP scale degree
    }

    // Calculate the frequency by applying the Just Intonation ratio and tritave transposition
    double frequency = baseFreq * bohlenPierceScale[scaleDegree] * std::pow(3.0, tritaveOffset);

    return frequency;
}

// Convert MIDI note to frequency based on Maqam Bayati (Just Intonation)
double midiNoteToMaqamBayatiFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C)
    const double baseFreq = 261.63; // Frequency of Middle C in Hz

    // Define the Maqam Bayati scale degrees with Just Intonation frequency ratios
    std::unordered_map<int, double> bayatiScale = {
        {0, 1.0},           // C (Tonic - Bayati on C)
        {1, 16.0 / 15.0},   // C# / Db (Lowered Second - Sika) (~70 cents above C)
        {2, 6.0 / 5.0},     // D (Neutral Third) (~350 cents above C)
        {3, 32.0 / 25.0},   // D# (Alternative Neutral Third - sometimes used)
        {4, 4.0 / 3.0},     // E / F (Perfect Fourth)
        {5, 45.0 / 32.0},   // F# / Gb (Intermediate note between F and G)
        {6, 3.0 / 2.0},     // G (Perfect Fifth)
        {7, 8.0 / 5.0},     // G# / Ab (Major Sixth)
        {8, 27.0 / 16.0},   // A (Alternative Sixth - sometimes used)
        {9, 9.0 / 5.0},     // A# / Bb (Neutral Seventh)
        {10, 15.0 / 8.0},   // B (Major Seventh)
        {11, 2.0 / 1.0}     // C (Octave)
    };

    // Find the reference octave for the input MIDI note
    int octaveOffset = (midiNote / 12) - 5; // MIDI 60 (C4) is the reference octave
    int scaleDegree = midiNote % 12;        // Get the note within the octave

    // Ensure we have a valid interval from Maqam Bayati
    if (bayatiScale.find(scaleDegree) == bayatiScale.end())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid scale degree
    }

    // Calculate the frequency by applying the Just Intonation ratio and octave transposition
    double frequency = baseFreq * bayatiScale[scaleDegree] * std::pow(2.0, octaveOffset);

    return frequency;
}


// Convert MIDI note to frequency based on Slendro-Pelog Hybrid (Beyond Octave)
double midiNoteToSlendroPelogFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C)
    const double baseFreq = 261.63; // Frequency of Middle C in Hz

    // Define the Slendro-Pelog Extended scale degrees with Just Intonation ratios
    std::unordered_map<int, double> slendroPelogScale = {
        {0, 1.0},        // C (Base)
        {1, 9.0 / 8.0},  // D
        {2, 5.0 / 4.0},  // E
        {3, 4.0 / 3.0},  // F
        {4, 3.0 / 2.0},  // G
        {5, 5.0 / 3.0},  // A
        {6, 7.0 / 4.0},  // Bb
        {7, 2.0 / 1.0},  // C (Octave)
        {8, 13.0 / 6.0}, // D (Low second)
        {9, 11.0 / 5.0}, // E (High third)
        {10, 9.0 / 4.0}, // F (Overtone fourth)
        {11, 8.0 / 3.0}, // G (Stretched fifth)
        {12, 7.0 / 3.0}, // A (Low sixth)
        {13, 3.0 / 1.0}, // C (Beyond octave)
        {14, 10.0 / 3.0},// D# (Beyond octave)
        {15, 14.0 / 4.0},// F# (Weird Tritone)
        {16, 22.0 / 7.0} // G# (Far overtone)
    };

    // Find the reference range for the input MIDI note
    int extendedOctaveOffset = (midiNote / 17) - 3; // MIDI 60 is reference C
    int scaleDegree = midiNote % 17;  // Get the note within the extended cycle

    // Ensure we have a valid interval from Slendro-Pelog
    if (slendroPelogScale.find(scaleDegree) == slendroPelogScale.end())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid scale degree
    }

    // Calculate the frequency by applying the ratio and extended range transposition
    double frequency = baseFreq * slendroPelogScale[scaleDegree] * std::pow(3.0, extendedOctaveOffset);

    return frequency;
}

double midiNoteToHarmonicSeriesFreq(int midiNote)
{
    // Base frequency for MIDI note 60 (Middle C)
    const double baseFreq = 261.63; // Frequency of Middle C in Hz

    // Define the first 16 Harmonic Series degrees with exact frequency ratios
    std::unordered_map<int, double> harmonicSeriesScale = {
        {0, 1.0},   // C (Tonic)
        {1, 2.0},   // C (Octave)
        {2, 3.0},   // G (Perfect Fifth)
        {3, 4.0},   // C (Two Octaves)
        {4, 5.0},   // E (Major Third)
        {5, 6.0},   // G (Perfect Fifth)
        {6, 7.0},   // Bb (Natural 7th)
        {7, 8.0},   // C (Three Octaves)
        {8, 9.0},   // D (Major Second)
        {9, 10.0},  // E (Major Third)
        {10, 11.0}, // F# (Overtone 11th)
        {11, 12.0}, // G (Perfect Fifth)
        {12, 13.0}, // Ab (Overtone 13th)
        {13, 14.0}, // Bb (Natural 7th)
        {14, 15.0}, // B (Major Seventh)
        {15, 16.0}  // C (Four Octaves)
    };

    // Calculate the octave and scale degree
    int octave = (midiNote - 60) / 16; // MIDI 60 is reference C (octave 0)
    int scaleDegree = (midiNote - 60) % 16; // Get the note within the harmonic series

    // Ensure the scale degree is valid
    if (scaleDegree < 0 || scaleDegree >= harmonicSeriesScale.size())
    {
        return -1.0; // Error case: MIDI note does not correspond to a valid harmonic series degree
    }

    // Calculate the frequency by applying the harmonic series ratio and octave transposition
    double frequency = baseFreq * harmonicSeriesScale[scaleDegree] * std::pow(2.0, octave);

    return frequency;
}