# Voice-Controlled Big Mouth Billy Bass

Building custom speech recognition from scratch for a Big Mouth Billy Bass fish toy.

## What We're Building

A custom speech recognition system that lets you control a Big Mouth Billy Bass fish with voice commands. Instead of using existing APIs like Whisper, we're building the entire audio processing and recognition pipeline from scratch.

This is a **learning project** - we'll build incrementally, understanding each component deeply.

## How It Works

```
Microphone → Audio Buffer → MFCC Features → DTW Matcher → Command → Fish Response
```

1. **Audio Input**: You handle recording (raw audio data)
2. **Feature Extraction**: Convert audio to MFCCs (Mel-Frequency Cepstral Coefficients)
3. **Pattern Matching**: Use Dynamic Time Warping (DTW) to compare to recorded templates
4. **Command Recognition**: Match audio pattern to known commands
5. **Response**: Fish acts based on recognized command

## Why Custom Speech Recognition?

- **Educational**: Learn audio signal processing, FFT, MFCCs, DTW
- **Explainable**: You can see why it matched a specific pattern
- **No Cloud**: Everything runs locally
- **Sufficient**: Template-based recognition works great for limited vocabulary (10-50 commands)

## Key Technologies

- **MFCC**: Mel-Frequency Cepstral Coefficients for audio feature extraction
- **DTW**: Dynamic Time Warping for comparing audio sequences of different lengths
- **NumPy/SciPy**: For signal processing and matrix operations
- **Librosa**: For audio analysis (optional - we can implement MFCCs ourselves)

## Project Structure

```
Building_An_LLM/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── audio/                       # Audio I/O and preprocessing
│   ├── io.py                    # Load/save audio files
│   └── preprocess.py            # Normalization, silence removal
├── features/                    # Feature extraction
│   └── mfcc.py                  # MFCC implementation
├── recognition/                 # Speech recognition
│   ├── dtw.py                   # Dynamic Time Warping
│   └── matcher.py               # Template matching logic
├── templates/                   # Voice command templates
│   ├── record.py                # Record new templates
│   └── commands/                # Stored template files
│       ├── sing/
│       ├── dance/
│       └── ...
├── brain/                       # Response generation
│   └── responses.py             # Command → action mapping
├── fish/                        # Fish hardware control
│   └── controller.py            # Actuator interface
├── main.py                      # Main entry point
└── utils/
    └── visualize.py             # Plot waveforms, spectrograms
```

## Learning Path

### Phase 1: Audio Basics
- Understand digital audio (sampling, bit depth, sample rates)
- Load and save WAV files
- Visualize waveforms
- Normalize audio levels

### Phase 2: Feature Extraction (MFCCs)
- Learn what MFCCs are and why they matter
- Implement or use MFCC extraction
- Extract 13 coefficients from audio
- Visualize MFCC features as heatmap

### Phase 3: Pattern Matching (DTW)
- Learn Dynamic Time Warping algorithm
- Compare two audio sequences
- Handle different speaking speeds
- Find optimal alignment path

### Phase 4: Template Recording
- Record 3-5 samples of each command
- Store as templates
- Build command database

### Phase 5: Recognition System
- Match incoming audio to templates
- Calculate confidence scores
- Handle "unknown" commands
- Real-time recognition loop

### Phase 6: Fish Integration
- Map commands to fish actions
- Trigger actuators
- Add personality/responses

## Key Concepts to Learn

### Audio Processing
- **Sample Rate**: 16kHz (16,000 samples per second)
- **Frame Size**: 25ms windows
- **Frame Step**: 10ms overlap
- **Normalization**: Scale amplitude to consistent range
- **Silence Removal**: Detect speech vs silence

### MFCC Features
1. Pre-emphasis (boost high frequencies)
2. Framing (split into overlapping windows)
3. Windowing (Hamming window to reduce edge artifacts)
4. FFT (Fast Fourier Transform to frequency domain)
5. Mel Filterbank (apply perceptual frequency scale)
6. Log compression (human perception is logarithmic)
7. DCT (Discrete Cosine Transform to decorrelate)
8. Keep first 13 coefficients

**Output**: Matrix of (num_frames, 13) for each audio clip

### Dynamic Time Warping
- Compares two sequences of different lengths
- Finds optimal alignment (warping path)
- Uses dynamic programming
- Returns distance (smaller = more similar)

## Example Commands

```
"sing"      → Fish sings a song
"dance"     → Fish dances (flaps tail)
"hello"     → Fish says hello
"joke"      → Fish tells a joke
"weather"   → Fish reports weather
"time"      → Fish tells time
"sleep"     → Fish goes to sleep mode
"wake"      → Fish wakes up
```

## Getting Started

### 1. Install Dependencies

```bash
pip install numpy scipy librosa soundfile matplotlib
```

### 2. Record Your Voice

You handle the microphone recording. Save as:
- Format: WAV
- Sample rate: 16kHz
- Channels: Mono
- Bit depth: 16-bit

### 3. Try the Examples

```bash
# Phase 1: Visualize audio
python -c "from audio.io import load; import matplotlib.pyplot as plt; audio, sr = load('test.wav'); plt.plot(audio); plt.show()"

# Phase 2: Extract MFCCs
python -c "from features.mfcc import extract; import numpy as np; mfcc = extract('test.wav'); print(mfcc.shape)"

# Phase 3: Compare two audio files with DTW
python recognition/dtw.py audio1.wav audio2.wav
```

### 4. Record Templates

```bash
python templates/record.py --command "sing" --samples 3
```

### 5. Test Recognition

```bash
python main.py --mode test --input test_audio.wav
```

## Resources

### Audio Processing
- [Think DSP](https://greenteapress.com/thinkdsp/) - Free book on digital signal processing
- [DSP Guide](https://dspguide.com/) - Comprehensive resource

### MFCC
- [MFCC Tutorial](https://www.youtube.com/watch?v=4_rn4THcRis) - YouTube explanation
- [Librosa MFCC Docs](https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html)

### DTW
- [DTW Wikipedia](https://en.wikipedia.org/wiki/Dynamic_time_warping)
- [DTW Algorithm Explained](https://www.youtube.com/watch?v=ERKDNmlPiNU) - YouTube

### Speech Recognition
- [CMU Sphinx Tutorial](https://cmusphinx.github.io/wiki/tutorial/) - Open source ASR
- [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech) - For reference

## Success Criteria

- [ ] Can load and visualize audio files
- [ ] Can extract MFCC features from audio
- [ ] DTW correctly measures similarity between audio clips
- [ ] Can record and store command templates
- [ ] Recognizes 10 commands with >80% accuracy
- [ ] Real-time recognition in <2 seconds
- [ ] Integrates with fish actuators

## Notes

- **Start simple**: Get audio loading working first
- **Visualize everything**: Plot waveforms, spectrograms, MFCCs
- **Debug with small examples**: Test DTW on identical audio (should be 0 distance)
- **Iterate on templates**: Record yourself saying commands multiple times
- **Threshold tuning**: Adjust DTW distance threshold for acceptance

## Limitations

This is **template-based recognition**, not deep learning ASR:

- Works best with same speaker as templates
- Limited vocabulary (10-50 commands works well)
- Sensitive to background noise
- Not suitable for continuous speech
- Requires recording templates for each command

For a fish with 10-20 commands, this is perfect and very educational!

---

**Ready?** Start with `audio/io.py` - let's get audio loading working.
