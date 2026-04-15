# Audio Recognition System

Real-time audio capture implementations without keyword detection.

## Options

### 1. Continuous Recorder (`continuous_recorder.py`)
Records everything in fixed-time chunks (e.g., 30-second files).

```bash
python continuous_recorder.py
python continuous_recorder.py --list-devices  # See available mics
```

**Best for:** Archiving all audio, security recording
**Storage:** High (records silence too)

---

### 2. VAD Recorder (`vad_recorder.py`) ⭐ **Recommended**
Records only when voice/sound is detected using Voice Activity Detection.

```bash
# Install WebRTC VAD (optional but recommended)
pip install webrtcvad

python vad_recorder.py
python vad_recorder.py --calibrate  # Set threshold for your environment
```

**Best for:** Long-running capture, meeting notes, voice memos
**Storage:** Low (silence skipped)
**Features:**
- 500ms pre-buffer (captures speech start)
- 500ms post-buffer (captures speech end)
- Auto-splits on silence
- WebRTC VAD or energy-based fallback

---

### 3. Real-Time Transcriber (`realtime_transcriber.py`)
Captures and transcribes speech in real-time using Whisper.

```bash
# Install Whisper
pip install openai-whisper

# Real-time mode (5-second sliding window)
python realtime_transcriber.py

# Buffered mode (waits for silence, more accurate)
python realtime_transcriber.py buffered
```

**Best for:** Live captioning, dictation, command recognition
**Storage:** None (text output only), optional audio save
**Models:** tiny/base/small/medium/large (trade speed vs accuracy)

---

## Comparison

| Feature | Continuous | VAD | Transcriber |
|---------|-----------|-----|-------------|
| Records Silence | Yes | No | Configurable |
| Storage | High | Low | Minimal |
| Latency | None | ~1s | 5-10s |
| CPU Usage | Low | Low | High |
| Best For | Archiving | Voice memos | Live captioning |

## Requirements

```bash
pip install sounddevice numpy

# Optional but recommended
pip install webrtcvad  # For VAD recorder
pip install openai-whisper  # For transcriber
```

## Hardware Setup

- **Microphone:** Any USB mic or built-in
- **Sample Rate:** 16kHz (optimal for speech)
- **Format:** Mono, 16-bit

## Advanced: Custom Processing

To add your own processing pipeline:

```python
import sounddevice as sd
import numpy as np

def custom_callback(indata, frames, time, status):
    """Your processing logic here."""
    audio = indata.flatten()

    # Example: Detect loud sounds
    volume = np.sqrt(np.mean(audio**2))
    if volume > 0.1:
        print(f"Loud sound detected! {volume:.3f}")

    # Example: Run your model
    # prediction = my_model.predict(audio)

with sd.InputStream(callback=custom_callback):
    while True:
        pass
```

## Next Steps

1. **Try the VAD recorder first** - best balance of utility and efficiency
2. **Add speaker diarization** - identify who is speaking
3. **Integrate with LLM** - process transcriptions with GPT/Claude
4. **Add wake word** - trigger specific actions on custom phrases
