"""
Real-Time Audio Transcriber
Captures audio continuously and transcribes using Whisper.
"""

import sounddevice as sd
import numpy as np
import wave
import os
import threading
import time
import tempfile
from datetime import datetime
from collections import deque
from queue import Queue

# Optional: For Whisper transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper not installed. Install with: pip install openai-whisper")

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 5  # Process in 5-second chunks for near-real-time
OVERLAP_SECONDS = 1  # 1 second overlap between chunks


class RealtimeTranscriber:
    def __init__(self, model_size="base", chunk_seconds=5, overlap_seconds=1):
        self.sample_rate = SAMPLE_RATE
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.frame_size = int(self.sample_rate * 0.1)  # 100ms blocks
        self.buffer = deque()
        self.buffer_duration = 0
        self.recording = False
        self.transcription_queue = Queue()

        # Load Whisper model
        if WHISPER_AVAILABLE:
            print(f"Loading Whisper model: {model_size}...")
            self.model = whisper.load_model(model_size)
            print("Model loaded!")
        else:
            self.model = None

    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            print(f"Audio status: {status}")

        self.buffer.append(indata.copy())
        self.buffer_duration += len(indata) / self.sample_rate

        # Process when we have enough audio
        while self.buffer_duration >= self.chunk_seconds:
            self._process_chunk()

    def _process_chunk(self):
        """Extract chunk and send for transcription."""
        # Calculate frames needed
        frames_needed = int(self.chunk_seconds * self.sample_rate)
        overlap_frames = int(self.overlap_seconds * self.sample_rate)

        # Concatenate buffer
        audio_data = np.concatenate(list(self.buffer), axis=0).flatten()

        # Extract chunk (with overlap for next chunk)
        chunk = audio_data[:frames_needed]
        remainder = audio_data[frames_needed - overlap_frames:]

        # Update buffer
        self.buffer.clear()
        if len(remainder) > 0:
            self.buffer.append(remainder)
        self.buffer_duration = len(remainder) / self.sample_rate

        # Send for transcription
        if self.model:
            threading.Thread(
                target=self._transcribe,
                args=(chunk.copy(),),
                daemon=True
            ).start()

    def _transcribe(self, audio_chunk):
        """Transcribe audio chunk using Whisper."""
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_int16.tobytes())

            # Transcribe
            result = self.model.transcribe(
                temp_path,
                fp16=False,
                language="en"
            )

            text = result["text"].strip()
            if text:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {text}")
                self.transcription_queue.put({
                    "timestamp": timestamp,
                    "text": text
                })

            # Cleanup
            os.remove(temp_path)

        except Exception as e:
            print(f"Transcription error: {e}")

    def start(self):
        """Start real-time transcription."""
        if not self.model:
            print("Whisper not available. Install with: pip install openai-whisper")
            return

        self.recording = True
        print(f"Real-time transcription started")
        print(f"Processing chunks every {self.chunk_seconds}s")
        print("Speak now... Press Ctrl+C to stop\n")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                dtype=np.float32,
                blocksize=self.frame_size,
                callback=self.audio_callback
            ):
                while self.recording:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop transcription."""
        self.recording = False

        # Process remaining audio
        if self.buffer:
            audio_data = np.concatenate(list(self.buffer), axis=0).flatten()
            if len(audio_data) > self.sample_rate * 1:  # At least 1 second
                print("\nFinal transcription...")
                self._transcribe(audio_data)

        print("Transcription stopped.")


class BufferedTranscriber:
    """
    Alternative: Only transcribe after silence detected (complete utterances).
    More accurate but higher latency.
    """

    def __init__(self, model_size="base", silence_threshold=0.01, silence_duration=1.5):
        self.sample_rate = SAMPLE_RATE
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.silence_frames_needed = int(silence_duration * self.sample_rate / self.frame_size)

        if WHISPER_AVAILABLE:
            self.model = whisper.load_model(model_size)
        else:
            self.model = None

        self.buffer = []
        self.silence_count = 0
        self.recording = False

    def is_silence(self, audio_frame):
        """Check if frame is silence."""
        return np.sqrt(np.mean(audio_frame**2)) < self.silence_threshold

    def audio_callback(self, indata, frames, time_info, status):
        """Process audio frame."""
        audio = indata.flatten()

        if self.is_silence(audio):
            self.silence_count += 1
            self.buffer.append(audio)

            # If silence for long enough, process buffer
            if self.silence_count >= self.silence_frames_needed and len(self.buffer) > self.sample_rate * 1:
                self._transcribe_buffer()
                self.buffer = []
                self.silence_count = 0
        else:
            self.silence_count = 0
            self.buffer.append(audio)

    def _transcribe_buffer(self):
        """Transcribe current buffer."""
        if not self.model or len(self.buffer) < 10:  # Ignore very short
            return

        audio_data = np.concatenate(self.buffer)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            audio_int16 = (audio_data * 32767).astype(np.int16)
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())

        try:
            result = self.model.transcribe(temp_path, fp16=False, language="en")
            text = result["text"].strip()
            if text:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {text}")
        finally:
            os.remove(temp_path)

    def start(self):
        """Start buffered transcription."""
        if not self.model:
            print("Whisper not available")
            return

        self.recording = True
        print("Buffered transcription started (waits for silence)...")
        print("Press Ctrl+C to stop\n")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.sample_rate * 0.1),
                callback=self.audio_callback
            ):
                while self.recording:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop transcription."""
        self.recording = False
        if self.buffer:
            self._transcribe_buffer()
        print("Stopped.")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "realtime"

    if mode == "buffered":
        transcriber = BufferedTranscriber(model_size="base")
    else:
        transcriber = RealtimeTranscriber(model_size="base", chunk_seconds=5)

    transcriber.start()
