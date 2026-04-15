"""
Continuous Audio Recorder
Records audio continuously in chunks, optionally using VAD to skip silence.
"""

import sounddevice as sd
import numpy as np
import wave
import os
import threading
import time
from datetime import datetime
from collections import deque

# Configuration
SAMPLE_RATE = 16000      # 16kHz is standard for speech recognition
CHANNELS = 1             # Mono
CHUNK_DURATION = 30      # Seconds per file
DEVICE = None            # None = default device

def generate_filename():
    """Generate timestamped filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"recording_{timestamp}.wav"

def save_wav(filename, audio_data, sample_rate):
    """Save numpy array as WAV file."""
    # Convert float32 to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"Saved: {filename} ({len(audio_data)/sample_rate:.1f}s)")

class ContinuousRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS,
                 chunk_duration=CHUNK_DURATION, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.device = device
        self.recording = False
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.thread = None
        self.frames_per_chunk = int(sample_rate * chunk_duration)

    def audio_callback(self, indata, frames, time_info, status):
        """Called for each audio block (runs in separate thread)."""
        if status:
            print(f"Audio status: {status}")

        with self.buffer_lock:
            self.audio_buffer.append(indata.copy())

            # Check if buffer has enough for a chunk
            total_frames = sum(len(buf) for buf in self.audio_buffer)
            if total_frames >= self.frames_per_chunk:
                self._save_chunk()

    def _save_chunk(self):
        """Concatenate buffer and save to file."""
        # Concatenate all buffers
        audio_data = np.concatenate(self.audio_buffer, axis=0)

        # Extract chunk and keep remainder
        chunk = audio_data[:self.frames_per_chunk]
        remainder = audio_data[self.frames_per_chunk:]

        # Update buffer with remainder
        if len(remainder) > 0:
            self.audio_buffer = [remainder]
        else:
            self.audio_buffer = []

        # Save in background thread to avoid blocking
        filename = generate_filename()
        threading.Thread(
            target=save_wav,
            args=(filename, chunk.flatten(), self.sample_rate),
            daemon=True
        ).start()

    def start(self):
        """Start continuous recording."""
        self.recording = True
        self.audio_buffer = []

        print(f"Starting recording...")
        print(f"Sample rate: {self.sample_rate}Hz")
        print(f"Chunk duration: {self.chunk_duration}s")
        print(f"Device: {self.device or 'default'}")
        print("Press Ctrl+C to stop\n")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                device=self.device,
                blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
                callback=self.audio_callback
            ):
                while self.recording:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop recording and save any remaining audio."""
        self.recording = False

        with self.buffer_lock:
            if self.audio_buffer:
                audio_data = np.concatenate(self.audio_buffer, axis=0)
                if len(audio_data) > self.sample_rate * 1:  # Save if > 1 second
                    filename = generate_filename()
                    save_wav(filename, audio_data.flatten(), self.sample_rate)

        print("Recording stopped.")

def list_devices():
    """List available audio input devices."""
    print("Available input devices:")
    print(sd.query_devices())

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--list-devices":
        list_devices()
    else:
        recorder = ContinuousRecorder(
            sample_rate=16000,
            channels=1,
            chunk_duration=30  # Save every 30 seconds
        )
        recorder.start()
