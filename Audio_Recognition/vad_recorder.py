"""
Voice Activity Detection (VAD) Recorder
Records only when voice/sound is detected. More efficient than continuous.
"""

import sounddevice as sd
import numpy as np
import wave
import os
import threading
import time
from datetime import datetime
from collections import deque

# Try to import webrtcvad, fallback to energy-based detection
try:
    import webrtcvad
    HAS_WEBRTC_VAD = True
    print("Using WebRTC VAD")
except ImportError:
    HAS_WEBRTC_VAD = False
    print("WebRTC VAD not available, using energy-based detection")

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_AGGRESSIVENESS = 1  # 0-3 (0 = least aggressive, 3 = most aggressive)

# Timing
FRAME_DURATION_MS = 30  # 10, 20, or 30ms (WebRTC requirement)
PRE_BUFFER_MS = 500    # Keep 500ms before speech detected
POST_BUFFER_MS = 500   # Keep 500ms after speech ends
MIN_SPEECH_MS = 300    # Minimum speech to save


class VADRecorder:
    def __init__(self, sample_rate=16000, aggressiveness=1):
        self.sample_rate = sample_rate
        self.frame_duration_ms = 30
        self.frame_size = int(sample_rate * self.frame_duration_ms / 1000)

        if HAS_WEBRTC_VAD:
            self.vad = webrtcvad.Vad(aggressiveness)
        else:
            self.vad = None
            self.energy_threshold = 0.01  # Adjust based on your mic

        self.recording = False
        self.audio_buffer = deque(maxlen=int(PRE_BUFFER_MS / self.frame_duration_ms))
        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speech_active = False

    def is_speech(self, audio_frame):
        """Check if audio frame contains speech."""
        if HAS_WEBRTC_VAD:
            # WebRTC VAD requires specific frame sizes
            pcm_bytes = (audio_frame * 32767).astype(np.int16).tobytes()
            return self.vad.is_speech(pcm_bytes, self.sample_rate)
        else:
            # Energy-based fallback
            energy = np.sqrt(np.mean(audio_frame**2))
            return energy > self.energy_threshold

    def process_frame(self, audio_frame):
        """Process a single audio frame."""
        speech_detected = self.is_speech(audio_frame)

        if speech_detected:
            if not self.is_speech_active:
                # Speech started - flush pre-buffer
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Speech detected")
                self.speech_buffer = list(self.audio_buffer)
                self.is_speech_active = True
            self.silence_frames = 0
            self.speech_buffer.append(audio_frame)
        else:
            if self.is_speech_active:
                self.silence_frames += 1
                self.speech_buffer.append(audio_frame)

                # Check if silence duration exceeds post-buffer
                silence_duration_ms = self.silence_frames * self.frame_duration_ms
                if silence_duration_ms > POST_BUFFER_MS:
                    self._finalize_segment()

        # Always add to pre-buffer
        self.audio_buffer.append(audio_frame)

    def _finalize_segment(self):
        """Save the current speech segment."""
        speech_duration_ms = (len(self.speech_buffer) - self.silence_frames) * self.frame_duration_ms

        if speech_duration_ms >= MIN_SPEECH_MS:
            # Trim trailing silence
            trimmed = self.speech_buffer[:-self.silence_frames]
            audio_data = np.concatenate(trimmed)

            filename = f"speech_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
            self._save_wav(filename, audio_data)
            print(f"Saved: {filename} ({speech_duration_ms/1000:.1f}s)")
        else:
            print(f"Skipped short segment ({speech_duration_ms}ms)")

        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speech_active = False

    def _save_wav(self, filename, audio_data):
        """Save audio to WAV file."""
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())

    def audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status:
            print(f"Status: {status}")

        # Split into VAD frames
        audio = indata.flatten()
        for i in range(0, len(audio) - self.frame_size + 1, self.frame_size):
            frame = audio[i:i + self.frame_size]
            if len(frame) == self.frame_size:
                self.process_frame(frame)

    def start(self):
        """Start VAD-based recording."""
        self.recording = True
        print(f"VAD Recording started")
        print(f"Sample rate: {self.sample_rate}Hz")
        print(f"Pre-buffer: {PRE_BUFFER_MS}ms")
        print(f"Post-buffer: {POST_BUFFER_MS}ms")
        print("Listening for speech... Press Ctrl+C to stop\n")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self.frame_size * 10,  # Process 10 frames at a time
                callback=self.audio_callback
            ):
                while self.recording:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()

    def stop(self):
        """Stop and save any pending audio."""
        self.recording = False
        if self.is_speech_active and self.speech_buffer:
            self._finalize_segment()
        print("Recording stopped.")


def calibrate_threshold(seconds=3):
    """Calibrate energy threshold based on ambient noise."""
    print(f"Calibrating... please be silent for {seconds} seconds")

    noise_samples = []
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        noise_samples.append(np.sqrt(np.mean(indata**2)))

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype=np.float32, callback=callback):
        time.sleep(seconds)

    avg_noise = np.mean(noise_samples)
    threshold = avg_noise * 3  # 3x ambient
    print(f"Ambient noise level: {avg_noise:.4f}")
    print(f"Suggested threshold: {threshold:.4f}")
    return threshold


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate":
        calibrate_threshold()
    else:
        recorder = VADRecorder(aggressiveness=VAD_AGGRESSIVENESS)
        recorder.start()
