import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import time
import queue

class VoiceListener:
    def __init__(self, model_size="base", vad_threshold=0.01, silence_limit=1.5, pre_buffer_size=0.5, samplerate=16000):
        self.model_size = model_size
        self.vad_threshold = vad_threshold
        self.silence_limit = silence_limit
        self.pre_buffer_size = pre_buffer_size
        self.samplerate = samplerate
        
        self.model = None
        
        # State variables
        self.recording_buffer = []
        self.is_recording = False
        self.silence_start_time = None
        self.result_queue = queue.Queue()

    def _load_model(self):
        if self.model is None:
            print("Loading Whisper model...")
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Error: {status}")

        # Calculate Volume (RMS)
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))

        if volume_norm > self.vad_threshold:
            # SPEECH DETECTED
            if not self.is_recording:
                print(">>> Voice detected! Recording...")
                self.is_recording = True
            
            self.recording_buffer.append(indata.copy())
            self.silence_start_time = None  # Reset silence timer
        else:
            # SILENCE DETECTED
            if self.is_recording:
                self.recording_buffer.append(indata.copy())
                
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                
                # Check if silence duration exceeded limit
                if time.time() - self.silence_start_time > self.silence_limit:
                    self._process_audio()
                    self.is_recording = False
                    self.silence_start_time = None

    def _process_audio(self):
        if not self.recording_buffer:
            return

        print("Transcribing...")
        # Combine list of arrays into one NumPy array
        audio_data = np.concatenate(self.recording_buffer, axis=0).flatten()
        self.recording_buffer = [] # Clear buffer for next sentence

        # Run Whisper
        segments, _ = self.model.transcribe(audio_data, beam_size=5)
        
        full_text = " ".join([segment.text.strip() for segment in segments]).strip()
        
        if full_text:
            print(f"Result: [{full_text}]")
            self.result_queue.put(full_text)
        else:
            print("No speech transcribed.")

    def start(self):
        """Starts the background audio stream for continuous listening."""
        self._load_model()
        
        # Clear any old results
        while not self.result_queue.empty():
            self.result_queue.get()
            
        self.recording_buffer = []
        self.is_recording = False
        self.silence_start_time = None
        
        self._stream = sd.InputStream(samplerate=self.samplerate, channels=1, callback=self._audio_callback)
        self._stream.start()

    def stop(self):
        """Stops the background audio stream and processes any remaining audio."""
        if hasattr(self, "_stream") and self._stream is not None:
            # If we were in the middle of recording, trigger transcription for the last chunk
            if self.is_recording:
                self._process_audio()
                self.is_recording = False
                
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_text(self) -> str:
        """Non-blocking poll for transcribed text. Returns None if nothing is ready."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

    def listen_once(self) -> str:
        """
        Blocks until voice is detected and transcribed.
        Returns the transcribed text.
        (For standalone testing)
        """
        self.start()
        print(f"\nRobot is listening (Threshold: {self.vad_threshold})...")
        print("Speak now. Press Ctrl+C to stop.")

        try:
            while True:
                text = self.get_text()
                if text:
                    return text
                time.sleep(0.1)
        finally:
            self.stop()

# Main Execution (for standalone testing)
if __name__ == "__main__":
    try:
        listener = VoiceListener()
        while True:
            text = listener.listen_once()
            print(f"Got text: {text}")
            print("Ready for next sentence")
            time.sleep(1) # Small pause before starting next stream
    except KeyboardInterrupt:
        print("\nRobot shutting down.")