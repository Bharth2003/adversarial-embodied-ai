import hashlib
import os
import logging
from typing import Optional
import soundfile as sf
from kokoro import KPipeline

logger = logging.getLogger(__name__)

class TTSManager:
    """
    Manages Text-To-Speech generation using the Kokoro AI model.
    Handles caching of generated audio files to avoid redundant processing.
    """
    def __init__(self, output_dir: str = "static/audio", lang_code: str = 'a', default_voice: str = 'af_bella'):
        self.output_dir = output_dir
        self.lang_code = lang_code
        self.default_voice = default_voice
        self.pipeline = None # Lazy initialization
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Created TTS output directory: {self.output_dir}")

    def _init_pipeline(self):
        """Lazy load the Kokoro pipeline."""
        if self.pipeline is None:
            logger.info("Initializing Kokoro TTS Pipeline...")
            # This will download the model and voices if not present
            self.pipeline = KPipeline(lang_code=self.lang_code)
            logger.info("Kokoro TTS Pipeline initialized.")

    def generate_audio(self, text: str, voice: Optional[str] = None) -> tuple[str, float]:
        """
        Generates audio for the given text and returns (filename, duration).
        Caches the result based on the text hash.
        """
        if not text:
            return "", 0.0
            
        voice = voice or self.default_voice
        # Create a unique hash for the text and voice
        text_hash = hashlib.md5(f"{text}|{voice}".encode()).hexdigest()
        filename = f"{text_hash}.wav"
        filepath = os.path.join(self.output_dir, filename)

        # If cached, we still need to get the duration
        if os.path.exists(filepath):
            info = sf.info(filepath)
            return filename, info.duration

        try:
            self._init_pipeline()
            # Kokoro generator yields (gs, ps, audio)
            generator = self.pipeline(
                text, voice=voice, 
                speed=1, split_pattern=r'\n+'
            )
            
            # Combine all chunks if multiple are generated
            import numpy as np
            all_audio = []
            for _, _, audio in generator:
                all_audio.append(audio)
            
            if not all_audio:
                logger.error(f"Kokoro failed to generate audio for text: {text}")
                return "", 0.0

            final_audio = np.concatenate(all_audio)
            sf.write(filepath, final_audio, 24000)
            
            duration = len(final_audio) / 24000.0
            logger.info(f"Generated and cached TTS: {filename} ({duration:.2f}s)")
            return filename, duration
            
        except Exception as e:
            logger.error(f"Error generating TTS for '{text}': {e}")
            return "", 0.0
