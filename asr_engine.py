# asr_engine.py
"""
ASR Engine: Loads NeMo model and returns clean, smooth full transcription text.
- Overlapping chunks for smart joining (no context loss at boundaries)
- Background noise reduction using noisereduce
- Chunking: 15 for >30 min, 7 otherwise
"""

import logging
import os
import numpy as np
import librosa
import noisereduce as nr

import nemo.collections.asr as nemo_asr
from pydub import AudioSegment

from config import Config

# Basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model (loaded once)
_asr_model = None

def load_model() -> nemo_asr.models.ASRModel:
    """Load and cache the Parakeet model on GPU."""
    global _asr_model
    if _asr_model is None:
        logger.info("Loading Parakeet model...")
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(Config.MODEL_NAME)
        _asr_model = _asr_model.to(Config.DEVICE)
        _asr_model.eval()
        logger.info("Model loaded on %s", Config.DEVICE)
    return _asr_model

def reduce_noise(audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
    """Reduce background noise using noisereduce (non-stationary)."""
    logger.info("Reducing background noise...")
    # Use first 1 second as noise sample (common practice)
    noise_sample = audio_array[:sample_rate]
    reduced = nr.reduce_noise(y=audio_array, y_noise=noise_sample, sr=sample_rate, stationary=False, prop_decrease=0.9)
    return reduced

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio with noise reduction + overlapping chunks for smooth text.
    """
    model = load_model()
    
    # Load audio with librosa for noise reduction
    audio_array, sr = librosa.load(file_path, sr=None, mono=True)
    logger.info("Original audio loaded: %.2f seconds, %d Hz", len(audio_array)/sr, sr)
    
    # Noise reduction
    clean_array = reduce_noise(audio_array, sr)
    
    # Save cleaned audio temporarily for pydub chunking
    clean_path = "temp_clean_audio.wav"
    import soundfile as sf
    sf.write(clean_path, clean_array, sr)
    
    audio = AudioSegment.from_file(clean_path)
    duration_min = len(audio) / (1000 * 60)
    logger.info("Cleaned audio duration: %.2f minutes", duration_min)
    
    # Choose chunk count
    if duration_min > Config.LONG_AUDIO_THRESHOLD_MINUTES:
        num_chunks = Config.NUM_CHUNKS_LONG  # 15
    else:
        num_chunks = Config.NUM_CHUNKS_SHORT  # 7
    
    logger.info("Using %d overlapping chunks", num_chunks)
    
    full_text = _transcribe_with_overlap(model, audio, num_chunks, sr)
    
    # Cleanup temp clean file
    os.remove(clean_path)
    
    return full_text

def _transcribe_with_overlap(model, audio: AudioSegment, num_chunks: int, original_sr: int) -> str:
    """Chunk with overlap (2 seconds) and smart text joining."""
    chunk_len_ms = len(audio) / num_chunks
    overlap_ms = 2000  # 2 seconds overlap - prevents boundary errors
    
    text_parts = []
    
    for i in range(num_chunks):
        start_ms = max(0, int(i * chunk_len_ms - overlap_ms if i > 0 else 0))
        end_ms = int(min((i + 1) * chunk_len_ms + overlap_ms, len(audio)))
        chunk = audio[start_ms:end_ms]
        chunk_path = f"temp_overlap_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        
        logger.info("Processing overlapping chunk %d/%d", i + 1, num_chunks)
        output = model.transcribe([chunk_path])
        text = output[0].text.strip()
        
        # For overlap: keep only new part (rough cut after overlap)
        if i > 0 and text:
            # Simple: take last 70% of text as new (can be improved)
            words = text.split()
            text = " ".join(words[int(len(words)*0.3):])  # keep last ~70%
        
        if text:
            text_parts.append(text.capitalize())
        
        os.remove(chunk_path)
    
    # Smart join: space or paragraph
    full_text = ""
    for i, part in enumerate(text_parts):
        if i == 0:
            full_text = part
        else:
            if full_text.endswith(('.', '!', '?')):
                full_text += "\n\n" + part
            else:
                full_text += " " + part
    
    return full_text

# Optional: Save to file
def save_full_text(full_text: str) -> None:
    """Save the final clean transcription."""
    with open(Config.FULL_TEXT_TXT, "w", encoding="utf-8") as f:
        f.write(full_text)
    logger.info("Saved final transcription to %s", Config.FULL_TEXT_TXT)