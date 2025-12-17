# config.py
"""
Configuration settings for the Parakeet ASR API.
Only full transcription text (no timestamps).
"""

class Config:
    # Model
    MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"  # Parakeet TDT 0.6B v3
    
    # Device (GPU or CPU)
    DEVICE = "cuda"  # Change to "cpu" if no GPU
    
    # Chunking settings (as per your preference)
    LONG_AUDIO_THRESHOLD_MINUTES = 30   # If audio > 30 minutes → use more chunks
    NUM_CHUNKS_LONG = 15                # For >30 min audio: 15 chunks (faster processing)
    NUM_CHUNKS_SHORT = 7                # For ≤30 min audio: 7 chunks
    
    # Supported audio formats
    SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
    
    # Output file (only full text)
    FULL_TEXT_TXT = "transcription.txt"