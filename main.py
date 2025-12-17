# main.py
"""
FastAPI application for Parakeet ASR transcription service.
Returns only the full transcription text.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException

from asr_engine import transcribe_audio, save_full_text  # save_full_text is optional
from config import Config

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NVIDIA Parakeet TDT 0.6B v3 ASR API",
    description="Multilingual speech-to-text transcription (only full text).",
    version="1.0.0"
)

@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": "Parakeet ASR API is running!",
        "model": Config.MODEL_NAME,
        "note": "Returns only full transcription text."
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and get only the full transcription text.
    Supported formats: wav, mp3, m4a, flac, ogg
    """
    if not file.filename.lower().endswith(Config.SUPPORTED_AUDIO_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Unsupported audio format.")

    temp_path = f"temp_upload_{file.filename}"
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Starting transcription for %s", file.filename)

        # Get only full text (no timestamps)
        full_text = transcribe_audio(temp_path)

        # Optional: Save to file on server
        save_full_text(full_text)

        return {
            "full_text": full_text.strip(),
            "file_saved": Config.FULL_TEXT_TXT  # Optional info
        }

    except Exception as e:
        logger.error("Transcription failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Transcription error")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)