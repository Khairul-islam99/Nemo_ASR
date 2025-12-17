# NVIDIA Parakeet TDT 0.6B v3 ASR API

**A fast, production-ready Speech-to-Text (STT/ASR) API** built with NVIDIA's powerful **Parakeet-TDT-0.6b-v3** model.

This project provides a clean FastAPI web service that accepts audio file uploads and returns **high-quality, readable transcription text only** (no timestamps, no extra data).

### Key Features
- **Smart overlapping chunking** – Prevents broken sentences at chunk boundaries (15 chunks for >30 min audio, 7 chunks otherwise)
- **Background noise reduction** – Uses `noisereduce` for clearer audio and better accuracy
- **Smooth natural text flow** – Intelligent joining with proper paragraph breaks
- **GPU accelerated** – Extremely fast on NVIDIA GPUs
- **Multilingual support** – Auto-detects and transcribes 25 European languages (best on English)
- **Simple & clean output** – Returns only full transcription text + saves to `transcription.txt`
- **Interactive Swagger UI** – Test easily at `/docs`

Ideal for transcribing podcasts, audiobooks, interviews, meetings, narrations, or any long-form audio.

## Tech Stack
- **ASR Model**: NVIDIA Parakeet-TDT-0.6b-v3 (600M params)
- **Framework**: FastAPI + Uvicorn
- **Audio Processing**: pydub, librosa, noisereduce
- **Deep Learning**: NVIDIA NeMo Toolkit + PyTorch

## Requirements
- NVIDIA GPU (RTX 20xx or newer recommended)
- CUDA driver supporting 12.4+ (tested with CUDA 12.5 driver)
- Python 3.10+
- Conda (recommended)

## Installation

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/parakeet-asr-api.git
cd parakeet-asr-api
```
2. Create and activate conda environment
```bash
conda create -n nemo_asr python=3.10 -y
conda activate nemo_asr
```
3.Install PyTorch with CUDA support
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
4.Install project dependencies
```bash
pip install -r requirements.txt
```
## How to Run
Start the server:
```bash
python main.py
```
or (with auto-reload for development):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
### API will be available at:
Base URL: http://localhost:8000

Swagger UI (Interactive Docs): http://localhost:8000/docs

Health Check: http://localhost:8000/
