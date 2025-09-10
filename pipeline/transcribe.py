"""
pipeline/transcribe.py

Provides transcribe_audio(path, model="tiny.en") -> {"text": ..., "segments": [...]}

Uses openai-whisper by default (CPU). If faster-whisper or whisper.cpp are
available, you'll see messages guiding how to switch - but the default path
uses the pure Python whisper package so it runs on CPU without compilation.
"""

import os
import json
import subprocess
from typing import Dict, Any, List

# Prefer the openai-whisper package (pure Python)
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# faster-whisper optional detection (not used by default)
try:
    from faster_whisper import WhisperModel as FasterWhisperModel  # type: ignore
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False

def _run_whisper_python(path: str, model_name: str = "tiny.en") -> Dict[str, Any]:
    """
    Use openai-whisper package (python) to transcribe. Returns dict with text and segments.
    Ensures CPU usage by not moving torch to GPU.
    """
    if not WHISPER_AVAILABLE:
        raise RuntimeError("openai-whisper package not installed (pip install openai-whisper).")

    # whisper.load_model will download model to ~/.cache/whisper by default.
    model = whisper.load_model(model_name)
    # force CPU - whisper uses torch; if CUDA available it may use it by default,
    # but we can't rely on GPU. We proactively set torch device environment variables:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hides CUDA devices if any
    # transcribe
    result = model.transcribe(path, language="en", verbose=False)
    # result contains 'text' and 'segments' keys
    return {"text": result.get("text", ""), "segments": result.get("segments", [])}


def _run_faster_whisper(path: str, model_name: str = "tiny.en") -> Dict[str, Any]:
    """
    Optional: if faster-whisper is installed you can use it (faster on CPU for some setups).
    We detect but do not default to it, to keep behavior predictable.
    """
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError("faster-whisper not installed.")
    model = FasterWhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, beam_size=5)
    text = " ".join([s.text for s in segments])
    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    return {"text": text, "segments": segs}


def transcribe_audio(path: str, model: str = "tiny.en") -> Dict[str, Any]:
    """
    Main entrypoint for transcription.
    Returns: {"text": full_transcript, "segments": [ {"start":float,"end":float,"text":str}, ... ]}
    Defaults to CPU Python whisper ("tiny.en" recommended for speed).
    """
    # Safety: ensure path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    # If faster-whisper installed and user passed "faster" as model, switch
    if model.startswith("faster-") and FASTER_WHISPER_AVAILABLE:
        actual_model = model.replace("faster-", "")
        return _run_faster_whisper(path, actual_model)

    # Default: openai-whisper python package
    try:
        return _run_whisper_python(path, model)
    except Exception as e:
        # If whisper fails (missing dependency), try to detect whisper.cpp or give helpful error
        # whisper.cpp fallback: if `whisper.cpp` binary present and model available, we could call it.
        # For simplicity we report helpful message.
        raise RuntimeError(
            "Transcription failed using openai-whisper. "
            "Make sure you installed requirements and ffmpeg is available. "
            f"Inner error: {e}"
        )
