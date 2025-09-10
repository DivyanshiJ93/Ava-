"""
pipeline/summarize.py

Provides summarize_transcript(transcript, max_chunk_words=800) -> minutes_text

Uses a small summarization model (distilbart-cnn-12-6) on CPU by default,
chunking the transcript to avoid length limits.
"""

from typing import List
import math
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch

# Ensure CPU-only pipelines by setting device=-1 when creating pipeline
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

# Simple word-based chunker
def _chunk_text(text: str, max_words: int = 800) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def _clean_summary(s: str) -> str:
    # remove duplicated spaces and weird newlines
    s = re.sub(r'\s+\n', '\n', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    return s.strip()

def summarize_transcript(transcript: str, max_chunk_words: int = 800) -> str:
    """
    Summarize transcript into 'meeting minutes' style text.
    - Chunk transcript into manageable pieces
    - Summarize each chunk, then combine and summarize the combination
    """
    if not transcript or not transcript.strip():
        return ""

    chunks = _chunk_text(transcript, max_chunk_words)

    # Load summarization pipeline (CPU)
    # device=-1 ensures CPU
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=-1, truncation=True)

    partial_summaries = []
    for i, chunk in enumerate(chunks):
        # small models sometimes require short max_length; we keep conservative values
        try:
            out = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            text = out[0]["summary_text"]
        except Exception:
            # fallback naive: take first & last sentences
            sentences = re.split(r'(?<=[.!?]) +', chunk)
            text = " ".join(sentences[:3])  # naive fallback
        partial_summaries.append(text.strip())

    # Combine partial summaries and condense to a final minutes summary
    combined = "\n\n".join(partial_summaries)
    # A final pass summarization to unify tone
    try:
        final = summarizer(combined, max_length=180, min_length=60, do_sample=False)
        final_text = final[0]["summary_text"]
    except Exception:
        final_text = combined

    return _clean_summary(final_text)
