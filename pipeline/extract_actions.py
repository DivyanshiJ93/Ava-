"""
pipeline/extract_actions.py

Provides extract_actions(transcript) -> list of dicts:
[ {"id": 1, "action": "...", "owner": "...", "deadline": "...", "context": "..."}, ... ]

Primary method: use google/flan-t5-small (text2text-generation) via transformers.
If model output can't be parsed to JSON, fallback to regex-based heuristics.
This module is defensive at import time (no heavy work runs on import).
"""

import json
import re
from typing import List, Dict, Any
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

FLAN_MODEL = "google/flan-t5-small"

# Keywords to look for in regex fallback
ACTION_KEYWORDS = [
    "action", "todo", "to do", "will", "should", "please", "assign", "deadline", "by", "due", "follow up", "follow-up"
]

def _build_prompt_for_flan(transcript: str) -> str:
    instruction = (
        "You are an assistant that reads meeting transcripts and extracts action items. "
        "Output a JSON array where each item has: id (int), action (short action text), "
        "owner (person or null), deadline (date or text or null), context (the sentence from the transcript). "
        "If there are no action items, output an empty JSON array: [].\n\n"
        "Transcript:\n"
    )
    return instruction + transcript.strip()

def _parse_model_output_to_json(text: str) -> List[Dict[str, Any]]:
    """
    Try to safely extract a JSON array from model text output.
    """
    # First try direct load
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # Try to find first '[' and last ']' and parse substring
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1 and end > start:
        substr = text[start:end+1]
        try:
            data = json.loads(substr)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    # As last resort, return empty list to indicate parse failure
    return []

def _regex_fallback(transcript: str) -> List[Dict[str, Any]]:
    """
    Simple heuristics: split into sentences, pick sentences containing keywords
    or that look like imperatives (start with verb).
    """
    # Normalize
    s = transcript.replace("\n", " ")
    # Split into sentences (naive)
    sentences = re.split(r'(?<=[.!?]) +', s)
    actions = []
    id_counter = 1

    # Pattern for sentences that begin with common imperative verbs or polite requests
    verb_start_pattern = re.compile(
        r'^(?:Please |please |(?:Add|Assign|Follow|Follow up|Follow-up|Call|Email|Schedule|Create|Prepare|Send|Complete|Finish|Investigate|Confirm|Book|Plan|Share|Provide|Discuss|Setup|Set up|Make|Organize|Arrange)\b)',
        re.IGNORECASE
    )

    for sent in sentences:
        text = sent.strip()
        if not text:
            continue
        lowered = text.lower()
        # if contains keywords or looks imperative
        if any(k in lowered for k in ACTION_KEYWORDS) or verb_start_pattern.search(text):
            # Attempt to detect owner (simple heuristics)
            owner = None
            deadline = None

            # by <name/date>
            m_by = re.search(r'\bby ([A-Z][\w\s\-\']+|\d{1,2}(?:st|nd|rd|th)? [A-Za-z]+|\d{4}|\w+)\b', text)
            if m_by:
                candidate = m_by.group(1).strip()
                # detect if candidate looks like a date (month name or year)
                if re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', candidate, re.IGNORECASE) or re.search(r'\d{4}', candidate):
                    deadline = candidate
                else:
                    owner = candidate

            # "<Name> will ..." pattern
            m_owner = re.search(r'([A-Z][a-z]+(?: [A-Z][a-z]+)*) will\b', text)
            if m_owner:
                owner = m_owner.group(1)

            actions.append({
                "id": id_counter,
                "action": text if len(text) < 1000 else text[:1000],
                "owner": owner if owner else None,
                "deadline": deadline if deadline else None,
                "context": text
            })
            id_counter += 1

    return actions

def extract_actions(transcript: str, use_model: bool = True) -> List[Dict[str, Any]]:
    """
    Extract action items from a transcript.
    Tries model-based extraction first (flan-t5-small). If model output can't be parsed, uses regex fallback.

    Parameters:
      - transcript: full meeting transcript text
      - use_model: whether to try the Flan-T5 model (defaults to True). Set False to force regex fallback.

    Returns:
      List of normalized action dicts: {id, action, owner, deadline, context}
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return []

    # If user disabled model use, go regex-only
    if not use_model:
        return _regex_fallback(transcript)

    prompt = _build_prompt_for_flan(transcript)
    try:
        # device=-1 ensures CPU
        generator = pipeline("text2text-generation", model=FLAN_MODEL, device=-1)
        resp = generator(prompt, max_length=512, do_sample=False)
        raw = resp[0].get("generated_text", "") if isinstance(resp, list) else str(resp)
        items = _parse_model_output_to_json(raw)
        if not items:
            # Try another heuristic: sometimes the model returns plain text with JSON embedded
            items = _parse_model_output_to_json(raw)
        if not items:
            # fallback to regex heuristics
            items = _regex_fallback(transcript)
    except Exception as e:
        logger.exception("Flan-T5 extraction failed, using regex fallback: %s", e)
        items = _regex_fallback(transcript)

    # Normalize to required keys and ensure sequential ids
    normalized = []
    for idx, it in enumerate(items, start=1):
        if isinstance(it, dict):
            action_text = it.get("action") or it.get("action_text") or it.get("task") or str(it)
            owner = it.get("owner") if it.get("owner") else None
            deadline = it.get("deadline") if it.get("deadline") else None
            context = it.get("context") or it.get("sentence") or action_text
        else:
            action_text = str(it)
            owner = None
            deadline = None
            context = action_text

        normalized.append({
            "id": idx,
            "action": action_text,
            "owner": owner,
            "deadline": deadline,
            "context": context
        })

    return normalized

# If this module is run directly, provide a tiny example
if __name__ == "__main__":  # pragma: no cover
    sample = (
        "Alice: We'll publish the Q3 report by Friday. "
        "Bob: I will draft the report. "
        "Carol: Please review and send feedback by Wednesday."
    )
    print("Regex fallback result:")
    print(json.dumps(_regex_fallback(sample), indent=2))
    print("\nAttempting model extraction (may download model):")
    try:
        print(json.dumps(extract_actions(sample, use_model=False), indent=2))
    except Exception as ex:
        print("Model extraction skipped/failure:", ex)
