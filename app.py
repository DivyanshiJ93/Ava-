"""
Ava — AI Meeting Assistant (Dark + Hot Pink UI) with inline Settings card

This is the polished UI version with:
- No sidebar
- Black background, hot-pink accents
- Clean centered layout with cards
- Inline Settings card (model choice, summarization chunk size, action extraction toggle, minutes tone, timestamps)
- Same pipeline functionality (transcribe, summarize, extract actions)
"""

import streamlit as st
from pipeline.transcribe import transcribe_audio
from pipeline.summarize import summarize_transcript
from pipeline.extract_actions import extract_actions
from utils.io_helpers import timestamped_filename, ensure_dir

from pydub import AudioSegment
import tempfile, os, io, traceback, json, pandas as pd

# ---- Page config ----
st.set_page_config(page_title="Ava — AI Meeting Assistant", layout="wide")

# ---- Custom CSS (dark + hot pink theme) ----
st.markdown(
    """
    <style>
    body {
        background-color: #0d0d0d;
        color: #f5f5f5;
    }
    .main {
        background-color: #0d0d0d;
    }
    h1, h2, h3, h4 {
        color: #ff007f !important;
    }
    .card {
        background: #151515;
        border-radius: 14px;
        padding: 18px;
        margin-top: 18px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.6);
    }
    .stButton>button {
        background-color: #ff007f;
        color: white;
        border-radius: 28px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 700;
    }
    .stButton>button:hover {
        background-color: #e60073;
        color: white;
    }
    .stDownloadButton>button {
        background-color: #ff007f;
        color: white;
        border-radius: 20px;
        padding: 0.45rem 0.95rem;
        border: none;
        font-weight: 600;
    }
    .stDownloadButton>button:hover {
        background-color: #e60073;
    }
    .textarea, textarea {
        background-color: #0f0f10 !important;
        color: #f5f5f5 !important;
        border-radius: 10px;
    }
    .muted { color: #9ca3af; }
    .label { color: #f8f8f8; font-weight:600; }
    .small { font-size:0.95rem; color:#cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:14px;">
        <div style="width:56px; height:56px; border-radius:50%; background:#ff007f; 
                    display:flex; align-items:center; justify-content:center; 
                    font-weight:800; font-size:22px; color:#fff; box-shadow: 0 6px 18px rgba(255,0,127,0.12);">
            A
        </div>
        <div>
            <h1 style="margin-bottom:2px;">Ava — AI Meeting Assistant</h1>
            <div class="muted" style="margin-top:2px;">Upload meeting audio. I transcribe, summarize into minutes, and extract action items.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Inline Settings Card (no sidebar) ----
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex; justify-content:space-between; align-items:center;'>", unsafe_allow_html=True)
st.markdown("<div><div class='label'>Settings</div><div class='muted small'>Choose model & summary style</div></div>", unsafe_allow_html=True)
st.markdown("<div class='muted small'>Ava • CPU-only • Free models</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Layout: compact controls
col_a, col_b, col_c, col_d = st.columns([2.2, 1.2, 1.2, 1.6])
with col_a:
    model_choice = st.selectbox("Whisper model", options=["tiny.en", "base.en", "tiny", "base"], index=0, help="tiny.* are faster; base.* are more accurate (slower).")
with col_b:
    use_model_for_actions = st.selectbox("Action extraction", options=["Flan-T5 (recommended)", "Regex fallback only"], index=0)
    use_model_for_actions = True if use_model_for_actions.startswith("Flan") else False
with col_c:
    max_chunk_words = st.slider("Summary chunk size", min_value=200, max_value=1200, value=800, step=100, help="Smaller = faster but less context")
with col_d:
    include_timestamps = st.checkbox("Include timestamps in transcript", value=False)

# Minutes tone/style (second row)
st.markdown("<div style='margin-top:8px; display:flex; gap:12px; align-items:center;'>", unsafe_allow_html=True)
minutes_tone = st.selectbox("Minutes tone", options=["Concise (bullet-style)", "Detailed (paragraphs)", "Action-focused (highlight tasks)", "Executive (short summary)"], index=0)
custom_prefix = st.text_input("Optional prefix for minutes (e.g., 'Meeting Summary —')", value="", help="Text prefixed to the generated minutes")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)  # close card

# ---- File upload ----
st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload meeting audio (.mp3, .wav, .m4a)", type=["mp3", "wav", "m4a"])

# ---- Generate button ----
generate_col1, generate_col2 = st.columns([0.8, 0.2])
with generate_col1:
    if st.button("Generate Minutes ✨"):
        if not uploaded_file:
            st.error("Please upload an audio file first.")
        else:
            wav_path = None
            try:
                with st.spinner("Converting audio..."):
                    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
                    tmp_in.write(uploaded_file.getvalue())
                    tmp_in.flush()
                    tmp_in.close()
                    audio = AudioSegment.from_file(tmp_in.name)
                    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    audio.export(tmp_out.name, format="wav")
                    wav_path = tmp_out.name
                    try:
                        os.unlink(tmp_in.name)
                    except Exception:
                        pass
                    duration_sec = len(audio) / 1000.0
                    st.success(f"Loaded: {uploaded_file.name} — {duration_sec:.1f}s")
            except Exception as e:
                st.error("Audio conversion failed: " + str(e))
                wav_path = None

            if wav_path:
                try:
                    # Transcription (with timestamps optionally)
                    with st.spinner("Transcribing..."):
                        trans_res = transcribe_audio(wav_path, model=model_choice)
                        raw_text = trans_res.get("text", "") or ""
                        segments = trans_res.get("segments", []) or []

                        if include_timestamps and segments:
                            # build a timestamped transcript
                            lines = []
                            for seg in segments:
                                start = int(seg.get("start", 0))
                                m, s = divmod(start, 60)
                                ts = f"[{m:02d}:{s:02d}]"
                                lines.append(f"{ts} {seg.get('text','').strip()}")
                            transcript_text = "\n".join(lines)
                        else:
                            transcript_text = raw_text

                        if not transcript_text.strip():
                            st.warning("Transcription returned no text. Try a different model.")
                        st.session_state.transcript = transcript_text

                    # Summarization (pass chunk size)
                    with st.spinner("Summarizing..."):
                        # We don't have a 'tone' parameter in the summarizer; we can slightly modify behavior by
                        # including the style as a simple prompt-like prefix in the transcript for the model pipeline,
                        # but summarizer here is a simple pipeline — so we'll post-process prefix for clarity.
                        summary_text = summarize_transcript(st.session_state.transcript, max_chunk_words=max_chunk_words)
                        # Apply tone adjustments (light heuristics)
                        if minutes_tone.startswith("Concise"):
                            # keep as-is (already concise)
                            final_summary = summary_text
                        elif minutes_tone.startswith("Detailed"):
                            final_summary = summary_text + "\n\n(Details: " + "See full transcript above.)"
                        elif minutes_tone.startswith("Action"):
                            final_summary = "**Action-focused minutes**\n\n" + summary_text
                        else:  # Executive
                            # take first 2 sentences as executive highlight (naive)
                            sentences = summary_text.split(". ")
                            final_summary = ". ".join(sentences[:2]).strip()
                            if not final_summary.endswith("."):
                                final_summary += "."
                        if custom_prefix:
                            final_summary = f"{custom_prefix}\n\n{final_summary}"
                        st.session_state.summary = final_summary

                    # Action extraction
                    with st.spinner("Extracting action items..."):
                        actions = extract_actions(st.session_state.transcript, use_model=use_model_for_actions)
                        st.session_state.actions = actions

                    st.success("Done — see results below.")
                except Exception as e:
                    st.error("Processing failed: " + str(e))
                    st.text(traceback.format_exc())
                finally:
                    try:
                        os.unlink(wav_path)
                    except Exception:
                        pass

with generate_col2:
    st.write("")  # spacer

# ---- Results area ----
if "transcript" in st.session_state and st.session_state.transcript:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Transcript")
    st.session_state.transcript = st.text_area("", value=st.session_state.transcript, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

if "summary" in st.session_state and st.session_state.summary:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Meeting Minutes")
    st.markdown(st.session_state.summary)
    st.markdown("</div>", unsafe_allow_html=True)

if "actions" in st.session_state and st.session_state.actions:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Action Items")
    df = pd.DataFrame(st.session_state.actions)
    for col in ["id", "action", "owner", "deadline", "context"]:
        if col not in df.columns:
            df[col] = None
    st.dataframe(df[["id", "action", "owner", "deadline", "context"]], height=240)
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Downloads ----
if any([st.session_state.get("transcript"), st.session_state.get("summary"), st.session_state.get("actions")]):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Download Results")
    ensure_dir("outputs")
    if st.session_state.get("transcript"):
        st.download_button("Download Transcript (.txt)", data=st.session_state["transcript"].encode("utf-8"),
                           file_name=timestamped_filename("transcript", "txt"), mime="text/plain")
    if st.session_state.get("summary"):
        st.download_button("Download Minutes (.md)", data=st.session_state["summary"].encode("utf-8"),
                           file_name=timestamped_filename("minutes", "md"), mime="text/markdown")
    if st.session_state.get("actions"):
        st.download_button("Download Action Items (.json)", data=json.dumps(st.session_state["actions"], indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name=timestamped_filename("action_items", "json"), mime="application/json")
    st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("<div style='margin-top:18px; text-align:center; color:#9ca3af;'>Tip: For long meetings, split files into smaller chunks. Use base models for better accuracy (slower).</div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#666; margin-top:8px;'>Built with ❤️ using open-source Whisper & Transformers</div>", unsafe_allow_html=True)
