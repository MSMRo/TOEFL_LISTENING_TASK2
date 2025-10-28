# app.py
import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from gtts import gTTS
from pydub import AudioSegment

# ---- silence creation (works with any pydub version) ----
try:
    from pydub.generators import Silence
    def make_silence(duration_ms: int) -> AudioSegment:
        return Silence(duration=duration_ms).to_audio_segment()
except Exception:
    def make_silence(duration_ms: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration_ms)

# =============== UI & parsing ===============

st.set_page_config(page_title="Multi-Voice Role Reader (gTTS)", page_icon="🗣️", layout="wide")
st.title("🗣️ Multi-Voice Role Reader — gTTS edition")

LINE_RE = re.compile(r'^\s*(?P<role>[^:]+)\s*:\s*(?P<text>.+?)\s*$', re.IGNORECASE)
PAUSE_TAG_RE = re.compile(r'\[pause\s*=\s*(\d+)\s*ms\]', re.IGNORECASE)

SAMPLE_DIALOGUE = """Narrator: Welcome! Paste a dialogue below. Each line must be 'Role: text'.
Professor: Good morning. Today we discuss how to differentiate signals from noise.
Student: Professor, how do we control for baseline wander in ECG?
Professor: Great question. We apply high-pass filtering and careful detrending.
Student: Thank you! And what cutoff is typical? [pause=300ms]
Professor: Around 0.5 Hz to remove slow drifts without distorting the PQRST complexes.
"""

def parse_dialogue(text: str) -> List[Tuple[str, str]]:
    segs: List[Tuple[str, str]] = []
    for raw in text.strip().splitlines():
        if not raw.strip():
            continue
        m = LINE_RE.match(raw)
        if m:
            segs.append((m.group('role').strip(), m.group('text').strip()))
    return segs

def extract_pause_chunks(text: str) -> List[Tuple[str, Optional[int]]]:
    """
    Returns [(chunk_text, pause_ms_after_chunk)] based on [pause=###ms] tags.
    """
    pos = 0
    chunks: List[Tuple[str, Optional[int]]] = []
    buffer = []
    for m in PAUSE_TAG_RE.finditer(text):
        prefix = text[pos:m.start()]
        buffer.append(prefix)
        chunk_text = "".join(buffer).strip()
        pause_ms = int(m.group(1))
        if chunk_text:
            chunks.append((chunk_text, pause_ms))
        buffer = []
        pos = m.end()
    buffer.append(text[pos:])
    tail = "".join(buffer).strip()
    if tail:
        chunks.append((tail, None))
    return chunks if chunks else [(text, None)]

# =============== gTTS Synthesis ===============

@dataclass
class GTTSVoice:
    lang: str = "en"      # language code, e.g., 'en', 'en-us', 'es'
    tld: str = "com"      # top-level domain to influence accent: 'com', 'co.uk', 'com.au', 'co.in', 'com.mx', etc.
    slow: bool = False    # slower reading (True) or normal (False)
    pitch_semitones: float = 0.0  # optional pitch approximation via resampling

def gtts_tts_bytes(text: str, lang: str = "en", tld: str = "com", slow: bool = False) -> bytes:
    """
    Synthesize using gTTS and return MP3 bytes (in-memory).
    """
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def apply_pitch_shift(seg: AudioSegment, semitones: float) -> AudioSegment:
    """
    Approximate pitch shift by resampling (changes pitch and slightly affects timbre/time).
    For small shifts (±2–4 semitones) it’s often acceptable.
    """
    if abs(semitones) < 0.01:
        return seg
    factor = 2.0 ** (semitones / 12.0)
    new_frame_rate = int(seg.frame_rate * factor)
    shifted = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
    # resample back to original frame rate to keep a consistent output rate
    return shifted.set_frame_rate(seg.frame_rate)

def synthesize_dialogue_gtts(
    segments: List[Tuple[str, str]],
    role_to_voice: Dict[str, GTTSVoice],
    interturn_pause_ms: int,
    output_format: str = "mp3",   # "mp3" or "wav"
) -> bytes:
    """
    For each (role, text), synthesize with that role's GTTSVoice settings.
    Supports inline [pause=###ms]. Concatenates to one audio and returns bytes.
    """
    final = AudioSegment.silent(duration=0)
    for idx, (role, text) in enumerate(segments):
        v = role_to_voice.get(role)
        if not v:
            raise RuntimeError(f"No gTTS config for role '{role}'.")

        # break the line into chunks with inline pauses
        chunks = extract_pause_chunks(text)
        part = AudioSegment.silent(duration=0)
        for chunk_text, pause_ms in chunks:
            mp3_bytes = gtts_tts_bytes(chunk_text, lang=v.lang, tld=v.tld, slow=v.slow)
            seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
            seg = apply_pitch_shift(seg, v.pitch_semitones)
            part += seg
            if pause_ms and pause_ms > 0:
                part += make_silence(pause_ms)

        final += part
        if interturn_pause_ms and interturn_pause_ms > 0 and idx < len(segments) - 1:
            final += make_silence(interturn_pause_ms)

    buf = io.BytesIO()
    if output_format.lower() == "wav":
        final.export(buf, format="wav")
        mime = "audio/wav"
        filename = "dialogue_roles.wav"
    else:
        final.export(buf, format="mp3", bitrate="192k")
        mime = "audio/mpeg"
        filename = "dialogue_roles.mp3"
    buf.seek(0)
    return buf.read(), mime, filename

# =============== UI ===============

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
- Paste your dialog using **`Role: text`** lines.
- For each role, choose **gTTS** settings:
  - **lang** (language code),
  - **tld** (accent domain: `com`, `co.uk`, `com.au`, `co.in`, `com.mx`, etc.),
  - **slow** (reading speed),
  - **pitch shift** (approximate male/female timbre via resampling).
- Use inline pauses like `[pause=300ms]` and the *Pause between turns* slider.
- Click **Synthesize** to play/download a single audio file.
    """)

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("1) Paste your dialogue")
    text_input = st.text_area(
        "Each line: Role: text",
        height=260,
        value=SAMPLE_DIALOGUE,
        placeholder="Professor: Hello class...\nStudent: I have a question...\n...",
    )

segments = parse_dialogue(text_input)
roles = sorted({r for (r, _) in segments})

with right:
    st.subheader("2) Global parameters")
    interturn_pause_ms = st.slider("Pause between turns (ms)", 0, 2000, 250, 50)
    out_fmt = st.selectbox("Output format", ["mp3", "wav"], index=0)

st.subheader("3) Role → gTTS settings")

# sensible defaults: Student → slightly higher pitch, Professor → slightly lower
DEFAULTS = {
    "student": GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=+2.0),
    "professor": GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=-2.0),
    "narrator": GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=0.0),
}

# per-role UI
role_to_voice: Dict[str, GTTSVoice] = {}
if roles:
    # common language/tld presets (extend as needed)
    common_langs = [
        ("English (US)", "en", ["com", "com.au", "co.uk", "co.in"]),
        ("English (UK)", "en", ["co.uk", "com"]),
        ("Spanish (ES)", "es", ["es", "com"]),
        ("Spanish (MX/LatAm)", "es", ["com.mx", "com"]),
        ("Portuguese (BR)", "pt", ["com.br", "com"]),
        ("French (FR)", "fr", ["fr", "com"]),
    ]

    cols = st.columns(min(3, len(roles)) or 1)
    for i, role in enumerate(roles):
        with cols[i % len(cols)]:
            rlow = role.lower()
            if "student" in rlow:
                d = DEFAULTS["student"]
            elif "prof" in rlow:
                d = DEFAULTS["professor"]
            elif "narrator" in rlow:
                d = DEFAULTS["narrator"]
            else:
                d = GTTSVoice()

            st.markdown(f"**{role}**")
            lang_display = st.selectbox(
                "Language",
                options=[name for name, _, _ in common_langs],
                key=f"langname_{role}",
            )
            # resolve code and tld options
            lang_code = next(code for name, code, _ in common_langs if name == lang_display)
            tld_options = next(tlds for name, code, tlds in common_langs if name == lang_display)
            tld_sel = st.selectbox("Accent (tld)", options=tld_options, index=0, key=f"tld_{role}")

            slow = st.checkbox("Slow", value=d.slow, key=f"slow_{role}")
            pitch = st.slider("Pitch shift (semitones)", -6.0, 6.0, float(d.pitch_semitones), 0.5, key=f"pitch_{role}")

            role_to_voice[role] = GTTSVoice(lang=lang_code, tld=tld_sel, slow=slow, pitch_semitones=pitch)
else:
    st.info("Paste some dialogue to configure per-role settings.")

st.subheader("4) Synthesize")
if st.button("▶️ Synthesize & Play (gTTS)"):
    if not segments:
        st.error("No valid lines found. Use 'Role: text' per line.")
    elif any(role not in role_to_voice for role in roles):
        st.error("Some roles have not been configured.")
    else:
        try:
            audio_bytes, mime, filename = synthesize_dialogue_gtts(
                segments=segments,
                role_to_voice=role_to_voice,
                interturn_pause_ms=interturn_pause_ms,
                output_format=out_fmt,
            )
            st.success("Done! Listen or download below.")
            st.audio(audio_bytes, format=mime)
            st.download_button("⬇️ Download", data=audio_bytes, file_name=filename, mime=mime)
        except Exception as e:
            st.exception(e)

with st.expander("🔬 Notes / scientific considerations", expanded=False):
    st.markdown(
        """
- **gTTS constraints**: No true gender choice; only language/locale (`lang`, `tld`) and `slow`.  
- **Pitch shift**: Implemented via resampling; it approximates a more “male/female” timbre but is not a formant-preserving pitch shifter (so it slightly affects time/timbre).  
- **Pauses**: Inline `[pause=XXXms]` and inter-turn pause are rendered with silence insertion.  
- **Quality**: For the most natural voices, consider neural TTS (Google/Azure/Polly) or local neural engines (e.g., Coqui TTS).  
- **Reproducibility**: Document gTTS version, pydub version, and ffmpeg version.  
        """
    )
