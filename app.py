# app.py
import io
import re
import time
import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from gtts import gTTS
from pydub import AudioSegment

# ---- pydub silence: compatible with any version ----
try:
    from pydub.generators import Silence, Sine
    HAVE_SINE = True
    def make_silence(duration_ms: int) -> AudioSegment:
        return Silence(duration=duration_ms).to_audio_segment()
except Exception:
    HAVE_SINE = False
    def make_silence(duration_ms: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration_ms)

st.set_page_config(page_title="Multi-Voice Role Reader (gTTS) + Timer", page_icon="🗣️", layout="wide")
st.title("🗣️ Multi-Voice Role Reader — gTTS edition (max 4 roles) + ⏱️ Timer")

# ================= Parsing & pauses =================

LINE_RE = re.compile(r'^\s*(?P<role>[^:]+)\s*:\s*(?P<text>.+?)\s*$', re.IGNORECASE)
PAUSE_TAG_RE = re.compile(r'\[pause\s*=\s*(\d+)\s*ms\]', re.IGNORECASE)

SAMPLE_DIALOGUE = """Narrator: Welcome! Paste a dialogue below. Each line must be 'Role: text'.
Professor: Good morning. Today we discuss how to differentiate signals from noise.
Student 1: Professor, how do we control for baseline wander in ECG?
Professor: Great question. We apply high-pass filtering and careful detrending.
Student 2: Thank you! And what cutoff is typical? [pause=300ms]
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

# ================= gTTS Synthesis =================

@dataclass
class GTTSVoice:
    lang: str = "en"          # e.g., 'en', 'es'
    tld: str = "com"          # accents: 'com', 'co.uk', 'com.au', 'com.mx', 'fr', 'es', ...
    slow: bool = False        # slower = True
    pitch_semitones: float = 0.0  # pitch approx via resampling

def gtts_tts_bytes(text: str, lang: str = "en", tld: str = "com", slow: bool = False) -> bytes:
    tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

def apply_pitch_shift(seg: AudioSegment, semitones: float) -> AudioSegment:
    """
    Approximate pitch shift via resampling (affects pitch and slightly timing/timbre).
    """
    if abs(semitones) < 0.01:
        return seg
    factor = 2.0 ** (semitones / 12.0)
    new_frame_rate = int(seg.frame_rate * factor)
    shifted = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
    return shifted.set_frame_rate(seg.frame_rate)

def synthesize_dialogue_gtts(
    segments: List[Tuple[str, str]],
    role_to_voice: Dict[str, GTTSVoice],
    interturn_pause_ms: int,
    output_format: str = "mp3",
) -> Tuple[bytes, str, str]:
    final = AudioSegment.silent(duration=0)
    for idx, (role, text) in enumerate(segments):
        if role not in role_to_voice:
            raise RuntimeError(f"No gTTS config for role '{role}'. Add or rename roles to match the script.")
        v = role_to_voice[role]
        part = AudioSegment.silent(duration=0)
        for chunk_text, pause_ms in extract_pause_chunks(text):
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

# ================= Main UI (roles + synthesis) =================

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
- Paste your dialogue using **`Role: text`** lines (e.g., `Student 1: ...`).
- You can add/rename **extra roles** (up to a total of **4**) below.
- For each role, choose `lang`, `tld` (accent), `slow`, and **pitch shift** (± semitones).
- Inline pauses: `[pause=300ms]` + global inter-turn pause.
- Uses **gTTS** (free). For best naturalness, consider neural TTS later.
    """)

left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("1) Paste your dialogue")
    text_input = st.text_area(
        "Each line: Role: text",
        height=260,
        value=SAMPLE_DIALOGUE,
        placeholder="Professor: Hello class...\nStudent 1: I have a question...\nStudent 2: ...",
    )

segments = parse_dialogue(text_input)
roles_in_text = sorted({r for (r, _) in segments})

with right:
    st.subheader("2) Global parameters")
    interturn_pause_ms = st.slider("Pause between turns (ms)", 0, 2000, 250, 50)
    out_fmt = st.selectbox("Output format", ["mp3", "wav"], index=0)

# -------- Additional Roles (max total = 4) --------
st.subheader("3) Additional roles (enable & rename, max total = 4)")
total_existing = len(roles_in_text)

if total_existing > 4:
    st.error(f"Your script already contains {total_existing} unique roles; maximum supported is 4. "
             f"Please reduce roles in the text.")
    extra_allowed = 0
else:
    extra_allowed = 4 - total_existing

if "extra_roles" not in st.session_state:
    st.session_state.extra_roles = {}  # slot_i -> name

extra_names_collected: List[str] = []
if extra_allowed > 0:
    cols = st.columns(extra_allowed)
    default_labels = [f"Student {i+1}" for i in range(extra_allowed)]
    for i in range(extra_allowed):
        with cols[i]:
            enabled = st.checkbox(f"Enable extra role {i+1}", key=f"extra_enable_{i}", value=False)
            if enabled:
                name = st.text_input(
                    "Role name",
                    value=st.session_state.extra_roles.get(f"slot_{i}", default_labels[i]),
                    key=f"extra_name_{i}"
                ).strip()
                if name:
                    st.session_state.extra_roles[f"slot_{i}"] = name
                    extra_names_collected.append(name)
else:
    st.caption("No more extra roles can be added (already at maximum of 4).")

# Build final role list (preserve roles from text first; then extras)
final_roles: List[str] = roles_in_text[:]
for nm in extra_names_collected:
    if nm and nm not in final_roles:
        final_roles.append(nm)
final_roles = final_roles[:4]  # hard cap

# -------- Per-role gTTS settings --------
st.subheader("4) Role → gTTS settings")

DEFAULTS = {
    "student":     GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=+2.0),
    "student 1":   GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=+2.0),
    "student 2":   GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=+1.0),
    "professor":   GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=-2.0),
    "narrator":    GTTSVoice(lang="en", tld="com", slow=False, pitch_semitones=0.0),
}

role_to_voice: Dict[str, GTTSVoice] = {}

if final_roles:
    common_langs = [
        ("English (US)", "en", ["com", "com.au", "co.uk", "co.in"]),
        ("English (UK)", "en", ["co.uk", "com"]),
        ("Spanish (ES)", "es", ["es", "com"]),
        ("Spanish (MX/LatAm)", "es", ["com.mx", "com"]),
        ("Portuguese (BR)", "pt", ["com.br", "com"]),
        ("French (FR)", "fr", ["fr", "com"]),
    ]

    cols = st.columns(min(4, len(final_roles)))
    for i, role in enumerate(final_roles):
        with cols[i % len(cols)]:
            rlow = role.lower()
            if rlow in DEFAULTS:
                d = DEFAULTS[rlow]
            elif "student" in rlow:
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
            lang_code = next(code for name, code, _ in common_langs if name == lang_display)
            tld_options = next(tlds for name, code, tlds in common_langs if name == lang_display)
            tld_sel = st.selectbox("Accent (tld)", options=tld_options, index=0, key=f"tld_{role}")

            slow = st.checkbox("Slow", value=d.slow, key=f"slow_{role}")
            pitch = st.slider("Pitch shift (semitones)", -6.0, 6.0, float(d.pitch_semitones), 0.5, key=f"pitch_{role}")

            role_to_voice[role] = GTTSVoice(lang=lang_code, tld=tld_sel, slow=slow, pitch_semitones=pitch)
else:
    st.info("Paste some dialogue and/or enable extra roles to configure settings.")

# -------- Synthesis --------
st.subheader("5) Synthesize")
if st.button("▶️ Synthesize & Play (gTTS)"):
    if not segments:
        st.error("No valid lines found. Use 'Role: text' per line.")
    else:
        # Ensure every role that appears in the text has a config
        missing = [r for r in roles_in_text if r not in role_to_voice]
        if missing:
            st.error(f"The following roles from your script are not configured: {missing}. "
                     f"Either add/rename them in 'Additional roles' or adjust your text.")
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

# ===================== ⏱️ TIMER PANEL (bottom of page) =====================
# ===================== ⏱️ TIMER PANEL (bottom of page) =====================
st.divider()
st.header("⏱️ Speaking Timer: 15s → 45s")

# ------------ Session state ------------
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False       # True while a sequence is active
if "abort_version" not in st.session_state:
    st.session_state.abort_version = 0           # increment to abort any running loop

# ------------ Beep helpers ------------
def _beep_bytes(freq_hz=880, dur_ms=250, level_db=-3):
    try:
        from pydub.generators import Sine
        tone = Sine(freq_hz).to_audio_segment(duration=dur_ms).apply_gain(level_db)
        buf = io.BytesIO(); tone.export(buf, format="wav"); buf.seek(0)
        return buf.read()
    except Exception:
        silent = AudioSegment.silent(duration=dur_ms)
        buf = io.BytesIO(); silent.export(buf, format="wav"); buf.seek(0)
        return buf.read()

def _autoplay_audio(wav_bytes: bytes):
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    st.markdown(
        f"""
        <audio autoplay>
          <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>
        """,
        unsafe_allow_html=True,
    )

def _beep(freq=880, ms=250):
    _autoplay_audio(_beep_bytes(freq_hz=freq, dur_ms=ms))

# ------------ Controls ------------
c1, c2 = st.columns([1, 1])
with c1:
    start_clicked = st.button("▶ Start 15s → 45s", use_container_width=True, type="primary")
with c2:
    reset_clicked = st.button("⟲ Reset", use_container_width=True)

# If Reset pressed, signal any loop to stop
if reset_clicked:
    st.session_state.abort_version += 1
    st.session_state.timer_running = False
    _beep(freq=700, ms=200)

# ------------ UI placeholders ------------
status_box = st.empty()
phase_label = st.empty()
progress_ph = st.empty()  # will hold st.progress widget
time_left_ph = st.empty()

def _run_phase(total_seconds: int, label: str, abort_token: int) -> bool:
    """
    Run one countdown phase locally, updating placeholders.
    Returns True if phase completed (was not aborted).
    """
    # Create a progress bar at 0
    prog = progress_ph.progress(0, text=f"{label}: {total_seconds}s left")
    t_start = time.monotonic()
    while True:
        # Abort if user pressed Reset
        if abort_token != st.session_state.abort_version or not st.session_state.timer_running:
            return False

        elapsed = time.monotonic() - t_start
        left = max(0.0, total_seconds - elapsed)
        frac = (elapsed / total_seconds) if total_seconds > 0 else 1.0
        frac = max(0.0, min(1.0, frac))

        # Update UI
        phase_label.markdown(f"### **{label}**")
        try:
            # Newer Streamlit supports `text=` argument on progress
            prog.progress(frac, text=f"{label}: {int(left)}s left")
        except TypeError:
            # Fallback for older versions
            prog.progress(frac)
            time_left_ph.write(f"{label}: {int(left)}s left")

        # End of phase?
        if left <= 0.0:
            break

        time.sleep(0.1)  # smooth updates

    return True

# ------------ Orchestrate sequence ------------
if not st.session_state.timer_running:
    status_box.info("Timer is idle. Click **Start** to run 15s → 45s (beeps at start, switch, finish).")

if start_clicked and not st.session_state.timer_running:
    st.session_state.timer_running = True
    local_abort_token = st.session_state.abort_version

    # Start beep
    _beep(freq=1000, ms=250)
    status_box.info("Running…")

    # Phase 1: 15 seconds
    ok = _run_phase(15, "15 seconds", local_abort_token)
    if ok and st.session_state.timer_running and local_abort_token == st.session_state.abort_version:
        # Transition beep (15 → 45)
        _beep(freq=1200, ms=300)

        # Phase 2: 45 seconds
        ok2 = _run_phase(45, "45 seconds", local_abort_token)
        if ok2 and st.session_state.timer_running and local_abort_token == st.session_state.abort_version:
            # Finish beep
            _beep(freq=600, ms=500)
            status_box.success("✅ Finished (15s + 45s).")
            st.session_state.timer_running = False
        else:
            status_box.warning("⏹️ Stopped.")
            st.session_state.timer_running = False
    else:
        status_box.warning("⏹️ Stopped.")
        st.session_state.timer_running = False
