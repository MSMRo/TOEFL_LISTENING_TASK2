# app.py
import io
import re
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from pydub import AudioSegment

# ---------- pydub silence (works on any version) ----------
try:
    from pydub.generators import Silence
    def make_silence(duration_ms: int) -> AudioSegment:
        return Silence(duration=duration_ms).to_audio_segment()
except Exception:
    def make_silence(duration_ms: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration_ms)

# ---------- optional Google Cloud TTS imports (lazy) ----------
GCTTS_AVAILABLE = True
try:
    # imported lazily later, but probe now for availability messaging
    from google.cloud import texttospeech  # noqa: F401
except Exception:
    GCTTS_AVAILABLE = False

# ---------- offline engine ----------
import pyttsx3

# ---------- UI config ----------
st.set_page_config(page_title="Multi-Voice Role Reader (Offline + Neural)", page_icon="🗣️", layout="wide")
st.title("🗣️ Multi-Voice Role Reader — Offline + Neural (Google TTS)")

# ---------- regex and samples ----------
LINE_RE = re.compile(r'^\s*(?P<role>[^:]+)\s*:\s*(?P<text>.+?)\s*$', re.IGNORECASE)
PAUSE_TAG_RE = re.compile(r'\[pause\s*=\s*(\d+)\s*ms\]', re.IGNORECASE)

SAMPLE_DIALOGUE = """Narrator: Welcome! Paste a dialogue below. Each line must be 'Role: text'.
Professor: Good morning. Today we discuss how to differentiate signals from noise.
Student: Professor, how do we control for baseline wander in ECG?
Professor: Great question. We apply high-pass filtering and careful detrending.
Student: Thank you! And what cutoff is typical? [pause=300ms]
Professor: Around 0.5 Hz to remove slow drifts without distorting the PQRST complexes.
"""

# Hints to prefer female/male timbres when auto-assigning offline voices
FEMALE_HINTS = [
    "samantha", "victoria", "karen", "serena",     # macOS
    "zira",                                        # Windows
    "en-us+f", "en-uk+f", "en+f", "+f1", "+f2",    # eSpeak-NG
    "female"
]
MALE_HINTS = [
    "alex", "daniel", "david",                     # macOS/Windows
    "en-us+m", "en-uk+m", "en+m", "+m1", "+m2",    # eSpeak-NG
    "male"
]

# ---------- helpers: parsing & pauses ----------
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

# ---------- eSpeak-NG CLI (offline fallback on Linux) ----------
def espeak_cli_available() -> bool:
    return shutil.which("espeak-ng") is not None

def espeak_cli_list_voices() -> List[Dict]:
    if not espeak_cli_available():
        return []
    try:
        out = subprocess.check_output(["espeak-ng", "--voices"], text=True)
    except Exception:
        return []
    voices: List[Dict] = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        if line.lower().startswith(("ply", "pty", "language")):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        lang = parts[1]
        name = parts[3]
        gender = ""
        ag = parts[2].lower()
        if "f" in ag: gender = "female"
        elif "m" in ag: gender = "male"
        voices.append({"id": name, "name": name, "languages": [lang], "age": "", "gender": gender})
    return voices

def espeak_cli_tts(text: str, voice_id: str, rate_wpm: int, volume: float) -> bytes:
    if not espeak_cli_available():
        raise RuntimeError("espeak-ng not found in PATH")
    tmpdir = Path(tempfile.mkdtemp(prefix="espeakng_"))
    wav_path = tmpdir / "utt.wav"
    amp = max(0, min(200, int(volume * 200)))  # 0..200
    cmd = ["espeak-ng", "-v", voice_id, "-s", str(rate_wpm), "-a", str(amp), "-w", str(wav_path), text]
    subprocess.check_call(cmd)
    return wav_path.read_bytes()

# ---------- offline voice discovery ----------
@st.cache_resource(show_spinner=False)
def list_system_voices() -> List[Dict]:
    # Try pyttsx3 first
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices') or []
        out = []
        for v in voices:
            out.append({
                "id": getattr(v, "id", ""),
                "name": getattr(v, "name", ""),
                "languages": getattr(v, "languages", []),
                "age": getattr(v, "age", ""),
                "gender": getattr(v, "gender", ""),
            })
        try: engine.stop()
        except Exception: pass
        if out: return out
    except Exception:
        pass
    # Fallback to espeak-ng CLI
    return espeak_cli_list_voices()

def match_any(s: str, hints: List[str]) -> bool:
    s = (s or "").lower()
    return any(h in s for h in hints)

def looks_female(v: Dict) -> bool:
    return match_any(v.get("gender",""), ["female"]) or match_any(v.get("name",""), FEMALE_HINTS) or match_any(v.get("id",""), FEMALE_HINTS)

def looks_male(v: Dict) -> bool:
    return match_any(v.get("gender",""), ["male"]) or match_any(v.get("name",""), MALE_HINTS) or match_any(v.get("id",""), MALE_HINTS)

def pick_offline_voice_for_role(role: str, voices_all: List[Dict], used_ids: set) -> Optional[str]:
    rlow = (role or "").lower()
    prefer_female = any(k in rlow for k in ["student", "woman", "female", "girl"])
    prefer_male   = any(k in rlow for k in ["prof", "teacher", "man", "male", "boy"])

    def pick(fn):
        cands = [v for v in voices_all if fn(v) and v["id"] not in used_ids]
        return cands[0]["id"] if cands else None

    if prefer_female:
        vid = pick(looks_female)
        if vid: return vid
    if prefer_male:
        vid = pick(looks_male)
        if vid: return vid

    # Try female then male for diversity
    vid = pick(looks_female)
    if vid: return vid
    vid = pick(looks_male)
    if vid: return vid

    # Fallback: first unused
    for v in voices_all:
        if v["id"] not in used_ids:
            return v["id"]
    return voices_all[0]["id"] if voices_all else None

# ---------- offline synthesis ----------
def tts_to_wav_bytes_pyttsx3(text: str, voice_id: str, rate_wpm: int, volume: float) -> bytes:
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', rate_wpm)
    engine.setProperty('volume', volume)
    tmpdir = Path(tempfile.mkdtemp(prefix="seg_tts_"))
    wav_path = tmpdir / "utt.wav"
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    try: engine.stop()
    except Exception: pass
    return wav_path.read_bytes()

def tts_to_wav_bytes_offline(text: str, voice_id: str, rate_wpm: int, volume: float) -> bytes:
    try:
        return tts_to_wav_bytes_pyttsx3(text, voice_id, rate_wpm, volume)
    except Exception:
        if espeak_cli_available():
            return espeak_cli_tts(text, voice_id, rate_wpm, volume)
        raise RuntimeError(
            "No offline TTS backend available. Install 'espeak-ng' (Linux) or fix pyttsx3 voices."
        )

def synthesize_dialogue_offline(
    segments: List[Tuple[str, str]],
    role_to_voice_id: Dict[str, str],
    rate_wpm: int,
    volume: float,
    interturn_pause_ms: int,
) -> bytes:
    final = AudioSegment.silent(duration=0)
    for idx, (role, text) in enumerate(segments):
        voice_id = role_to_voice_id.get(role)
        if not voice_id:
            raise RuntimeError(f"No voice assigned for role '{role}'.")
        part = AudioSegment.silent(duration=0)
        for chunk_text, pause_ms in extract_pause_chunks(text):
            wav_bytes = tts_to_wav_bytes_offline(chunk_text, voice_id, rate_wpm, volume)
            seg = AudioSegment.from_wav(io.BytesIO(wav_bytes))
            part += seg
            if pause_ms and pause_ms > 0:
                part += make_silence(pause_ms)
        final += part
        if interturn_pause_ms and interturn_pause_ms > 0 and idx < len(segments) - 1:
            final += make_silence(interturn_pause_ms)
    buf = io.BytesIO()
    final.export(buf, format="wav")
    buf.seek(0)
    return buf.read()

# ---------- Google Cloud TTS (neural) ----------
def gcloud_client():
    from google.cloud import texttospeech
    return texttospeech.TextToSpeechClient(), texttospeech

@st.cache_resource(show_spinner=False)
def gcloud_list_voices() -> List[Dict]:
    try:
        client, tts = gcloud_client()
        resp = client.list_voices()
        out = []
        for v in resp.voices:
            # voice.name often like "en-US-Neural2-F"
            # languages codes in v.language_codes
            out.append({
                "name": v.name,
                "language_codes": list(v.language_codes),
                "ssml_gender": tts.SsmlVoiceGender(v.ssml_gender).name if v.ssml_gender is not None else "",
                "natural_sample_rate_hertz": v.natural_sample_rate_hertz,
            })
        return out
    except Exception:
        return []

def guess_lang_from_voice_name(voice_name: str, default="en-US") -> str:
    # voice names usually start with language code, e.g., "en-US-..."
    parts = (voice_name or "").split("-")
    if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
        return f"{parts[0]}-{parts[1]}"
    return default

def build_ssml_dialogue(
    segments: List[Tuple[str, str]],
    role_to_voice_name: Dict[str, str],
    interturn_pause_ms: int,
) -> str:
    # Insert <voice name="">...</voice> per turn; translate [pause=###ms] to <break time="###ms"/>
    def to_ssml_text(t: str) -> str:
        # escape basic XML entities
        t = (t.replace("&", "&amp;")
               .replace("<", "&lt;")
               .replace(">", "&gt;")
               .replace('"', "&quot;")
               .replace("'", "&apos;"))
        # replace [pause=###ms] with <break>
        def repl(m): return f'<break time="{m.group(1)}ms"/>'
        return PAUSE_TAG_RE.sub(repl, t)

    ssml_parts = ["<speak>"]
    for idx, (role, text) in enumerate(segments):
        vname = role_to_voice_name.get(role)
        if not vname:
            continue
        ssml_parts.append(f'<voice name="{vname}">{to_ssml_text(text)}</voice>')
        if interturn_pause_ms and idx < len(segments) - 1:
            ssml_parts.append(f'<break time="{interturn_pause_ms}ms"/>')
    ssml_parts.append("</speak>")
    return "\n".join(ssml_parts)

def synthesize_dialogue_gcloud(
    segments: List[Tuple[str, str]],
    role_to_voice_name: Dict[str, str],
    speaking_rate: float,
    volume_gain_db: float,
    interturn_pause_ms: int,
    audio_encoding: str = "LINEAR16",  # or "MP3", "OGG_OPUS"
) -> bytes:
    client, tts = gcloud_client()
    ssml = build_ssml_dialogue(segments, role_to_voice_name, interturn_pause_ms)
    input_text = tts.SynthesisInput(ssml=ssml)

    # choose a default voice for request; SSML <voice> overrides per segment
    first_voice = next(iter(role_to_voice_name.values()))
    language_code = guess_lang_from_voice_name(first_voice, default="en-US")

    voice = tts.VoiceSelectionParams(name=first_voice, language_code=language_code)
    enc_map = {
        "LINEAR16": tts.AudioEncoding.LINEAR16,
        "MP3": tts.AudioEncoding.MP3,
        "OGG_OPUS": tts.AudioEncoding.OGG_OPUS,
    }
    audio_config = tts.AudioConfig(
        audio_encoding=enc_map.get(audio_encoding, tts.AudioEncoding.LINEAR16),
        speaking_rate=float(speaking_rate),
        volume_gain_db=float(volume_gain_db),
    )
    resp = client.synthesize_speech(request={"input": input_text, "voice": voice, "audio_config": audio_config})
    return resp.audio_content

# ---------- UI: How it works ----------
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
- Paste one line per turn using **`Role: text`**.
- Choose **Offline** (system/pyttsx3 voices) or **Neural** (Google Cloud TTS) mode.
- Use inline pauses like `[pause=300ms]`. Inter-turn pause is also configurable.
- **Neural mode** needs Google credentials (set `GOOGLE_APPLICATION_CREDENTIALS`).
- Output can be played and downloaded.
    """)

# ---------- INPUT: dialogue ----------
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

# ---------- MODE selection ----------
with right:
    st.subheader("2) Mode & engine")
    use_neural = st.toggle("Use Neural TTS (Google Cloud) for natural voices", value=False)
    if use_neural and not GCTTS_AVAILABLE:
        st.warning("`google-cloud-texttospeech` not installed or import failed. Install it to enable Neural mode.", icon="⚠️")
    if use_neural and GCTTS_AVAILABLE and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        st.info("Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON path to enable Neural mode.", icon="ℹ️")

# ---------- VOICE discovery / mapping ----------
offline_voices = list_system_voices() if not use_neural else []
gcloud_voices = gcloud_list_voices() if (use_neural and GCTTS_AVAILABLE and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")) else []

if "role_voice_map_offline" not in st.session_state:
    st.session_state.role_voice_map_offline = {}
if "role_voice_map_neural" not in st.session_state:
    st.session_state.role_voice_map_neural = {}

# Auto-assign
st.subheader("3) Role → Voice mapping")
auto_reassign = st.checkbox("Re-assign automatically on every change", value=True, help="If unchecked, keep prior choices unless new roles appear.")

if use_neural:
    if not gcloud_voices:
        st.warning("No Google voices listed yet. Check credentials or network.", icon="⚠️")
    else:
        # Prepare options
        gvoice_names = [v["name"] for v in gcloud_voices]
        # Simple gender-oriented suggestions
        def suggest_neural(role: str) -> str:
            r = role.lower()
            prefer_f = any(k in r for k in ["student", "woman", "female", "girl"])
            prefer_m = any(k in r for k in ["prof", "teacher", "man", "male", "boy"])
            if prefer_f:
                for n in gvoice_names:
                    if ("-F" in n) or ("-Female" in n) or ("female" in n.lower()):
                        return n
            if prefer_m:
                for n in gvoice_names:
                    if ("-M" in n) or ("-Male" in n) or ("male" in n.lower()):
                        return n
            # Otherwise prefer Neural2/Neural
            for n in gvoice_names:
                if "Neural2" in n or "Neural" in n:
                    return n
            return gvoice_names[0]

        if roles:
            for role in roles:
                if auto_reassign or role not in st.session_state.role_voice_map_neural:
                    st.session_state.role_voice_map_neural[role] = suggest_neural(role)
            # manual selectors
            cols = st.columns(min(3, len(roles)) or 1)
            for i, role in enumerate(roles):
                with cols[i % len(cols)]:
                    st.session_state.role_voice_map_neural[role] = st.selectbox(
                        f"**{role}** (Google voice)",
                        options=gvoice_names,
                        index=gvoice_names.index(st.session_state.role_voice_map_neural[role]) if st.session_state.role_voice_map_neural[role] in gvoice_names else 0,
                        key=f"gsel_{role}"
                    )
        else:
            st.info("Paste some dialogue to configure voices.")
else:
    if not offline_voices:
        st.warning("No offline voices found. On Linux, install espeak-ng; on Windows/macOS, ensure system voices are available.", icon="⚠️")
    else:
        name_by_id = {v["id"]: (v["name"] or v["id"]) for v in offline_voices}
        id_by_option = {f'{v["name"]} | {v["id"]}': v["id"] for v in offline_voices}
        options = list(id_by_option.keys())

        # auto-assign
        used = set()
        if roles:
            for role in roles:
                if auto_reassign or role not in st.session_state.role_voice_map_offline:
                    vid = pick_offline_voice_for_role(role, offline_voices, used)
                    if vid:
                        st.session_state.role_voice_map_offline[role] = vid
                        used.add(vid)
            # manual selectors
            cols = st.columns(min(3, len(roles)) or 1)
            for i, role in enumerate(roles):
                with cols[i % len(cols)]:
                    current_id = st.session_state.role_voice_map_offline.get(role, offline_voices[0]["id"])
                    default_label = f'{name_by_id.get(current_id, current_id)} | {current_id}'
                    if default_label not in options:
                        options = [default_label] + options
                    sel = st.selectbox(
                        f"**{role}** (offline)",
                        options=options,
                        index=options.index(default_label),
                        key=f"sel_{role}"
                    )
                    st.session_state.role_voice_map_offline[role] = id_by_option.get(sel, current_id)
        else:
            st.info("Paste some dialogue to configure voices.")

# ---------- synthesis parameters ----------
st.subheader("4) Synthesis parameters")
c1, c2, c3 = st.columns(3)
with c1:
    interturn_pause_ms = st.slider("Pause between turns (ms)", 0, 1500, 250, 50)

if use_neural:
    with c2:
        speaking_rate = st.slider("Neural speaking rate (0.25–4.0)", 0.25, 4.0, 1.0, 0.05)
    with c3:
        volume_gain_db = st.slider("Neural volume gain (dB)", -96.0, 16.0, 0.0, 0.5)
    audio_enc = st.selectbox("Neural output format", ["LINEAR16", "MP3", "OGG_OPUS"], index=0)
else:
    with c2:
        rate_wpm = st.slider("Offline rate (words/min)", 100, 240, 175, 5)
    with c3:
        volume = st.slider("Offline volume", 0.1, 1.0, 1.0, 0.05)

# ---------- action ----------
st.subheader("5) Read now")
if st.button("▶️ Synthesize & Play"):
    if not segments:
        st.error("No valid lines found. Use 'Role: text' per line.")
    else:
        try:
            if use_neural:
                if not (GCTTS_AVAILABLE and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")):
                    st.error("Neural mode requires Google credentials and the google-cloud-texttospeech package.")
                elif not st.session_state.role_voice_map_neural or any(r not in st.session_state.role_voice_map_neural for r in roles):
                    st.error("Some roles lack a Google voice. Check mapping above.")
                else:
                    wav_or_bytes = synthesize_dialogue_gcloud(
                        segments=segments,
                        role_to_voice_name=st.session_state.role_voice_map_neural,
                        speaking_rate=speaking_rate,
                        volume_gain_db=volume_gain_db,
                        interturn_pause_ms=interturn_pause_ms,
                        audio_encoding=audio_enc
                    )
                    # Play & download (set mime)
                    mime = {"LINEAR16": "audio/wav", "MP3": "audio/mpeg", "OGG_OPUS": "audio/ogg"}[audio_enc]
                    st.success("Neural synthesis complete.")
                    st.audio(wav_or_bytes, format=mime)
                    st.download_button("⬇️ Download audio", data=wav_or_bytes,
                                       file_name=f"dialogue_roles.{ 'wav' if audio_enc=='LINEAR16' else ('mp3' if audio_enc=='MP3' else 'ogg') }",
                                       mime=mime)
            else:
                if not offline_voices:
                    st.error("No offline voices available. Install espeak-ng (Linux) or system voices.")
                elif not st.session_state.role_voice_map_offline or any(r not in st.session_state.role_voice_map_offline for r in roles):
                    st.error("Some roles have no assigned offline voice. Check the mapping above.")
                else:
                    wav_bytes = synthesize_dialogue_offline(
                        segments=segments,
                        role_to_voice_id=st.session_state.role_voice_map_offline,
                        rate_wpm=rate_wpm,
                        volume=volume,
                        interturn_pause_ms=interturn_pause_ms,
                    )
                    st.success("Offline synthesis complete.")
                    st.audio(wav_bytes, format="audio/wav")
                    st.download_button("⬇️ Download WAV", data=wav_bytes, file_name="dialogue_roles.wav", mime="audio/wav")
        except FileNotFoundError as e:
            st.error("Export error. Ensure **ffmpeg** is installed and on your PATH.\n" + str(e))
        except subprocess.CalledProcessError as e:
            st.error(f"espeak-ng error: {e}")
        except Exception as e:
            st.exception(e)

with st.expander("🔬 Notes / Best practices", expanded=False):
    st.markdown("""
- **Neural vs Offline**: Neural (Google) voices are far more natural. Offline is fully local but depends on your OS voices.
- **Linux**: If offline fails, `sudo apt-get install -y espeak-ng espeak-ng-data libasound2`.
- **Pauses**: Use `[pause=XXXms]` inside lines. The “Pause between turns” slider inserts a break between speakers.
- **Google TTS voices**: The app auto-lists voices via API when credentials are set. Choose Neural/Neural2 voices for best quality.
- **Reproducibility**: Document OS, voices, and package versions (pyttsx3, google-cloud-texttospeech, pydub).
""")
