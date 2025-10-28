# app.py
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from pydub import AudioSegment

# ---- Silence generator: compatible con todas las versiones de pydub ----
try:
    from pydub.generators import Silence
    def make_silence(duration_ms: int) -> AudioSegment:
        return Silence(duration=duration_ms).to_audio_segment()
except Exception:
    def make_silence(duration_ms: int) -> AudioSegment:
        return AudioSegment.silent(duration=duration_ms)

# pyttsx3 no es 100% thread-safe → inicializar/cerrar dentro de funciones
import pyttsx3


# ======================== Config & Regex ========================

st.set_page_config(
    page_title="Multi-Voice Role Reader (Offline)",
    page_icon="🗣️",
    layout="wide"
)

LINE_RE = re.compile(r'^\s*(?P<role>[^:]+)\s*:\s*(?P<text>.+?)\s*$', re.IGNORECASE)
PAUSE_TAG_RE = re.compile(r'\[pause\s*=\s*(\d+)\s*ms\]', re.IGNORECASE)

SAMPLE_DIALOGUE = """Narrator: Welcome! Paste or type a dialogue below. Each line must be 'Role: text'.
Professor: Good morning. Today we discuss how to differentiate signals from noise.
Student: Professor, how do we control for baseline wander in ECG?
Professor: Great question. We apply high-pass filtering and careful detrending.
Student: Thank you! And what cutoff is typical? [pause=300ms]
Professor: Around 0.5 Hz to remove slow drifts without distorting the PQRST complexes.
"""

# IDs que suelen ser voces femeninas en distintos SO (se usan como pistas)
FEMALE_HINTS = [
    "samantha", "victoria", "karen", "serena",              # macOS
    "zira",                                                 # Windows SAPI
    "en-us+f", "en-uk+f", "en+f", "+f1", "+f2", "+f3",      # eSpeak-NG
    "female"
]
MALE_HINTS = [
    "alex", "daniel", "david",                              # macOS/Windows
    "en-us+m", "en-uk+m", "en+m", "+m1", "+m2", "+m3",      # eSpeak-NG
    "male"
]


# ======================== eSpeak-NG CLI helpers ========================

def _espeak_cli_available() -> bool:
    return shutil.which("espeak-ng") is not None

def _espeak_cli_list_voices() -> List[Dict]:
    """
    Parse 'espeak-ng --voices' to a list of voices.
    Devuelve: [{"id","name","languages","age","gender"}...]
    """
    if not _espeak_cli_available():
        return []
    try:
        out = subprocess.check_output(["espeak-ng", "--voices"], text=True)
    except Exception:
        return []
    lines = out.strip().splitlines()
    voices: List[Dict] = []
    # Saltar cabeceras; cada línea tiene columnas: pty, language, age/gender, name, ...
    for line in lines:
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
        if "f" in ag:
            gender = "female"
        elif "m" in ag:
            gender = "male"
        voices.append({
            "id": name,
            "name": name,
            "languages": [lang],
            "age": "",
            "gender": gender,
        })
    return voices

def _espeak_cli_tts(text: str, voice_id: str, rate_wpm: int, volume: float) -> bytes:
    """
    Usa espeak-ng CLI para sintetizar a WAV y devuelve bytes.
    - rate_wpm → -s
    - volume (0.1..1.0) → -a (0..200)
    """
    if not _espeak_cli_available():
        raise RuntimeError("espeak-ng not found in PATH")
    tmpdir = Path(tempfile.mkdtemp(prefix="espeakng_"))
    wav_path = tmpdir / "utt.wav"
    amp = max(0, min(200, int(volume * 200)))
    cmd = [
        "espeak-ng",
        "-v", voice_id,
        "-s", str(rate_wpm),
        "-a", str(amp),
        "-w", str(wav_path),
        text
    ]
    subprocess.check_call(cmd)
    return wav_path.read_bytes()


# ======================== Voice discovery (robusto) ========================

@st.cache_resource(show_spinner=False)
def list_system_voices() -> List[Dict]:
    """
    Intenta pyttsx3 primero; si falla (p.ej., falta espeak-ng en Linux),
    cae a lista de voces via espeak-ng CLI. Si nada está disponible → [].
    """
    # pyttsx3
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
        try:
            engine.stop()
        except Exception:
            pass
        if out:
            return out
    except Exception:
        pass

    # Fallback: espeak-ng
    voices_cli = _espeak_cli_list_voices()
    if voices_cli:
        return voices_cli

    return []


# ======================== Parsing & prosodia ========================

def parse_dialogue(text: str) -> List[Tuple[str, str]]:
    """Extrae [(rol, texto)] de líneas 'Role: text'."""
    segments: List[Tuple[str, str]] = []
    for raw in text.strip().splitlines():
        if not raw.strip():
            continue
        m = LINE_RE.match(raw)
        if m:
            role = m.group('role').strip()
            seg_text = m.group('text').strip()
            segments.append((role, seg_text))
    return segments

def extract_pause_chunks(text: str) -> List[Tuple[str, Optional[int]]]:
    """
    Divide en [(chunk, pause_ms_despues)] usando [pause=###ms].
    Ej.: "Hola [pause=300ms] mundo" → [("Hola",300),("mundo",None)]
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


# ======================== Heurística de selección de voces ========================

def _match_any(s: str, hints: List[str]) -> bool:
    s = (s or "").lower()
    return any(h in s for h in hints)

def _looks_female(v: Dict) -> bool:
    return _match_any(v.get("gender",""), ["female"]) or _match_any(v.get("name",""), FEMALE_HINTS) or _match_any(v.get("id",""), FEMALE_HINTS)

def _looks_male(v: Dict) -> bool:
    return _match_any(v.get("gender",""), ["male"]) or _match_any(v.get("name",""), MALE_HINTS) or _match_any(v.get("id",""), MALE_HINTS)

def pick_auto_voice_for_role(role: str, voices_all: List[Dict], used_ids: set) -> Optional[str]:
    """
    Heurística:
      - roles con ['student','woman','female','girl'] → prioriza voces femeninas
      - roles con ['prof','teacher','man','male','boy'] → prioriza voces masculinas
      - evita repetir voces si es posible
      - fallback: primera voz no usada / primera voz
    """
    rlow = (role or "").lower()
    prefer_female = any(k in rlow for k in ["student", "woman", "female", "girl"])
    prefer_male   = any(k in rlow for k in ["prof", "teacher", "man", "male", "boy"])

    def pick(filter_fn):
        cands = [v for v in voices_all if filter_fn(v) and v["id"] not in used_ids]
        return cands[0]["id"] if cands else None

    # Preferencia explícita
    if prefer_female:
        vid = pick(_looks_female)
        if vid: return vid
    if prefer_male:
        vid = pick(_looks_male)
        if vid: return vid

    # Si el nombre del rol contiene 'female'/'male' también ayuda:
    if "female" in rlow and not prefer_female:
        vid = pick(_looks_female)
        if vid: return vid
    if "male" in rlow and not prefer_male:
        vid = pick(_looks_male)
        if vid: return vid

    # En general, intenta alternar mujer/hombre para diferenciar timbres
    # Si ya hay usadas, intenta escoger una con género distinto a la última
    # (simple: prueba femenina primero, luego masculina)
    vid = pick(_looks_female)
    if vid: return vid
    vid = pick(_looks_male)
    if vid: return vid

    # Fallback: primera no usada
    for v in voices_all:
        if v["id"] not in used_ids:
            return v["id"]
    return voices_all[0]["id"] if voices_all else None


# ======================== Síntesis (robusta) ========================

def tts_to_wav_bytes_pyttsx3(text: str, voice_id: str, rate_wpm: int, volume: float) -> bytes:
    """Síntesis con pyttsx3 → WAV bytes."""
    engine = pyttsx3.init()
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', rate_wpm)
    engine.setProperty('volume', volume)
    tmpdir = Path(tempfile.mkdtemp(prefix="seg_tts_"))
    wav_path = tmpdir / "utt.wav"
    engine.save_to_file(text, str(wav_path))
    engine.runAndWait()
    try:
        engine.stop()
    except Exception:
        pass
    return wav_path.read_bytes()

def tts_to_wav_bytes(text: str, voice_id: str, rate_wpm: int, volume: float) -> bytes:
    """
    Intenta pyttsx3; si falla, cae a espeak-ng CLI (si disponible).
    """
    try:
        return tts_to_wav_bytes_pyttsx3(text, voice_id, rate_wpm, volume)
    except Exception:
        if _espeak_cli_available():
            return _espeak_cli_tts(text, voice_id, rate_wpm, volume)
        raise RuntimeError(
            "No TTS backend available. Install 'espeak-ng' (Linux) o corrige backend de pyttsx3.\n"
            "Ubuntu/Debian: sudo apt-get install -y espeak-ng espeak-ng-data libasound2"
        )

def synthesize_dialogue(
    segments: List[Tuple[str, str]],
    role_to_voice_id: Dict[str, str],
    rate_wpm: int,
    volume: float,
    interturn_pause_ms: int,
) -> bytes:
    """
    Sintetiza el diálogo con voces por rol, soporta pausas inline [pause=XXXms].
    Devuelve WAV bytes.
    """
    final = AudioSegment.silent(duration=0)

    for idx, (role, text) in enumerate(segments):
        voice_id = role_to_voice_id.get(role)
        if not voice_id:
            raise RuntimeError(f"No voice assigned for role '{role}'.")

        chunks = extract_pause_chunks(text)
        part = AudioSegment.silent(duration=0)
        for (chunk_text, pause_ms) in chunks:
            wav_bytes = tts_to_wav_bytes(chunk_text, voice_id, rate_wpm, volume)
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


# ======================== UI ========================

st.title("🗣️ Multi-Voice Role Reader — Offline (pyttsx3 + pydub)")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        """
- Paste a dialogue with one line per turn → **`Role: text`**.
- Roles are auto-detected; the app auto-assigns distinct voices.
- **Female voices** are preferred for roles like *Student/Woman/Female*, etc.
- Use inline pauses: `[pause=300ms]`.
- Everything runs offline. If `pyttsx3` fails, it tries `espeak-ng` CLI (Linux).
- Requires `pyttsx3`, `pydub`, and **ffmpeg** in your PATH.
        """
    )

left, right = st.columns([2, 1], gap="large")

# 1) Texto
with left:
    st.subheader("1) Paste your dialogue")
    text_input = st.text_area(
        "Each line: Role: text",
        height=260,
        value=SAMPLE_DIALOGUE,
        placeholder="Professor: Hello class...\nStudent: I have a question...\n...",
    )

# 2) Voces disponibles
voices_all = list_system_voices()
if "role_voice_map" not in st.session_state:
    st.session_state.role_voice_map = {}

segments = parse_dialogue(text_input)
roles = sorted({r for (r, _) in segments})

with right:
    st.subheader("2) Auto-assign voices")
    if not voices_all:
        st.error("No OS voices found. Install voices or (on Linux) espeak-ng.")
    else:
        st.caption(f"Detected {len(voices_all)} voices")

    auto_reassign = st.checkbox(
        "Re-assign voices automatically on every change",
        value=True,
        help="If unchecked, existing selections are kept unless new roles appear."
    )

    if st.button("🔁 Assign/Refresh voices now"):
        st.toast("Re-assigning voices…", icon="🔊")
        st.session_state.role_voice_map = {}
        auto_reassign = True

# 3) Auto o manual mapping
if roles and voices_all:
    used = set()
    for role in roles:
        if not auto_reassign and role in st.session_state.role_voice_map:
            used.add(st.session_state.role_voice_map[role])
            continue
        vid = pick_auto_voice_for_role(role, voices_all, used)
        if vid:
            st.session_state.role_voice_map[role] = vid
            used.add(vid)

st.subheader("3) Role → Voice mapping (optional)")
if roles and voices_all:
    name_by_id = {v["id"]: (v["name"] or v["id"]) for v in voices_all}
    id_by_option = {f'{v["name"]} | {v["id"]}': v["id"] for v in voices_all}
    options = list(id_by_option.keys())

    cols = st.columns(min(3, len(roles)) or 1)
    for i, role in enumerate(roles):
        with cols[i % len(cols)]:
            current_id = st.session_state.role_voice_map.get(role, voices_all[0]["id"])
            default_label = f'{name_by_id.get(current_id, current_id)} | {current_id}'
            if default_label not in options:
                options = [default_label] + options
            sel = st.selectbox(
                f"**{role}**",
                options=options,
                index=options.index(default_label),
                key=f"sel_{role}"
            )
            st.session_state.role_voice_map[role] = id_by_option.get(sel, current_id)
else:
    st.info("Paste some dialogue to configure voices by role.")

# 4) Parámetros
st.subheader("4) Synthesis parameters")
c1, c2, c3 = st.columns(3)
with c1:
    rate_wpm = st.slider("Rate (words/min)", 100, 240, 175, 5)
with c2:
    volume = st.slider("Volume", 0.1, 1.0, 1.0, 0.05)
with c3:
    interturn_pause_ms = st.slider("Pause between turns (ms)", 0, 1500, 250, 50)

# 5) Síntesis
st.subheader("5) Read now")
read_now = st.button("▶️ Read now (auto-detect roles & voices)")

if read_now:
    if not segments:
        st.error("No valid lines found. Use 'Role: text' per line.")
    elif not st.session_state.role_voice_map or any(
        r not in st.session_state.role_voice_map for r in roles
    ):
        st.error("Some roles have no assigned voice. Check the mapping above.")
    else:
        try:
            wav_bytes = synthesize_dialogue(
                segments=segments,
                role_to_voice_id=st.session_state.role_voice_map,
                rate_wpm=rate_wpm,
                volume=volume,
                interturn_pause_ms=interturn_pause_ms,
            )
            st.success("Done! Listen or download below.")
            st.audio(wav_bytes, format="audio/wav")
            st.download_button(
                "⬇️ Download WAV",
                data=wav_bytes,
                file_name="dialogue_roles.wav",
                mime="audio/wav",
            )
        except FileNotFoundError as e:
            st.error(
                "Export error. Ensure **ffmpeg** is installed and on your PATH.\n"
                f"Details: {e}"
            )
        except Exception as e:
            st.exception(e)

with st.expander("🔬 Notes / Best practices", expanded=False):
    st.markdown(
        """
- **Reproducibility**: Document OS, installed voices, and `pyttsx3` version.
- **Female voices**: On Linux, install `espeak-ng` and try IDs like `en-us+f2`; on macOS, `Samantha`; on Windows, `Microsoft Zira`.
- **Prosody**: Inline `[pause=XXXms]` for timing. For advanced prosody, consider SSML (cloud TTS).
- **Offline**: If `pyttsx3` fails, the app tries `espeak-ng` automatically (when available).
        """
    )
