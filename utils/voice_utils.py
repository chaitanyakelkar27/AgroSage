"""
AgroSage — Voice Input / Output Utility
=========================================
Provides the full voice pipeline for regional Indian language support:

    transcribe_audio()       → Speech-to-text via OpenAI Whisper (local)
    translate_to_english()   → Regional language → English
    translate_from_english() → English → Regional language
    text_to_speech()         → Text → audio bytes via gTTS

Supported languages:
    Hindi, Marathi, Punjabi, Tamil, Telugu, Kannada, English
"""

from __future__ import annotations

import io
import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Language Configuration ────────────────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "English":  {"iso": "en", "whisper": "en", "gtts": "en",    "translate": "en"},
    "Hindi":    {"iso": "hi", "whisper": "hi", "gtts": "hi",    "translate": "hi"},
    "Marathi":  {"iso": "mr", "whisper": "mr", "gtts": "mr",    "translate": "mr"},
    "Punjabi":  {"iso": "pa", "whisper": "pa", "gtts": "pa",    "translate": "pa"},
    "Tamil":    {"iso": "ta", "whisper": "ta", "gtts": "ta",    "translate": "ta"},
    "Telugu":   {"iso": "te", "whisper": "te", "gtts": "te",    "translate": "te"},
    "Kannada":  {"iso": "kn", "whisper": "kn", "gtts": "kn",    "translate": "kn"},
}


def get_language_names() -> list[str]:
    """Return list of supported language display names for UI dropdowns."""
    return list(SUPPORTED_LANGUAGES.keys())


def get_language_config(language_name: str) -> dict[str, str]:
    """
    Return language config dict for a given display name.

    Parameters
    ----------
    language_name : str
        Display name like 'Hindi', 'Marathi', etc.

    Returns
    -------
    dict with keys: iso, whisper, gtts, translate
    """
    return SUPPORTED_LANGUAGES.get(language_name, SUPPORTED_LANGUAGES["English"])


# ── Speech-to-Text (Whisper) ─────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, language_code: str = "en") -> str:
    """
    Transcribe audio bytes to text using OpenAI Whisper (local model).

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio file bytes (WAV, MP3, OGG, WEBM, etc.)
    language_code : str
        Whisper language code (e.g., 'hi', 'mr', 'ta', 'en').
        Defaults to 'en'.

    Returns
    -------
    str
        Transcribed text. Returns empty string on failure.
    """
    try:
        import whisper
    except ImportError:
        logger.error("openai-whisper is not installed. Run: pip install openai-whisper")
        return ""

    if not audio_bytes:
        return ""

    try:
        # Determine cache directory for whisper model
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "models", "whisper-tiny")
        os.makedirs(cache_dir, exist_ok=True)

        # Load model (cached after first download)
        model = whisper.load_model("tiny", download_root=cache_dir)

        # Write audio bytes to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            result = model.transcribe(
                tmp_path,
                language=language_code if language_code != "en" else None,
                fp16=False,  # CPU-safe
            )
            transcribed_text = result.get("text", "").strip()
            logger.info(f"Whisper transcription ({language_code}): {transcribed_text[:80]}...")
            return transcribed_text
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as exc:
        logger.error(f"Whisper transcription failed: {type(exc).__name__}: {exc}")
        return ""


# ── Translation ──────────────────────────────────────────────────────────────

def translate_to_english(text: str, source_lang: str) -> str:
    """
    Translate text from a regional language to English.

    Parameters
    ----------
    text : str
        Text in the source regional language.
    source_lang : str
        ISO language code (e.g., 'hi', 'mr', 'ta').

    Returns
    -------
    str
        English translation. Returns original text on failure or if
        source_lang is already 'en'.
    """
    if not text or not text.strip():
        return ""

    if source_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source=source_lang, target="en").translate(text)
        logger.info(f"Translated ({source_lang}→en): {text[:40]}... → {translated[:40]}...")
        return translated or text
    except ImportError:
        logger.error("deep-translator is not installed. Run: pip install deep-translator")
        return text
    except Exception as exc:
        logger.error(f"Translation ({source_lang}→en) failed: {type(exc).__name__}: {exc}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """
    Translate text from English to a regional language.

    Parameters
    ----------
    text : str
        English text to translate.
    target_lang : str
        ISO language code (e.g., 'hi', 'mr', 'ta').

    Returns
    -------
    str
        Translated text in the target language. Returns original text
        on failure or if target_lang is 'en'.
    """
    if not text or not text.strip():
        return ""

    if target_lang == "en":
        return text

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        logger.info(f"Translated (en→{target_lang}): {text[:40]}... → {translated[:40]}...")
        return translated or text
    except ImportError:
        logger.error("deep-translator is not installed. Run: pip install deep-translator")
        return text
    except Exception as exc:
        logger.error(f"Translation (en→{target_lang}) failed: {type(exc).__name__}: {exc}")
        return text


# ── Text-to-Speech (gTTS) ────────────────────────────────────────────────────

def text_to_speech(text: str, lang_code: str = "en") -> bytes:
    """
    Convert text to speech audio bytes using Google Text-to-Speech.

    Parameters
    ----------
    text : str
        Text to speak.
    lang_code : str
        gTTS language code (e.g., 'hi', 'mr', 'ta', 'en').

    Returns
    -------
    bytes
        MP3 audio bytes. Returns empty bytes on failure.
        Play in Streamlit with: st.audio(audio_bytes, format='audio/mp3')
    """
    if not text or not text.strip():
        return b""

    try:
        from gtts import gTTS
    except ImportError:
        logger.error("gTTS is not installed. Run: pip install gTTS")
        return b""

    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
        logger.info(f"gTTS generated {len(audio_bytes)} bytes ({lang_code})")
        return audio_bytes
    except Exception as exc:
        logger.error(f"gTTS synthesis failed: {type(exc).__name__}: {exc}")
        return b""


# ── Full Voice Pipeline ──────────────────────────────────────────────────────

def voice_pipeline(
    audio_bytes: bytes,
    language_name: str,
    process_fn: Optional[callable] = None,
) -> dict:
    """
    Run the full voice-in → process → voice-out pipeline.

    1. Transcribe audio (Whisper) in the farmer's language
    2. Translate transcription to English
    3. Process the English text (via process_fn callback)
    4. Translate result back to farmer's language
    5. Generate spoken audio of the result (gTTS)

    Parameters
    ----------
    audio_bytes : bytes
        Raw audio from the farmer.
    language_name : str
        Display name of the language (e.g., 'Hindi', 'Marathi').
    process_fn : callable, optional
        Function that takes English text and returns English response text.
        If None, the translated English text is returned as-is.

    Returns
    -------
    dict with keys:
        transcription      : str  — what the farmer said (original language)
        english_input      : str  — translated to English
        english_response   : str  — response in English
        regional_response  : str  — response translated back to farmer's language
        audio_response     : bytes — spoken audio of the regional response (MP3)
        language           : str  — language name used
    """
    config = get_language_config(language_name)
    whisper_code = config["whisper"]
    translate_code = config["translate"]
    gtts_code = config["gtts"]

    # Step 1: Transcribe
    transcription = transcribe_audio(audio_bytes, whisper_code)

    # Step 2: Translate to English
    english_input = translate_to_english(transcription, translate_code)

    # Step 3: Process (if callback provided)
    if process_fn is not None and english_input:
        try:
            english_response = process_fn(english_input)
        except Exception as exc:
            logger.error(f"Process function failed: {exc}")
            english_response = f"Error processing your request: {exc}"
    else:
        english_response = english_input

    # Step 4: Translate back to regional language
    regional_response = translate_from_english(english_response, translate_code)

    # Step 5: Generate speech
    audio_response = text_to_speech(regional_response, gtts_code)

    return {
        "transcription": transcription,
        "english_input": english_input,
        "english_response": english_response,
        "regional_response": regional_response,
        "audio_response": audio_response,
        "language": language_name,
    }


def generate_result_audio(text: str, language_name: str) -> bytes:
    """
    Convenience function to generate spoken audio for any result text.

    Parameters
    ----------
    text : str
        Text to speak (in English — will be translated first).
    language_name : str
        Target language display name.

    Returns
    -------
    bytes
        MP3 audio bytes of the translated and spoken text.
    """
    config = get_language_config(language_name)
    translate_code = config["translate"]
    gtts_code = config["gtts"]

    # Translate if not English
    regional_text = translate_from_english(text, translate_code)

    # Generate audio
    return text_to_speech(regional_text, gtts_code)
