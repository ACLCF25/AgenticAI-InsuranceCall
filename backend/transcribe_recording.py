"""
Post-call transcription for agent-transferred recordings using OpenAI Whisper.

When a call is transferred to a real human agent, the conversation happens in a
Twilio Conference and is not captured as text.  This module downloads the
conference recording from Twilio and transcribes it via OpenAI Whisper, then
saves the transcript to conversation_history so it appears in the call detail UI.
"""

import io
import logging
import os
import re
import tempfile

import requests
from requests.auth import HTTPBasicAuth
from openai import OpenAI

logger = logging.getLogger(__name__)


def _chunk_transcript(text: str, max_chars: int = 420, max_sentences: int = 4) -> list[str]:
    """Split a long Whisper transcript into readable conversation blocks."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        projected_len = current_len + len(sentence) + (1 if current else 0)
        if current and (projected_len > max_chars or len(current) >= max_sentences):
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = projected_len

    if current:
        chunks.append(" ".join(current).strip())

    return chunks or [normalized]


def _has_agent_transcript(db, call_id: str, request_id: str = None) -> bool:
    with db.conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM conversation_history
            WHERE speaker = 'agent_transcript'
              AND (call_id = %s OR request_id::text = %s)
            LIMIT 1
            """,
            (call_id, request_id or ""),
        )
        return cur.fetchone() is not None


def transcribe_agent_recording(call_id: str, recording_url: str, request_id: str = None, force: bool = False):
    """
    Download a Twilio recording and transcribe it with OpenAI Whisper.
    Saves the resulting transcript to conversation_history.

    This is designed to be called in a background thread from the recording
    webhook when call_mode='agent'.
    """
    try:
        logger.info(f"[transcribe] Starting transcription for call_id={call_id}")

        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        try:
            if not force and _has_agent_transcript(db, call_id, request_id):
                logger.info(
                    f"[transcribe] Existing agent transcript found for call_id={call_id}; skipping"
                )
                return
        finally:
            db.close()

        # Build full URL if needed
        if not recording_url.startswith('http'):
            recording_url = f"https://api.twilio.com{recording_url}.mp3"
        elif not recording_url.endswith('.mp3'):
            recording_url = f"{recording_url}.mp3"

        # Download recording from Twilio
        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN")

        resp = requests.get(
            recording_url,
            auth=HTTPBasicAuth(twilio_sid, twilio_token),
            timeout=120,
        )
        if resp.status_code != 200:
            logger.error(
                f"[transcribe] Failed to download recording for call_id={call_id}: "
                f"HTTP {resp.status_code}"
            )
            return

        audio_data = resp.content
        if not audio_data or len(audio_data) < 1000:
            logger.warning(f"[transcribe] Recording too small for call_id={call_id}, skipping")
            return

        logger.info(f"[transcribe] Downloaded {len(audio_data)} bytes for call_id={call_id}")

        # Transcribe with OpenAI Whisper
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Write to a temp file since OpenAI SDK expects a file-like object with a name
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text",
                )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        transcript_text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()

        if not transcript_text:
            logger.warning(f"[transcribe] Empty transcript for call_id={call_id}")
            return

        logger.info(
            f"[transcribe] Transcribed {len(transcript_text)} chars for call_id={call_id}"
        )

        transcript_chunks = _chunk_transcript(transcript_text)

        # Save to conversation_history
        db = DatabaseManager()
        try:
            if not force and _has_agent_transcript(db, call_id, request_id):
                logger.info(
                    f"[transcribe] Existing agent transcript found after transcription "
                    f"for call_id={call_id}; skipping save"
                )
                return
            for chunk in transcript_chunks:
                db.save_conversation(
                    call_id=call_id,
                    speaker='agent_transcript',
                    message=chunk,
                    request_id=request_id,
                )
            logger.info(
                f"[transcribe] Saved {len(transcript_chunks)} transcript chunk(s) "
                f"to conversation_history for call_id={call_id}"
            )
        finally:
            db.close()

    except Exception as e:
        logger.error(f"[transcribe] Error transcribing call_id={call_id}: {e}")
        import traceback
        traceback.print_exc()
