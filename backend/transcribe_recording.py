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
import tempfile

import requests
from requests.auth import HTTPBasicAuth
from openai import OpenAI

logger = logging.getLogger(__name__)


def transcribe_agent_recording(call_id: str, recording_url: str, request_id: str = None):
    """
    Download a Twilio recording and transcribe it with OpenAI Whisper.
    Saves the resulting transcript to conversation_history.

    This is designed to be called in a background thread from the recording
    webhook when call_mode='agent'.
    """
    try:
        logger.info(f"[transcribe] Starting transcription for call_id={call_id}")

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

        # Save to conversation_history
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        try:
            db.save_conversation(
                call_id=call_id,
                speaker='agent_transcript',
                message=transcript_text,
                request_id=request_id,
            )
            logger.info(f"[transcribe] Saved transcript to conversation_history for call_id={call_id}")
        finally:
            db.close()

    except Exception as e:
        logger.error(f"[transcribe] Error transcribing call_id={call_id}: {e}")
        import traceback
        traceback.print_exc()
