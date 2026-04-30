"""
Flask API Server for Autonomous Credentialing Agent
Handles Twilio webhooks and real-time audio streaming
"""

import os
import re
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables BEFORE any other imports that use them
load_dotenv()

import asyncio
import json
import uuid
from typing import Dict, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from twilio.twiml.voice_response import VoiceResponse, Stream, Connect
from twilio.rest import Client as TwilioClient
import threading
from queue import Queue
import base64
import websockets
from urllib.parse import urlencode
from xml.sax.saxutils import escape as xml_escape

from credentialing_agent import (
    CredentialingAgent,
    CredentialingState,
    CallState,
    AudioType,
    ingest_call_knowledge,
)
from knowledge_base import KnowledgeBase, redact_text, EmbeddingError, SearchError, KnowledgeBaseError
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from flask import send_from_directory
from elevenlabs.client import ElevenLabs
import time

# ElevenLabs client for TTS
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default: Rachel
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000").rstrip("/")
ENABLE_ELEVENLABS_TTS = os.getenv("ENABLE_ELEVENLABS_TTS", "true").lower() == "true"
CHECK_CALLBACK_REACHABLE = os.getenv("CHECK_CALLBACK_REACHABLE", "false").lower() == "true"

# Deepgram live-streaming STT (replaces Twilio Gather speech recognition for the
# human-conversation phase, cutting ~1.5-2s per turn). Off by default for safety.
ENABLE_DEEPGRAM_STREAMING = os.getenv("ENABLE_DEEPGRAM_STREAMING", "false").lower() == "true"
try:
    import deepgram_streaming  # noqa: E402  (import after env load is intentional)
except ImportError:
    deepgram_streaming = None  # type: ignore[assignment]

# Raw-WebSocket adapter for Twilio Media Streams. Twilio uses raw WebSockets,
# not Socket.IO, so we need flask-sock to handle the /media-stream endpoint.
try:
    from flask_sock import Sock  # noqa: E402
    _FLASK_SOCK_AVAILABLE = True
except ImportError:
    _FLASK_SOCK_AVAILABLE = False
    logger.warning("flask-sock not installed; Twilio Media Streams /media-stream will 404. "
                   "Install with: pip install flask-sock")

# IVR menu detection phrases — used in both human-speech webhook handlers
IVR_MENU_INDICATORS = [
    'press 1', 'press 2', 'press 3', 'press 4', 'press 5',
    'press 6', 'press 7', 'press 8', 'press 9', 'press 0',
    'press zero', 'press star', 'press pound',
    'dial 1', 'dial 2', 'dial 3', 'dial 4', 'dial 5',
    'option 1', 'option 2', 'option 3', 'option 4', 'option 5',
    'say 1', 'say 2', 'press or say',
    'for english', 'para español',
    'please listen carefully', 'menu options have changed',
    'if you are calling about', 'if you need',
    'to repeat these options', 'to hear these options again',
    'for more options', 'you have reached', 'please enter',
]

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (Bearer token auth doesn't require restricted origins)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Raw-WebSocket support for Twilio Media Streams (separate from Socket.IO above).
# Twilio sends raw WebSocket frames, not the Socket.IO protocol, so we need a
# different handler. Sock attaches transparently to the same Flask app.
sock = Sock(app) if _FLASK_SOCK_AVAILABLE else None
from auth import (  # noqa: E402
    admin_or_above,
    admin_required,
    agent_or_above,
    auth_bp,
    get_current_user,
    get_current_user_id,
    super_admin_required,
)

app.register_blueprint(auth_bp)

# Global state management
active_calls: Dict[str, CredentialingAgent] = {}
call_states: Dict[str, CredentialingState] = {}
call_states_by_sid: Dict[str, CredentialingState] = {}
call_state_lock = threading.Lock()
audio_queues: Dict[str, Queue] = {}

# ── Per-call log buffering ──────────────────────────────────────────────────
import re as _re
import logging as _logging

call_log_buffers: Dict[str, list] = {}
_call_log_buffers_lock = threading.Lock()
_log_tl = threading.local()  # thread-local: .call_id set per webhook/ws handler

_RE_CALL_ID  = _re.compile(
    r'call_id[=:\s]+([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})',
    _re.I,
)
_RE_CALL_SID = _re.compile(r'\b(CA[a-f0-9]{32})\b', _re.I)


def _is_admin_like(user: Optional[dict]) -> bool:
    return bool(user and user.get("role") in {"admin", "super_admin"})


def _deny_if_call_not_owned(owner_user_id: Optional[str]):
    user = get_current_user()
    if _is_admin_like(user):
        return None
    if owner_user_id and user and str(owner_user_id) == str(user.get("id")):
        return None
    return jsonify({"success": False, "error": "Call not found"}), 404

# Twilio client
twilio_client = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

# =============================================================================
# Twilio Number Pool
# =============================================================================

import psycopg2
import psycopg2.extras
import psycopg2.pool

_twilio_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=5,
    host=os.getenv("SUPABASE_HOST"),
    database="postgres",
    user=os.getenv("SUPABASE_USER", "postgres"),
    password=os.getenv("SUPABASE_PASSWORD"),
    port=6543,
)


def _tp_get_conn():
    return _twilio_pool.getconn()


def _tp_put_conn(conn):
    _twilio_pool.putconn(conn)


def _seed_twilio_numbers():
    """Seed Twilio numbers from env vars into the DB on startup."""
    numbers = set()

    # Single number
    single = (os.getenv("TWILIO_PHONE_NUMBER") or "").strip()
    if single:
        numbers.add(single)

    # Comma-separated list
    multi = (os.getenv("TWILIO_PHONE_NUMBERS") or "").strip()
    if multi:
        for num in multi.split(","):
            num = num.strip()
            if num:
                numbers.add(num)

    if not numbers:
        return

    conn = _tp_get_conn()
    try:
        with conn.cursor() as cur:
            for phone in numbers:
                cur.execute(
                    "INSERT INTO twilio_numbers (phone_number, friendly_name, is_active) "
                    "VALUES (%s, %s, TRUE) "
                    "ON CONFLICT (phone_number) DO NOTHING",
                    (phone, f"ENV: {phone}"),
                )
        conn.commit()
        logger.info(f"Seeded {len(numbers)} Twilio number(s) from env")
    except Exception as e:
        logger.warning(f"Could not seed Twilio numbers: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        _tp_put_conn(conn)


def _acquire_twilio_number(call_id: str, call_sid: str = None):
    """Atomically acquire an available Twilio number for a call.
    Returns the phone number string, or None if all are busy."""
    conn = _tp_get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, phone_number FROM twilio_numbers "
                "WHERE is_active = TRUE AND current_call_id IS NULL "
                "ORDER BY updated_at ASC NULLS FIRST "
                "LIMIT 1 "
                "FOR UPDATE SKIP LOCKED"
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                "UPDATE twilio_numbers "
                "SET current_call_id = %s, current_call_sid = %s, "
                "    in_use_since = NOW(), updated_at = NOW() "
                "WHERE id = %s",
                (call_id, call_sid, row["id"]),
            )
        conn.commit()
        logger.info(f"Acquired Twilio number {row['phone_number']} for call {call_id}")
        return row["phone_number"]
    except Exception as e:
        logger.error(f"Error acquiring Twilio number: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        _tp_put_conn(conn)


def _release_twilio_number(call_id: str = None, call_sid: str = None):
    """Release a Twilio number back to the pool."""
    if not call_id and not call_sid:
        return
    conn = _tp_get_conn()
    try:
        with conn.cursor() as cur:
            if call_id:
                cur.execute(
                    "UPDATE twilio_numbers "
                    "SET current_call_id = NULL, current_call_sid = NULL, "
                    "    in_use_since = NULL, updated_at = NOW() "
                    "WHERE current_call_id = %s",
                    (call_id,),
                )
            elif call_sid:
                cur.execute(
                    "UPDATE twilio_numbers "
                    "SET current_call_id = NULL, current_call_sid = NULL, "
                    "    in_use_since = NULL, updated_at = NOW() "
                    "WHERE current_call_sid = %s",
                    (call_sid,),
                )
        conn.commit()
        logger.info(f"Released Twilio number (call_id={call_id}, call_sid={call_sid})")
    except Exception as e:
        logger.warning(f"Error releasing Twilio number: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        _tp_put_conn(conn)


def _update_twilio_number_call_sid(call_id: str, call_sid: str):
    """Persist the Twilio Call SID for an already-acquired phone line."""
    if not call_id or not call_sid:
        return
    conn = _tp_get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE twilio_numbers "
                "SET current_call_sid = %s, updated_at = NOW() "
                "WHERE current_call_id = %s",
                (call_sid, call_id),
            )
        conn.commit()
        logger.info(f"Updated Twilio number Call SID for call {call_id}: {call_sid}")
    except Exception as e:
        logger.warning(f"Error updating Twilio number Call SID for call {call_id}: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        _tp_put_conn(conn)


# Seed on startup
try:
    _seed_twilio_numbers()
except Exception as e:
    logger.warning(f"Twilio number seeding skipped: {e}")

# =============================================================================
# Call-log sink (Loguru + stdlib bridge)
# =============================================================================

class CallLogSink:
    """Loguru sink that routes records into per-call in-memory buffers."""

    def __call__(self, message):
        record = message.record
        text   = record["message"]

        call_id = getattr(_log_tl, "call_id", None)

        if not call_id:
            m = _RE_CALL_ID.search(text)
            if m:
                call_id = m.group(1)

        if not call_id:
            m = _RE_CALL_SID.search(text)
            if m:
                with call_state_lock:
                    state = call_states_by_sid.get(m.group(1))
                    if state:
                        call_id = state.get("call_id")

        if not call_id:
            return

        entry = {
            "call_id":       call_id,
            "logged_at":     record["time"].isoformat(),
            "level":         record["level"].name,
            "logger_name":   record["name"],
            "function_name": record["function"],
            "line_number":   record["line"],
            "message":       text,
        }
        with _call_log_buffers_lock:
            call_log_buffers.setdefault(call_id, []).append(entry)


class _StdLogBridge(_logging.Handler):
    """Bridges Python standard-library log records (deepgram_streaming) into call buffers."""

    def emit(self, record):
        msg     = record.getMessage()
        call_id = getattr(_log_tl, "call_id", None)

        if not call_id:
            m = _RE_CALL_SID.search(msg)
            if m:
                with call_state_lock:
                    state = call_states_by_sid.get(m.group(1))
                    if state:
                        call_id = state.get("call_id")

        if not call_id:
            return

        entry = {
            "call_id":       call_id,
            "logged_at":     datetime.utcnow().isoformat() + "Z",
            "level":         record.levelname,
            "logger_name":   record.name,
            "function_name": record.funcName,
            "line_number":   record.lineno,
            "message":       msg,
        }
        with _call_log_buffers_lock:
            call_log_buffers.setdefault(call_id, []).append(entry)


def _set_log_context(call_id: str) -> None:
    _log_tl.call_id = call_id


def _clear_log_context() -> None:
    _log_tl.call_id = None


def _flush_call_logs(call_id: str, call_sid: str = None) -> None:
    """Pop this call's buffer and batch-insert into Supabase call_logs table."""
    with _call_log_buffers_lock:
        entries = call_log_buffers.pop(call_id, [])
    if not entries:
        return
    if call_sid:
        for e in entries:
            e.setdefault("call_sid", call_sid)
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        db.save_call_logs(entries)
        db.close()
        logger.info(f"Saved {len(entries)} call log entries for call_id={call_id}")
    except Exception as exc:
        logger.error(f"Failed to save call logs for call_id={call_id}: {exc}")


# Register Loguru sink
logger.add(CallLogSink(), level="DEBUG", format="{message}")

# Bridge deepgram_streaming's standard-library logger
_dg_bridge = _StdLogBridge()
_logging.getLogger("deepgram_streaming").addHandler(_dg_bridge)
_logging.getLogger("deepgram_streaming").propagate = False


# =============================================================================
# Helper utilities
# =============================================================================

def _is_public_url(url: str) -> bool:
    """Return True when the URL is not obviously localhost/loopback."""
    return not re.match(r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:?|/)", (url or "").lower())


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with timezone info."""
    return utcnow().isoformat()


def serialize_timestamp(value: Optional[datetime]) -> Optional[str]:
    """
    Serialize timestamps for API responses.

    Existing database columns may still be naive; treat those values as UTC so
    browser-local rendering stays correct across viewer timezones.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def get_callback_base(req) -> str:
    """
    Prefer CALLBACK_URL (minus /webhook/voice suffix) if set and usable; otherwise derive from request host.
    Always returns a scheme+host without trailing slash.
    """
    callback_env = (os.getenv("CALLBACK_URL", "") or "").strip()
    if callback_env.endswith("/webhook/voice"):
        callback_env = callback_env[: -len("/webhook/voice")]
    if callback_env and _is_public_url(callback_env):
        base = callback_env
    else:
        # Fallback to request host, force https because Twilio requires it
        scheme = "https"
        base = f"{scheme}://{req.host}"
    return base.rstrip("/")


def build_recording_status_callback(base_url: str, call_id: str = None, request_id: str = None) -> str:
    """Build a Twilio recording callback URL with stable identifiers."""
    callback_url = f"{base_url.rstrip('/')}/webhook/recording-status"
    params = {}
    if call_id:
        params["call_id"] = call_id
    if request_id:
        params["request_id"] = request_id
    if params:
        callback_url = f"{callback_url}?{urlencode(params)}"
    return callback_url


def build_conference_twiml(conference_name: str, recording_callback_url: str, end_on_exit: bool) -> str:
    """Generate conference TwiML with recording enabled from join."""
    escaped_callback_url = xml_escape(recording_callback_url, {'"': '&quot;'})
    escaped_conference_name = xml_escape(conference_name)
    end_on_exit_value = "true" if end_on_exit else "false"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Response>'
        '<Dial>'
        f'<Conference startConferenceOnEnter="true" endConferenceOnExit="{end_on_exit_value}" '
        'record="record-from-start" '
        f'recordingStatusCallback="{escaped_callback_url}" '
        'recordingStatusCallbackMethod="POST" '
        'recordingStatusCallbackEvent="completed absent">'
        f'{escaped_conference_name}'
        '</Conference>'
        '</Dial>'
        '</Response>'
    )


ENV_VALIDATION_OK: bool = False
ENV_VALIDATION_ERRORS = []


def validate_environment() -> bool:
    """Validate critical environment variables for Twilio + audio; caches result."""
    global ENV_VALIDATION_OK, ENV_VALIDATION_ERRORS

    errors = []
    required_keys = [
        "CALLBACK_URL",
        "BASE_URL",
        "TWILIO_ACCOUNT_SID",
        "TWILIO_AUTH_TOKEN",
        "SUPABASE_HOST",
        "SUPABASE_PASSWORD",
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "ELEVENLABS_API_KEY",
    ]

    for key in required_keys:
        if not os.getenv(key):
            errors.append(f"Missing environment variable: {key}")

    callback_url = os.getenv("CALLBACK_URL", "")
    base_url = os.getenv("BASE_URL", "")

    if callback_url and not callback_url.endswith("/webhook/voice"):
        errors.append("CALLBACK_URL must end with /webhook/voice")
    if callback_url and not _is_public_url(callback_url):
        errors.append("CALLBACK_URL must be publicly reachable (not localhost)")
    if base_url and not _is_public_url(base_url):
        errors.append("BASE_URL must be publicly reachable (not localhost)")

    # Optional reachability probe
    if CHECK_CALLBACK_REACHABLE and callback_url and _is_public_url(callback_url):
        try:
            import requests

            resp = requests.head(callback_url, timeout=3, allow_redirects=True)
            if resp.status_code >= 400:
                errors.append(f"CALLBACK_URL HEAD check failed with {resp.status_code}")
        except Exception as probe_err:
            errors.append(f"CALLBACK_URL HEAD check error: {probe_err}")

    ENV_VALIDATION_ERRORS = errors
    ENV_VALIDATION_OK = len(errors) == 0

    if ENV_VALIDATION_OK:
        logger.info("Environment validation passed")
    else:
        logger.error(f"Environment validation failed: {errors}")

    return ENV_VALIDATION_OK


def ensure_environment_ready():
    """Re-run validation if needed and raise to caller for user-facing 500 responses."""
    if not ENV_VALIDATION_OK:
        validate_environment()
    if not ENV_VALIDATION_OK:
        raise RuntimeError("; ".join(ENV_VALIDATION_ERRORS))


def register_call_state(call_id: str, state: CredentialingState):
    """Store call state by call_id and call_sid (if present) with a light lock.
    Merges with any existing state so fields not returned by LangGraph are preserved."""
    with call_state_lock:
        existing = call_states.get(call_id, {})
        merged = {**existing, **state}
        call_states[call_id] = merged
        if merged.get('call_sid'):
            call_states_by_sid[merged['call_sid']] = merged


def bind_call_sid(call_id: str, call_sid: str):
    """Bind a call_sid to existing state for faster lookups."""
    with call_state_lock:
        if call_id in call_states:
            call_states[call_id]['call_sid'] = call_sid
            call_states_by_sid[call_sid] = call_states[call_id]
    _update_twilio_number_call_sid(call_id, call_sid)


def get_state_by_sid(call_sid: str) -> Optional[CredentialingState]:
    """Fetch state by call_sid, falling back to linear scan."""
    with call_state_lock:
        if call_sid and call_sid in call_states_by_sid:
            return call_states_by_sid[call_sid]
        for state in call_states.values():
            if state.get('call_sid') == call_sid:
                return state
    return None


# Run env validation once at startup to surface issues early
validate_environment()


# =============================================================================
# ElevenLabs TTS Functions
# =============================================================================

def generate_elevenlabs_audio_url(text: str) -> Optional[str]:
    """
    Generate audio with ElevenLabs and save to temp file.
    Returns URL accessible by Twilio.
    """
    if not ENABLE_ELEVENLABS_TTS:
        logger.info("ElevenLabs TTS disabled via ENABLE_ELEVENLABS_TTS=false")
        return None

    if not _is_public_url(BASE_URL):
        logger.warning("BASE_URL is not public; skipping ElevenLabs playback and using <Say> fallback")
        return None

    try:
        # Generate audio using the text_to_speech API
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2",
        )

        # Save to temp file
        temp_dir = os.path.join(os.getcwd(), 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)

        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(temp_dir, filename)

        with open(filepath, 'wb') as f:
            for chunk in audio_generator:
                if chunk:
                    f.write(chunk)

        # Return URL (needs to be publicly accessible)
        audio_url = f"{BASE_URL}/audio/{filename}"
        logger.info(f"Generated ElevenLabs audio: {audio_url}")
        return audio_url

    except Exception as e:
        logger.error(f"ElevenLabs TTS failed: {e}")
        return None  # Fallback to Polly


def speak_with_tts(response, text: str, gather=None):
    """
    Speak text using ElevenLabs, with Polly fallback.
    Works with both VoiceResponse and Gather objects.
    """
    target = gather if gather else response

    # Try ElevenLabs first
    audio_url = generate_elevenlabs_audio_url(text)

    if audio_url:
        target.play(audio_url)
        logger.info(f"Using ElevenLabs TTS for: {text[:50]}...")
    else:
        # Fallback to Polly
        target.say(text, voice='Polly.Joanna')
        logger.warning(f"Falling back to Polly TTS for: {text[:50]}...")


def normalize_ivr_action(preferred_action, action_value):
    """
    Normalize IVR action rows before saving or executing them.

    This protects existing rows that were entered with UI words in the value,
    such as preferred_action='dtmf' and action_value='Say credentialing'.
    """
    action = str(preferred_action or 'dtmf').strip().lower()
    value = '' if action_value is None else str(action_value).strip()

    if action not in ('dtmf', 'speech', 'wait'):
        action = 'dtmf'

    if action == 'wait':
        return action, None

    while value:
        original_value = value
        if action != 'speech':
            value = re.sub(
                r'^(?:press|dial|enter|choose option|option)\s+',
                '',
                value,
                flags=re.IGNORECASE,
            ).strip()
        if re.match(r'^(?:say|speak)\s+', value, flags=re.IGNORECASE):
            action = 'speech'
            value = re.sub(r'^(?:say|speak)\s+', '', value, flags=re.IGNORECASE).strip()
        if value == original_value:
            break

    if action == 'dtmf' and value and re.search(r'[^0-9*#wW\s]', value):
        logger.warning(f"IVR DTMF action has non-DTMF value; treating as speech: {value}")
        action = 'speech'

    return action, value or None


def _digits_only(value) -> str:
    return re.sub(r'\D', '', str(value or ''))


def _normalize_us_phone(value) -> Optional[str]:
    """Normalize a US phone number to E.164, or return None if invalid."""
    digits = _digits_only(value)
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return None


def _same_us_phone(left, right) -> bool:
    left_normalized = _normalize_us_phone(left)
    right_normalized = _normalize_us_phone(right)
    return bool(left_normalized and right_normalized and left_normalized == right_normalized)


def get_provider_ivr_flags(db, insurance_name: str, insurance_phone: str = None):
    """
    Load IVR auto-response flags without mixing similarly named insurers.
    Exact normalized name and exact phone matches win; fuzzy name matching is
    only a last fallback for legacy rows.
    """
    select_with_suffix = """
        SELECT insurance_name, phone_number,
               ivr_asks_npi, ivr_npi_method, ivr_asks_tax_id, ivr_tax_id_method,
               ivr_tax_id_digits_to_send, ivr_npi_suffix, ivr_tax_id_suffix
        FROM insurance_providers
    """
    select_without_suffix = """
        SELECT insurance_name, phone_number,
               ivr_asks_npi, ivr_npi_method, ivr_asks_tax_id, ivr_tax_id_method,
               ivr_tax_id_digits_to_send, NULL AS ivr_npi_suffix, NULL AS ivr_tax_id_suffix
        FROM insurance_providers
    """
    phone_digits = _digits_only(insurance_phone)

    def _fetch(select_sql):
        with db.conn.cursor() as cur:
            cur.execute(
                select_sql + """
                WHERE lower(trim(insurance_name)) = lower(trim(%s))
                   OR (%s <> '' AND regexp_replace(COALESCE(phone_number, ''), '\\D', '', 'g') = %s)
                ORDER BY
                    CASE
                        WHEN lower(trim(insurance_name)) = lower(trim(%s)) THEN 0
                        WHEN %s <> '' AND regexp_replace(COALESCE(phone_number, ''), '\\D', '', 'g') = %s THEN 1
                        ELSE 2
                    END,
                    last_updated DESC NULLS LAST
                LIMIT 1
                """,
                (insurance_name, phone_digits, phone_digits, insurance_name, phone_digits, phone_digits),
            )
            row = cur.fetchone()
            match_type = 'exact_or_phone'
            if not row:
                cur.execute(
                    select_sql + """
                    WHERE insurance_name ILIKE %s
                    ORDER BY last_updated DESC NULLS LAST
                    LIMIT 1
                    """,
                    (f"%{insurance_name}%",),
                )
                row = cur.fetchone()
                match_type = 'fuzzy'
            return row, match_type

    try:
        row, match_type = _fetch(select_with_suffix)
    except Exception:
        db.conn.rollback()
        row, match_type = _fetch(select_without_suffix)

    if not row:
        return None

    if match_type == 'fuzzy':
        logger.warning(
            f"Provider IVR flags used fuzzy match for requested insurance='{insurance_name}': "
            f"selected '{row[0]}' ({row[1]})"
        )

    return {
        'insurance_name': row[0],
        'phone_number': row[1],
        'ivr_asks_npi': row[2] or False,
        'ivr_npi_method': row[3] or 'speech',
        'ivr_asks_tax_id': row[4] or False,
        'ivr_tax_id_method': row[5] or 'speech',
        'ivr_tax_id_digits_to_send': row[6],
        'ivr_npi_suffix': row[7] or '',
        'ivr_tax_id_suffix': row[8] or '',
    }


# =============================================================================
# Deepgram streaming — emit <Connect><Stream> instead of <Gather> when enabled,
# and bridge final transcripts back into the existing /webhook/speech logic.
# =============================================================================

def _ws_url_from_request():
    """Derive a wss:// URL for the /media-stream namespace from the current request."""
    try:
        host = request.host
    except Exception:
        host = (BASE_URL or "").replace("https://", "").replace("http://", "")
    return f"wss://{host}/media-stream"


def emit_human_listening(response, callback_base, call_id, gather_action='/webhook/speech'):
    """
    Append the right TwiML to capture human speech.

    With ENABLE_DEEPGRAM_STREAMING=true, emits <Connect><Stream> so audio flows
    into Deepgram. Otherwise emits the original <Gather input='speech'>.

    NOTE: <Connect><Stream> takes over the call — no TwiML after it runs.
    Always speak (ElevenLabs / Polly) BEFORE calling this.
    """
    if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available():
        connect = Connect()
        stream = Stream(url=_ws_url_from_request())
        stream.parameter(name='call_id', value=str(call_id))
        connect.append(stream)
        response.append(connect)
        logger.info(f"Listening via Deepgram stream for call_id={call_id}")
        return None  # caller doesn't need a Gather handle
    else:
        from twilio.twiml.voice_response import Gather
        gather = Gather(
            input='speech',
            action=f'{callback_base}{gather_action}',
            method='POST',
            speech_timeout='auto',
            language='en-US',
        )
        response.append(gather)
        return gather


def _on_deepgram_final(call_sid: str, transcript: str):
    """
    Called by Deepgram when a final transcript is ready. Re-runs the existing
    /webhook/speech logic by asking Twilio to fetch it again, passing the
    transcript as a query param. The webhook's response (with ElevenLabs <Play>
    and the next listener) is post-processed to swap any <Gather> for
    <Connect><Stream> so streaming continues for the next turn.
    """
    try:
        from urllib.parse import urlencode
        callback_base = (os.getenv("CALLBACK_URL", "")
                         .replace("/webhook/voice", "")
                         .rstrip("/"))
        if not callback_base:
            callback_base = BASE_URL
        qs = urlencode({"SpeechResult": transcript, "CallSid": call_sid,
                        "DeepgramSource": "1"})
        next_url = f"{callback_base}/webhook/deepgram-process?{qs}"
        logger.info(f"Deepgram final → redirecting call {call_sid} to {next_url[:120]}")
        twilio_client.calls(call_sid).update(url=next_url, method='POST')
    except Exception as e:
        logger.error(f"_on_deepgram_final failed for {call_sid}: {e}", exc_info=True)


@app.route('/webhook/deepgram-process', methods=['GET', 'POST'])
def deepgram_process_webhook():
    """
    Internal endpoint Twilio is redirected to after Deepgram fires a final
    transcript. Forwards the request through the existing speech_webhook,
    then rewrites any <Gather> in the response to <Connect><Stream> so the
    streaming loop continues. ElevenLabs <Play> elements pass through unchanged.
    """
    import re
    # Reuse the existing speech webhook logic (it reads SpeechResult / CallSid
    # from request.values, which includes both query string and form body).
    twiml_response = speech_webhook()
    try:
        xml = twiml_response.get_data(as_text=True)
    except Exception:
        xml = str(twiml_response)

    if not ENABLE_DEEPGRAM_STREAMING or '<Gather' not in xml:
        return Response(xml, mimetype='text/xml')

    # Pull call_id from in-memory state for the parameter.
    call_sid = request.values.get('CallSid', '')
    call_id = call_sid
    for cid, st in call_states.items():
        if st.get('call_sid') == call_sid:
            call_id = cid
            break

    _set_log_context(call_id)

    ws_url = _ws_url_from_request()
    stream_block = (
        f'<Connect><Stream url="{ws_url}">'
        f'<Parameter name="call_id" value="{xml_escape(str(call_id))}"/>'
        f'</Stream></Connect>'
    )

    # Strip the <Gather ...>...</Gather> wrapper and replace with the inner
    # contents (Play / Say) followed by the Stream block. Anything inside the
    # Gather (the agent's spoken reply) plays first, then Stream takes over.
    pattern = re.compile(r'<Gather[^>]*>(.*?)</Gather>', re.DOTALL)
    xml = pattern.sub(lambda m: m.group(1) + stream_block, xml, count=1)
    # Drop any trailing extra Gathers (rare).
    xml = pattern.sub(lambda m: m.group(1), xml)

    return Response(xml, mimetype='text/xml')


# =============================================================================
# Raw WebSocket endpoint for Twilio Media Streams (replaces the dead
# Flask-SocketIO handler — Twilio uses raw WebSockets, not Socket.IO).
# =============================================================================

if sock is not None:
    @sock.route('/media-stream')
    def media_stream_ws(ws):
        """
        Raw WebSocket handler for Twilio Media Streams.

        Twilio sends a sequence of JSON messages over the WebSocket:
          {event: "connected"}                — protocol greeting
          {event: "start",  start: {callSid, customParameters: {...}}}
          {event: "media",  media: {payload: <base64 mu-law>}}     (many)
          {event: "stop"}
        We forward the audio payloads to a per-call Deepgram session and let
        Deepgram's on_final callback drive the next agent turn.
        """
        socket_sid = uuid.uuid4().hex
        call_sid = None
        stream_sid = None
        logger.info(f"🔌 Twilio WebSocket connected sid={socket_sid}")

        try:
            while True:
                raw = ws.receive(timeout=120)
                if raw is None:
                    break
                try:
                    data = json.loads(raw)
                except Exception:
                    continue

                event_type = data.get('event')

                if event_type == 'connected':
                    logger.info(f"📞 Twilio stream protocol={data.get('protocol')} sid={socket_sid}")

                elif event_type == 'start':
                    start_payload = data.get('start', {}) or {}
                    stream_sid = data.get('streamSid')
                    call_sid = start_payload.get('callSid')
                    custom_params = start_payload.get('customParameters', {}) or {}
                    stream_call_id = custom_params.get('call_id') or call_sid
                    logger.info(f"🎙️ Stream started — StreamSID={stream_sid}, CallSID={call_sid}, "
                                f"call_id={stream_call_id}")

                    _set_log_context(stream_call_id)

                    media_streams[socket_sid] = {
                        'stream_sid': stream_sid,
                        'call_sid': call_sid,
                        'call_id': stream_call_id,
                    }

                    # Mark the conversation phase in call state
                    for cid, st in call_states.items():
                        if st.get('call_sid') == call_sid:
                            st['current_audio_type'] = AudioType.HUMAN_SPEECH
                            break

                    if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available() and call_sid:
                        deepgram_streaming.open_session(
                            socket_sid=socket_sid,
                            call_sid=call_sid,
                            on_final=_on_deepgram_final,
                        )

                elif event_type == 'media':
                    if not (ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available()):
                        continue
                    payload = (data.get('media') or {}).get('payload', '')
                    if not payload:
                        continue
                    try:
                        audio_bytes = base64.b64decode(payload)
                        deepgram_streaming.feed_audio(socket_sid, audio_bytes)
                    except Exception as fe:
                        logger.debug(f"Failed to forward audio chunk: {fe}")

                elif event_type == 'stop':
                    logger.info(f"🛑 Stream stopped: {stream_sid} sid={socket_sid}")
                    break
        except Exception as e:
            logger.error(f"media_stream_ws error sid={socket_sid}: {e}", exc_info=True)
        finally:
            if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available():
                deepgram_streaming.close_session(socket_sid)
            _ws_stream_info = media_streams.pop(socket_sid, None)
            if _ws_stream_info:
                _ws_flush_call_id = _ws_stream_info.get('call_id')
                _ws_flush_call_sid = _ws_stream_info.get('call_sid')
                if _ws_flush_call_id:
                    _flush_call_logs(_ws_flush_call_id, _ws_flush_call_sid)
            logger.info(f"🔌 Twilio WebSocket closed sid={socket_sid}")


_classify_chain = None


def _get_classify_chain():
    """Lazy singleton for the speech classification LLM chain."""
    global _classify_chain
    if _classify_chain is None:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser

        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1,
                         api_key=os.getenv('OPENAI_API_KEY'))
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are classifying live phone call audio for an insurance credentialing system.
Classify the speech transcript into exactly one of:
- human_speech: A real person speaking (agent, representative, or staff member)
- ivr_menu: Automated IVR menu with navigation options
- hold_music: Hold messages, queue announcements, or wait music descriptions
- silence: No meaningful speech

Key signals for human_speech: introduces themselves by name ("my name is"), asks how they can help,
responds naturally to context, uses conversational language.
Key signals for ivr_menu: "press 1", "press 2", numbered options, "say provider", "for billing".
Key signals for hold_music: "all representatives are busy", "estimated wait time", "your call is important",
"please hold", "thank you for your patience".

Return ONLY valid JSON: {{"type": "...", "confidence": 0.0-1.0, "reasoning": "one sentence"}}"""),
            ("user", "Context: {context}\n\nSpeech: {speech}")
        ])
        _classify_chain = prompt | llm | JsonOutputParser()
    return _classify_chain


def _ai_classify_speech(speech_text: str, context: str = ''):
    """
    Use gpt-4o-mini to classify a speech chunk during the wait-for-human phase.
    Returns {"type": "human_speech|ivr_menu|hold_music|silence", "confidence": float, "reasoning": str}
    or None if the AI call fails.
    Only called for agent-mode calls — not on every webhook to control cost/latency.
    """
    try:
        chain = _get_classify_chain()
        return chain.invoke({"context": context, "speech": speech_text})
    except Exception as e:
        logger.warning(f"[ai_classify_speech] AI classification failed: {e}")
        return None


def cleanup_old_audio():
    """Remove audio files older than 5 minutes"""
    while True:
        time.sleep(300)  # Run every 5 minutes
        temp_dir = os.path.join(os.getcwd(), 'temp_audio')
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                filepath = os.path.join(temp_dir, f)
                try:
                    if time.time() - os.path.getmtime(filepath) > 300:
                        os.remove(filepath)
                        logger.debug(f"Cleaned up old audio file: {f}")
                except Exception as e:
                    logger.error(f"Failed to clean up audio file {f}: {e}")


# Start cleanup thread
threading.Thread(target=cleanup_old_audio, daemon=True).start()


def persist_call_progress(state: CredentialingState, finalize: bool = False):
    """
    Persist extracted answers to Supabase so they can be viewed mid-call.
    If finalize=True, also stamps completed_at.
    """
    request_id = state.get('db_request_id') or state.get('call_id')
    if not request_id:
        print("WARN: No request_id available to persist progress")
        return
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        if finalize:
            db.save_final_results(request_id, state)
        else:
            db.update_call_progress(request_id, state)
        db.close()
        print(f"INFO: Persisted call progress for request {request_id}")
    except Exception as e:
        print(f"WARN: Could not persist call progress: {e}")


def _enqueue_call_knowledge_ingestion(
    call_id: Optional[str],
    request_id: Optional[str] = None,
    state: Optional[CredentialingState] = None,
):
    """Summarize a completed AI call and save it to the pgvector knowledge base."""
    def _worker():
        db = None
        try:
            if state and state.get('_knowledge_ingested'):
                return

            canonical_call_id = call_id or (state.get('call_id') if state else None)
            resolved_request_id = request_id or (state.get('db_request_id') if state else None)

            from credentialing_agent import DatabaseManager
            db = DatabaseManager()

            with db.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if canonical_call_id:
                    cur.execute(
                        "SELECT 1 FROM call_knowledge WHERE call_id = %s LIMIT 1",
                        (canonical_call_id,),
                    )
                    if cur.fetchone():
                        if state is not None:
                            state['_knowledge_ingested'] = True
                        return
                elif resolved_request_id:
                    cur.execute(
                        "SELECT 1 FROM call_knowledge WHERE request_id = %s LIMIT 1",
                        (resolved_request_id,),
                    )
                    if cur.fetchone():
                        if state is not None:
                            state['_knowledge_ingested'] = True
                        return

                request_row = None
                if resolved_request_id:
                    cur.execute(
                        """
                        SELECT insurance_name, provider_name, status, reference_number,
                               missing_documents, turnaround_days
                        FROM credentialing_requests
                        WHERE id = %s
                        """,
                        (resolved_request_id,),
                    )
                    request_row = cur.fetchone()

                if canonical_call_id and resolved_request_id:
                    cur.execute(
                        """
                        SELECT speaker, message, timestamp
                        FROM conversation_history
                        WHERE call_id = %s OR request_id = %s
                        ORDER BY timestamp ASC
                        """,
                        (canonical_call_id, resolved_request_id),
                    )
                elif canonical_call_id:
                    cur.execute(
                        """
                        SELECT speaker, message, timestamp
                        FROM conversation_history
                        WHERE call_id = %s
                        ORDER BY timestamp ASC
                        """,
                        (canonical_call_id,),
                    )
                elif resolved_request_id:
                    cur.execute(
                        """
                        SELECT speaker, message, timestamp
                        FROM conversation_history
                        WHERE request_id = %s
                        ORDER BY timestamp ASC
                        """,
                        (resolved_request_id,),
                    )
                else:
                    return

                conversation = [
                    {"speaker": row["speaker"], "message": row["message"], "timestamp": row["timestamp"]}
                    for row in cur.fetchall()
                ]

            insurance_name = (
                (request_row and request_row["insurance_name"])
                or (state.get('insurance_name') if state else None)
            )
            provider_name = (
                (request_row and request_row["provider_name"])
                or (state.get('provider_name') if state else None)
            )
            metadata = {
                "status": (
                    (state.get('credentialing_status') if state else None)
                    or (request_row["status"] if request_row else None)
                ),
                "reference_number": (
                    (state.get('reference_number') if state else None)
                    or (request_row["reference_number"] if request_row else None)
                ),
                "missing_documents": (
                    (state.get('missing_documents') if state else None)
                    or (request_row["missing_documents"] if request_row else [])
                    or []
                ),
                "turnaround_days": (
                    (state.get('turnaround_days') if state else None)
                    or (request_row["turnaround_days"] if request_row else None)
                ),
            }
            transcript = state.get('transcript', []) if state else []

            if not insurance_name:
                logger.warning(
                    f"Skipping knowledge ingestion for call {canonical_call_id or resolved_request_id}: "
                    "missing insurance_name"
                )
                return

            inserted = ingest_call_knowledge(
                insurance_name=insurance_name,
                provider_name=provider_name,
                call_id=canonical_call_id,
                request_id=resolved_request_id,
                conversation=conversation,
                transcript=transcript,
                metadata=metadata,
            )
            if inserted and state is not None:
                state['_knowledge_ingested'] = True
        except Exception as e:
            logger.error(f"Knowledge ingestion worker failed for call {call_id or request_id}: {e}")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass

    threading.Thread(target=_worker, daemon=True).start()

def _log_ivr_event(call_id: str, event_type: str, transcript: str = None,
                    action_taken: str = None, confidence: float = None, metadata: dict = None):
    """Log an IVR navigation event to the call_events table (fire-and-forget).

    Runs in a background thread so it never delays the TwiML response to Twilio.
    """
    def _do_insert():
        try:
            from credentialing_agent import DatabaseManager
            db = DatabaseManager()
            with db.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO call_events (call_id, event_type, transcript, action_taken, confidence, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (call_id, event_type, transcript, action_taken, confidence,
                      json.dumps(metadata) if metadata else None))
            db.conn.commit()
            db.close()
        except Exception as e:
            print(f"WARN: Could not log IVR event: {e}")

    threading.Thread(target=_do_insert, daemon=True).start()


VALID_PHRASE_TYPES = ('human', 'ivr_definitive', 'ivr_passive', 'simple_greeting')

_DETECTION_DEFAULTS = {
    'human': [
        'how can i help', 'how may i help', 'how may i assist',
        'may i help you', 'who am i speaking with', 'may i ask who',
        'what is your name', 'can i get your name',
        'good morning', 'good afternoon', 'good evening',
        'my name is', 'may i have your', 'i can help you',
    ],
    'ivr_definitive': [
        'press 1', 'press 2', 'press 3', 'press 4', 'press 5',
        'press 6', 'press 7', 'press 8', 'press 9', 'press 0',
        'press star', 'press pound', 'press the',
        'press or say',
        'say ', 'option 1', 'option 2', 'option 3', 'option 4',
        'dial ', 'select ',
        'for billing', 'for claims', 'for provider', 'for member',
        'for english', 'for spanish',
        'main menu', 'previous menu', 'return to',
        'if you are a', 'if you know your',
        'are you a member', 'are you a provider',
        'please hang up and dial', 'leave a message', 'voicemail',
    ],
    'ivr_passive': [
        'this call may be monitored', 'this call may be recorded',
        'for quality assurance', 'quality assurance purposes',
        'please hold', 'please wait', 'please listen',
        'your call is important', 'your call will be',
        'all representatives', 'all agents',
        'estimated wait time', 'high call volume',
        'thank you for calling', 'you have reached',
    ],
    'simple_greeting': ['hello', 'hi', 'credentialing', 'speaking'],
}


def _get_detection_defaults() -> dict:
    """Return a fresh copy of hardcoded defaults (no DB query, zero latency)."""
    import copy
    return copy.deepcopy(_DETECTION_DEFAULTS)


def _get_human_detection_phrases(insurance_name: str = None) -> dict:
    """Load learned phrases from DB and merge with hardcoded defaults.

    Returns dict with keys: human, ivr_definitive, ivr_passive, simple_greeting.
    Hardcoded defaults are always included; DB phrases are appended.
    """
    # Hardcoded defaults — always present as baseline (single source of truth)
    defaults = _get_detection_defaults()

    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        with db.conn.cursor() as cur:
            if insurance_name:
                cur.execute("""
                    SELECT phrase, phrase_type FROM human_detection_phrases
                    WHERE is_active = TRUE AND confidence > 0.3
                      AND (insurance_name IS NULL OR insurance_name ILIKE %s)
                    ORDER BY confidence DESC
                """, (f"%{insurance_name}%",))
            else:
                cur.execute("""
                    SELECT phrase, phrase_type FROM human_detection_phrases
                    WHERE is_active = TRUE AND confidence > 0.3
                      AND insurance_name IS NULL
                    ORDER BY confidence DESC
                """)
            for row in cur.fetchall():
                phrase, phrase_type = row[0], row[1]
                if phrase_type in defaults and phrase not in defaults[phrase_type]:
                    defaults[phrase_type].append(phrase)
        db.close()
    except Exception as e:
        print(f"WARN: Could not load dynamic detection phrases: {e}")

    return defaults


# Helper function to format NPI for speech (digit by digit)
def format_npi_for_speech(npi: str) -> str:
    """Format NPI to be read digit by digit: 1234567890 -> 1. 2. 3. 4. 5. 6. 7. 8. 9. 0."""
    return '. '.join(list(npi)) + '.'

def format_tax_id_for_speech(tax_id: str) -> str:
    """Format Tax ID for speech: 12-3456789 -> 1 2, 3 4 5 6 7 8 9"""
    # Remove dashes and format
    clean = tax_id.replace('-', '')
    return '. '.join(list(clean)) + '.'


# GPT-4 Conversation Agent for intelligent responses
class SmartConversationAgent:
    """GPT-4 powered conversation agent for handling calls intelligently"""

    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional AI assistant making a phone call on behalf of a healthcare provider to check on insurance credentialing status.

CRITICAL RULES:
1. You ARE an automated AI assistant - if asked "are you human?", "are you a robot?", "are you real?", "say yes", respond honestly: "I'm an automated assistant calling on behalf of {provider_name}. I can still help with your credentialing inquiry."
2. Be professional, concise, and courteous
3. If asked verification questions (NPI, Tax ID, address), answer accurately using the provider information below
4. Stay focused on the credentialing inquiry
5. Keep responses SHORT (1-2 sentences max) - this is a phone call
6. If you don't understand, politely ask for clarification
7. TRANSFER RESTRICTIONS: If they want to transfer you, politely say you need to ask your questions first. Only agree to transfer AFTER all questions are asked (stage="wrapping_up").
8. When providing NPI, say it DIGIT BY DIGIT with clear pauses: "{npi_spoken}"
9. When providing Tax ID, say it DIGIT BY DIGIT with clear pauses: "{tax_id_spoken}"

Provider Information (USE THIS FOR VERIFICATION):
- Provider Name: {provider_name}
- NPI: When asked, say: "{npi_spoken}"
- Tax ID: When asked, say: "{tax_id_spoken}"
- Address: {address}

Relevant prior answers from previous calls (use to speed things up; don't repeat masked info):
{knowledge}

=== YOUR PRIMARY TASK ===
You MUST ask these credentialing questions one at a time. Current progress: Question {current_question_index} of {total_questions}.

Questions list:
{questions}

=== QUESTION ASKING LOGIC ===
- If current_question_index is 0 and they confirmed they can help, ASK QUESTION #1
- If you just got an answer to a question, ASK THE NEXT QUESTION
- ONLY use action="end_call" when ALL questions have been asked AND answered (stage="wrapping_up")
- If stage is "asking_question_X", you MUST include question X in your response

Current stage: {stage}
(If stage says "asking_question", you MUST ask the next question!)

=== STAGE-SPECIFIC BEHAVIOR ===
- "awaiting_disclosure_confirmation": Confirm they can help, then ask first question
- "initial_contact_ask_question_X": Ask question X - do NOT accept transfers yet
- "wrapping_up": All questions complete - now you may accept transfers or end call

Previous conversation context:
{context}

=== RESPONSE FORMAT ===
Respond ONLY with this JSON (no other text):
{{
    "response": "your spoken response - INCLUDE THE QUESTION if it's time to ask one",
    "action": "continue|end_call|request_transfer",
    "extracted_info": {{"status": "...", "reference": "...", "turnaround": "..."}},
    "question_asked": true/false
}}

=== ACTION RULES ===
- action="continue" - Use for normal conversation
- action="end_call" - Use when:
  * stage="wrapping_up" AND you've said goodbye, OR
  * Human explicitly refuses to work with AI / says they cannot help AI callers (end politely immediately)
- action="request_transfer" - Use ONLY when:
  * All questions have been asked/answered (stage="wrapping_up"), OR
  * Human explicitly refuses to answer after multiple attempts
- question_asked=true - Set to true ONLY when your response includes one of the credentialing questions from the list

Example when asking a question:
{{"response": "Thank you for verifying. Now, can you tell me the current credentialing status for this provider?", "action": "continue", "extracted_info": {{}}, "question_asked": true}}

Example when providing NPI (NOT asking a question):
{{"response": "The NPI is {npi_spoken}", "action": "continue", "extracted_info": {{}}, "question_asked": false}}"""),
            ("user", "The person on the phone said: \"{speech}\"")
        ])

    def generate_response(self, speech: str, state: dict, stage: str = "initial", current_question_index: int = 0) -> dict:
        """Generate intelligent response using GPT-4"""
        try:
            chain = self.response_prompt | self.llm | JsonOutputParser()

            # Fetch relevant knowledge snippets (cached per call after first turn)
            knowledge_text = ""
            cached_knowledge = state.get('_cached_knowledge')
            if cached_knowledge is not None:
                knowledge_text = cached_knowledge
            else:
                try:
                    kb = get_knowledge_base()
                    if kb:
                        snippets = kb.search(
                            insurance_name=state.get('insurance_name', ''),
                            provider_name=state.get('provider_name', None),
                            query=speech,
                            limit=5
                        )
                        if snippets:
                            formatted = []
                            for s in snippets:
                                formatted.append(f"- {s.get('summary','')}")
                            knowledge_text = "\n".join(formatted)
                        # Cache for subsequent turns
                        state['_cached_knowledge'] = knowledge_text
                except EmbeddingError as e:
                    logger.error(f"Embedding failed for knowledge query: {e}")
                    knowledge_text = ""
                except SearchError as e:
                    logger.error(f"Knowledge search failed: {e}")
                    knowledge_text = ""
                except KnowledgeBaseError as e:
                    logger.error(f"Knowledge base error: {e}")
                    knowledge_text = ""
                except Exception as kb_err:
                    logger.warning(f"Unexpected knowledge lookup error: {kb_err}")
                    knowledge_text = ""

            # Get conversation context
            context = ""
            if state.get('transcript'):
                recent = state['transcript'][-5:]  # Last 5 exchanges
                context = "\n".join([f"{t.get('speaker', 'unknown')}: {t.get('text', '')}" for t in recent])

            # Format NPI and Tax ID for speech
            npi = state.get('npi', 'N/A')
            tax_id = state.get('tax_id', 'N/A')
            npi_spoken = format_npi_for_speech(npi) if npi != 'N/A' else 'N/A'
            tax_id_spoken = format_tax_id_for_speech(tax_id) if tax_id != 'N/A' else 'N/A'

            # Get questions info
            questions = state.get('questions', [])
            total_questions = len(questions)

            # Format questions with numbers
            questions_formatted = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(questions)])

            result = chain.invoke({
                "provider_name": state.get('provider_name', 'the provider'),
                "npi_spoken": npi_spoken,
                "tax_id_spoken": tax_id_spoken,
                "address": state.get('address', 'N/A'),
                "questions": questions_formatted,
                "total_questions": total_questions,
                "current_question_index": current_question_index,
                "stage": stage,
                "context": context,
                "knowledge": knowledge_text or "None available",
                "speech": speech
            })

            return result
        except Exception as e:
            print(f"GPT-4 Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback response
            return {
                "response": "I apologize, could you please repeat that?",
                "action": "continue",
                "extracted_info": {},
                "question_asked": False
            }

# Initialize the smart agent
smart_agent = SmartConversationAgent()

# Singleton KnowledgeBase to avoid recreating DB connection + OpenAI client on every request
_kb_instance = None

def get_knowledge_base():
    """Return a shared KnowledgeBase instance, recreating if connection is lost."""
    global _kb_instance
    if _kb_instance is None:
        try:
            _kb_instance = KnowledgeBase()
        except Exception as e:
            logger.warning(f"Could not initialize KnowledgeBase singleton: {e}")
            return None
    else:
        # Check if connection is still alive
        try:
            _kb_instance.conn.cursor().execute("SELECT 1")
        except Exception:
            try:
                _kb_instance = KnowledgeBase()
            except Exception as e:
                logger.warning(f"Could not reinitialize KnowledgeBase: {e}")
                return None
    return _kb_instance


# =============================================================================
# Audio Serving Endpoint (for ElevenLabs TTS)
# =============================================================================

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files for Twilio to play"""
    temp_dir = os.path.join(os.getcwd(), 'temp_audio')
    return send_from_directory(temp_dir, filename, mimetype='audio/mpeg')


@app.route('/api/start-call', methods=['POST'])
@agent_or_above
def start_credentialing_call():
    """
    API endpoint to start a new credentialing call

    Request body:
    {
        "insurance_name": "Blue Cross Blue Shield",
        "provider_name": "Dr. Jane Smith",
        "npi": "1234567890",
        "tax_id": "12-3456789",
        "address": "123 Main St, City, ST 12345",
        "insurance_phone": "+18001234567",
        "questions": ["What is the status?", "Any missing docs?"]
    }
    """
    try:
        data = request.json
        from_number = None
        print(f"\n{'='*50}")
        print(f"📞 NEW CALL REQUEST RECEIVED")
        print(f"{'='*50}")
        print(f"Provider: {data.get('provider_name')}")
        print(f"Insurance: {data.get('insurance_name')}")
        print(f"Phone: {data.get('insurance_phone')}")
        print(f"NPI: {data.get('npi')}")

        # Validate required fields
        required = ['insurance_name', 'provider_name', 'npi', 'tax_id',
                   'address', 'insurance_phone', 'questions']
        for field in required:
            if field not in data:
                print(f"❌ Missing required field: {field}")
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400

        try:
            ensure_environment_ready()
        except Exception as env_err:
            logger.error(f"Environment validation failed: {env_err}")
            return jsonify({
                'success': False,
                'error': f'Environment validation failed: {env_err}'
            }), 500

        # Generate call_id upfront so we can track immediately
        call_id = str(uuid.uuid4())

        # Extract call mode and agent phone (for AI vs real-agent routing)
        call_mode = data.get('call_mode', 'ai')
        if call_mode not in ('ai', 'agent'):
            call_mode = 'ai'

        insurance_phone = _normalize_us_phone(data.get('insurance_phone'))
        if not insurance_phone:
            return jsonify({'success': False, 'error': 'insurance_phone must be a valid US phone number'}), 400
        data['insurance_phone'] = insurance_phone

        agent_phone = None
        raw_agent_phone = data.get('agent_phone')
        if call_mode == 'agent':
            if not raw_agent_phone:
                return jsonify({'success': False, 'error': 'agent_phone is required when call_mode is agent'}), 400
            agent_phone = _normalize_us_phone(raw_agent_phone)
            if not agent_phone:
                return jsonify({'success': False, 'error': 'agent_phone must be a valid US phone number'}), 400
            if _same_us_phone(agent_phone, insurance_phone):
                return jsonify({
                    'success': False,
                    'error': 'Agent phone must be a staff/user phone, not the insurance phone.'
                }), 400

        # Acquire a Twilio number only after request validation that can fail fast.
        from_number = _acquire_twilio_number(call_id)
        if not from_number:
            print(f"❌ All phone lines are currently busy for call: {call_id}")
            return jsonify({
                'success': False,
                'error': 'All phone lines are currently busy. Please try again shortly.'
            }), 503

        # Create initial state
        initiated_by = get_current_user_id()
        initial_state: CredentialingState = {
            'insurance_name': data['insurance_name'],
            'provider_name': data['provider_name'],
            'npi': data['npi'],
            'tax_id': data['tax_id'],
            'address': data['address'],
            'insurance_phone': data['insurance_phone'],
            'provider_phone': data.get('provider_phone'),
            'questions': data['questions'],
            'questions_asked_count': 0,  # Track how many questions have been asked
            'call_id': call_id,
            'call_sid': None,
            'db_request_id': None,
            'initiated_by': initiated_by,
            'call_state': CallState.INITIATING,
            'transcript': [],
            'conversation_history': [],
            'ivr_knowledge': [],
            'current_menu_level': 0,
            'current_audio_type': AudioType.UNKNOWN,
            'confidence': 0.0,
            'last_action': None,
            'retry_count': 0,
            'credentialing_status': None,
            'reference_number': None,
            'missing_documents': [],
            'turnaround_days': None,
            'notes': '',
            'should_continue': True,
            'disclosure_acknowledged': False,
            'error_message': None,
            # AI vs real-agent mode fields
            'call_mode': call_mode,
            'agent_phone': agent_phone,
            'conference_sid': None,
            'transfer_to_agent': False,
            'from_number': from_number,
            # Pre-load learned detection phrases for this insurance
            'detection_phrases': _get_human_detection_phrases(data['insurance_name']),
        }

        # Create DB record up front so answers can be stored during the call
        try:
            from credentialing_agent import DatabaseManager
            db = DatabaseManager()
            request_id = db.save_credentialing_request(initial_state)
            # Persist call_mode and agent_phone into the DB record
            with db.conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE credentialing_requests
                    SET call_mode = %s, agent_phone = %s
                    WHERE id = %s
                    """,
                    (call_mode, agent_phone, request_id)
                )
                db.conn.commit()
            db.close()
            initial_state['db_request_id'] = request_id
            print(f"INFO: Saved credentialing request to DB with id {request_id}")
        except Exception as db_error:
            print(f"WARN: Could not save initial request to DB: {db_error}")
            request_id = None

        # Store state immediately so frontend can track it
        register_call_state(call_id, initial_state)
        print(f"✅ Call ID generated: {call_id}")
        print(f"📊 Call state stored, status: {CallState.INITIATING.value}")
        logger.info(f"[{call_id}] Call initiated | mode: {call_mode}" + (f" | agent_phone: {agent_phone}" if call_mode == 'agent' else ""))

        # Create agent
        agent = CredentialingAgent()

        # Start call in background thread
        def run_call():
            print(f"🔄 Background thread started for call: {call_id}")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print(f"📞 Initiating Twilio call to: {data['insurance_phone']}")
                print(f"[{call_id}] Call mode: {initial_state.get('call_mode', 'ai').upper()}")
                final_state = loop.run_until_complete(agent.process_call(initial_state))
                loop.close()

                # Preserve call_mode and agent_phone from initial_state regardless of
                # what LangGraph returned — these are not part of the graph logic and
                # must survive the state replacement so webhooks can check them.
                final_state['call_mode'] = initial_state.get('call_mode', 'ai')
                final_state['agent_phone'] = initial_state.get('agent_phone')
                final_state['transfer_to_agent'] = initial_state.get('transfer_to_agent', False)

                # Update stored state with final state
                register_call_state(call_id, final_state)
                print(f"📊 call_mode preserved in state: {final_state.get('call_mode')}")
                print(f"✅ Call completed: {call_id}")
                print(f"📊 Final status: {final_state.get('call_state')}")
            except Exception as thread_error:
                print(f"❌ Error in call thread: {thread_error}")
                import traceback
                traceback.print_exc()
            finally:
                latest_state = call_states.get(call_id, {})
                if latest_state.get('transfer_to_agent'):
                    logger.info(f"[{call_id}] Transfer to agent active; preserving Twilio number until terminal status callback")
                else:
                    _release_twilio_number(call_id=call_id)

        thread = threading.Thread(target=run_call)
        thread.start()

        print(f"{'='*50}")
        print(f"✅ CALL INITIATED SUCCESSFULLY")
        print(f"{'='*50}\n")

        return jsonify({
            'success': True,
            'message': 'Call initiated',
            'call_id': call_id,
            'request_id': initial_state.get('db_request_id'),
            'provider': data['provider_name'],
            'insurance': data['insurance_name']
        }), 200

    except Exception as e:
        print(f"❌ Error starting call: {e}")
        import traceback
        traceback.print_exc()
        if 'call_id' in locals() and from_number:
            _release_twilio_number(call_id=call_id)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def _bridge_to_agent(call_info, callback_base):
    """
    Bridge the insurance call into a Twilio Conference and dial the real human agent in.
    Returns a Flask Response with conference TwiML for the insurance leg.
    Called when call_mode='agent' and a human has been detected (or no IVR to navigate).
    """
    call_state_obj = call_info['state']
    conf_name = call_info['call_id']
    agent_phone_number = call_state_obj.get('agent_phone')
    recording_callback_url = build_recording_status_callback(
        callback_base,
        call_id=conf_name,
        request_id=call_state_obj.get('db_request_id'),
    )

    logger.info(f"[bridge_to_agent] Bridging call_id={conf_name} to agent_phone={agent_phone_number}")

    if not agent_phone_number:
        logger.error(f"[bridge_to_agent] agent_phone missing for call_id={conf_name}")
        error_twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Response>'
            '<Say>We are sorry, no agent phone number was configured for this call.</Say>'
            '</Response>'
        )
        return Response(error_twiml, mimetype='text/xml')

    # TwiML for the insurance leg — enters the conference and closes it when insurance hangs up
    conference_twiml = build_conference_twiml(
        conf_name,
        recording_callback_url=recording_callback_url,
        end_on_exit=True,
    )

    # TwiML for the agent leg — enters the same conference
    agent_conference_twiml = build_conference_twiml(
        conf_name,
        recording_callback_url=recording_callback_url,
        end_on_exit=False,
    )

    def _dial_agent_into_conference():
        try:
            from_number = call_state_obj.get('from_number') or os.getenv("TWILIO_PHONE_NUMBER")
            agent_call = twilio_client.calls.create(
                to=agent_phone_number,
                from_=from_number,
                twiml=agent_conference_twiml,
                timeout=30,
            )
            call_state_obj['conference_sid'] = agent_call.sid
            logger.info(f"[bridge_to_agent] Dialed agent {agent_phone_number} into conference {conf_name}, agent call SID={agent_call.sid}")
        except Exception as dial_err:
            logger.error(f"[bridge_to_agent] Failed to dial agent into conference: {dial_err}")
            insurance_call_sid = call_state_obj.get('call_sid')
            if insurance_call_sid:
                failure_twiml = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<Response>'
                    '<Say>We are sorry, we could not connect the agent for this call. Goodbye.</Say>'
                    '<Hangup/>'
                    '</Response>'
                )
                try:
                    twilio_client.calls(insurance_call_sid).update(twiml=failure_twiml)
                except Exception as update_err:
                    logger.error(f"[bridge_to_agent] Failed to update insurance leg after agent dial failure: {update_err}")

    call_state_obj['call_state'] = CallState.SPEAKING_WITH_HUMAN
    call_state_obj['transfer_to_agent'] = True
    threading.Thread(target=_dial_agent_into_conference, daemon=True).start()

    return Response(conference_twiml, mimetype='text/xml')


@app.route('/webhook/voice', methods=['POST'])
def voice_webhook():
    """
    Twilio webhook for incoming/outgoing voice calls
    This is called when the call connects
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()

        # Get call SID and find associated call state
        call_sid = request.values.get('CallSid')
        logger.info(f"[voice_webhook] CallSID={call_sid}")

        # Get the base URL from environment or request
        callback_base = get_callback_base(request)
        logger.info(f"[voice_webhook] callback_base={callback_base}")

        # Find the call state by call_sid
        state = get_state_by_sid(call_sid)
        call_info = None
        if state:
            call_info = {'call_id': state.get('call_id') or state.get('db_request_id') or call_sid, 'state': state}
        else:
            for call_id, st in list(call_states.items()):
                if st.get('call_sid') == call_sid or st.get('call_sid') is None:
                    if st.get('call_sid') is None and call_sid:
                        bind_call_sid(call_id, call_sid)
                        st['call_state'] = CallState.IVR_NAVIGATION
                        logger.info(f"[voice_webhook] Linked CallSID {call_sid} to call_id {call_id}")
                    call_info = {'call_id': call_id, 'state': st}
                    break

        if not call_info:
            logger.warning(f"[voice_webhook] No call state found for CallSID={call_sid}; using minimal disclosure flow")
            fallback_state = {
                'insurance_name': None,
                'provider_name': 'a healthcare provider',
                'call_state': CallState.SPEAKING_WITH_HUMAN,
                'ivr_knowledge': [],
                'current_menu_level': 0,
                'transcript': [],
                'questions': [],
                'questions_asked_count': 0
            }
            call_info = {'call_id': call_sid or 'unknown', 'state': fallback_state}

        if call_info:
            _set_log_context(call_info['call_id'])

        # Check if insurance has IVR patterns configured
        ivr_patterns = []
        insurance_name = None
        if call_info:
            insurance_name = call_info['state'].get('insurance_name')
            if insurance_name:
                try:
                    from credentialing_agent import DatabaseManager
                    db = DatabaseManager()
                    ivr_patterns = db.get_ivr_knowledge(insurance_name)

                    provider_flags = get_provider_ivr_flags(
                        db,
                        insurance_name,
                        call_info['state'].get('insurance_phone') if call_info else None,
                    )
                    if provider_flags and call_info:
                        call_info['state']['ivr_asks_npi'] = provider_flags['ivr_asks_npi']
                        call_info['state']['ivr_npi_method'] = provider_flags['ivr_npi_method']
                        call_info['state']['ivr_asks_tax_id'] = provider_flags['ivr_asks_tax_id']
                        call_info['state']['ivr_tax_id_method'] = provider_flags['ivr_tax_id_method']
                        call_info['state']['ivr_tax_id_digits_to_send'] = provider_flags['ivr_tax_id_digits_to_send']
                        call_info['state']['ivr_npi_suffix'] = provider_flags['ivr_npi_suffix']
                        call_info['state']['ivr_tax_id_suffix'] = provider_flags['ivr_tax_id_suffix']
                        print(
                            f"📋 IVR flags - Provider: {provider_flags['insurance_name']}, "
                            f"NPI: {provider_flags['ivr_asks_npi']} ({provider_flags['ivr_npi_method']}), "
                            f"Tax ID: {provider_flags['ivr_asks_tax_id']} ({provider_flags['ivr_tax_id_method']}), "
                            f"Tax Digits: {provider_flags['ivr_tax_id_digits_to_send']}"
                        )
                    elif not provider_flags:
                        print(f"⚠️ No insurance provider matched '{insurance_name}' — IVR auto-response flags not loaded")

                    db.close()
                    print(f"🎯 Found {len(ivr_patterns)} IVR patterns for {insurance_name}")
                except Exception as e:
                    print(f"⚠️ Could not load IVR patterns: {e}")

        # Get provider name for the disclosure message
        provider_name = "a healthcare provider"
        if call_info:
            provider_name = call_info['state'].get('provider_name', provider_name)

        # If IVR patterns exist, we need to navigate through them first
        if ivr_patterns and len(ivr_patterns) > 0:
            print(f"🤖 IVR Navigation Mode - Will navigate through {len(ivr_patterns)} menu levels")

            # Store IVR patterns in state for tracking
            call_info['state']['ivr_knowledge'] = ivr_patterns
            call_info['state']['current_menu_level'] = 0

            # Wait briefly then start IVR navigation
            # First, gather input (speech + dtmf) to detect IVR prompts
            gather = Gather(
                input='speech dtmf',
                action=f'{callback_base}/webhook/ivr-navigate',
                method='POST',
                speech_timeout='2',
                timeout=30,
                language='en-US'
            )
            response.append(gather)
            response.redirect(f'{callback_base}/webhook/ivr-navigate')
        else:
            # No IVR patterns - go straight to human conversation (or bridge agent)
            print(f"📞 Direct Human Mode - No IVR patterns configured")

            # If real human agent mode, bridge agent immediately (no IVR to navigate)
            if call_info['state'].get('call_mode') == 'agent':
                print(f"🔀 Real Human Agent mode - bridging agent directly (no IVR)")
                return _bridge_to_agent(call_info, callback_base)

            call_info['state']['call_state'] = CallState.SPEAKING_WITH_HUMAN

            # AI Disclosure message (required for automated calls)
            provider_phone = call_info['state'].get('provider_phone', '')
            phone_part = f" at {provider_phone}" if provider_phone else ""
            disclosure = f"Hello, I'm calling on behalf of {provider_name}{phone_part} for credentialing. Am I speaking with the credentialing department?"

            # Use Gather to capture the response
            gather = Gather(
                input='speech',
                action=f'{callback_base}/webhook/speech',
                method='POST',
                speech_timeout='auto',
                language='en-US'
            )
            speak_with_tts(response, disclosure, gather=gather)
            response.append(gather)

            # If no input, say goodbye
            speak_with_tts(response, "I didn't hear a response. I'll try again later. Goodbye.")

        print(f"📤 Returning TwiML: {str(response)[:200]}...")
        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in voice_webhook: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple error response
        response = VoiceResponse()
        speak_with_tts(response, "Sorry, there was a technical error. Please try again later.")
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/voice/human', methods=['POST'])
def voice_human_webhook():
    """
    Voice webhook for when we've passed IVR and reached a human
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()

        call_sid = request.values.get('CallSid')

        # Get the base URL
        callback_base = get_callback_base(request)
        logger.info(f"[voice_human_webhook] CallSID={call_sid} callback_base={callback_base}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                break

        if call_info:
            _set_log_context(call_info['call_id'])

        # If call_mode is 'agent', bridge to the real human agent instead of AI
        if call_info and call_info['state'].get('call_mode') == 'agent':
            print(f"🔀 Real Human Agent mode - redirecting to bridge-agent from voice/human webhook")
            return _bridge_to_agent(call_info, callback_base)

        provider_name = "a healthcare provider"
        provider_phone = ""
        if call_info:
            provider_name = call_info['state'].get('provider_name', provider_name)
            provider_phone = call_info['state'].get('provider_phone', '')

        # AI Disclosure message
        phone_part = f" at {provider_phone}" if provider_phone else ""
        disclosure = f"Hello, I'm calling on behalf of {provider_name}{phone_part} for credentialing. Am I speaking with the credentialing department?"

        if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available():
            # Speak the disclosure first via ElevenLabs, then hand the call to
            # Deepgram via <Connect><Stream>. (Stream takes over — no TwiML
            # after it will run.)
            speak_with_tts(response, disclosure)
            emit_human_listening(response, callback_base, call_info['call_id']
                                 if call_info else 'unknown')
        else:
            gather = Gather(
                input='speech',
                action=f'{callback_base}/webhook/speech',
                method='POST',
                speech_timeout='auto',
                language='en-US'
            )
            speak_with_tts(response, disclosure, gather=gather)
            response.append(gather)
            speak_with_tts(response, "I didn't hear a response. I'll try again later. Goodbye.")

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in voice_human_webhook: {e}")
        response = VoiceResponse()
        speak_with_tts(response, "Sorry, there was a technical error. Goodbye.")
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/bridge-agent', methods=['POST'])
def bridge_agent_webhook():
    """
    Called by Twilio (via redirect) after a human is detected on an agent-mode call.
    Moves the insurance call into a Twilio Conference and dials the real human agent in.
    """
    try:
        call_sid = request.values.get('CallSid')
        callback_base = get_callback_base(request)
        logger.info(f"[bridge_agent_webhook] CallSID={call_sid}")

        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                break

        if call_info:
            _set_log_context(call_info['call_id'])

        if not call_info:
            logger.error(f"[bridge_agent_webhook] No call state found for CallSID={call_sid}")
            error_twiml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<Response><Say>Sorry, there was a technical error. Goodbye.</Say></Response>'
            )
            return Response(error_twiml, mimetype='text/xml')

        return _bridge_to_agent(call_info, callback_base)

    except Exception as e:
        logger.error(f"[bridge_agent_webhook] Error: {e}")
        import traceback
        traceback.print_exc()
        error_twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Response><Say>Sorry, there was a technical error. Goodbye.</Say></Response>'
        )
        return Response(error_twiml, mimetype='text/xml')


@app.route('/webhook/ivr-navigate', methods=['POST'])
def ivr_navigate_webhook():
    """
    Handle IVR navigation - detect prompts and send DTMF tones
    """
    try:
        from twilio.twiml.voice_response import Gather, Play

        response = VoiceResponse()

        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult', '')
        digits = request.values.get('Digits', '')

        # Get the base URL
        callback_base = os.getenv('CALLBACK_URL', '').replace('/webhook/voice', '')
        if not callback_base:
            callback_base = f"https://{request.host}"

        print(f"🎯 IVR Navigate - CallSID: {call_sid}")
        print(f"📝 Speech: {speech_result}")
        print(f"🔢 Digits: {digits}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                break

        if call_info:
            _set_log_context(call_info['call_id'])
            state = call_info['state']
            ivr_patterns = state.get('ivr_knowledge', [])
            current_level = state.get('current_menu_level', 0)

            # Check if speech matches any IVR pattern
            matched_pattern = None
            speech_lower = speech_result.lower()

            # Log the speech for debugging
            if speech_lower:
                print(f"🔍 Analyzing speech: '{speech_lower[:100]}...'")
                print(f"📊 Current menu level: {current_level}, Available patterns: {len(ivr_patterns)}")

            # ── Credential confirmation check ────────────────────────────────
            # After sending NPI or Tax ID, some IVR systems read it back and ask
            # the caller to confirm (e.g. "Is that correct? Say yes or press 1.").
            awaiting_conf = state.get('awaiting_credential_confirmation')
            if awaiting_conf and speech_lower:
                confirm_keywords = [
                    r'\bcorrect\b', r'\bconfirm\b', r'\bis that right\b',
                    r'\bsay yes\b', r'\bpress 1\b', r'\bpress one\b', r'\bverif'
                ]
                is_confirmation_prompt = any(
                    re.search(kw, speech_lower) for kw in confirm_keywords
                )
                if is_confirmation_prompt:
                    print(f"✅ IVR asking to confirm {awaiting_conf} — responding yes")
                    _log_ivr_event(
                        state.get('db_request_id') or call_info['call_id'],
                        'ivr_credential_confirmed',
                        transcript=speech_result,
                        action_taken=f"confirmed {awaiting_conf}",
                        metadata={'awaiting': awaiting_conf}
                    )
                    state['awaiting_credential_confirmation'] = None
                    response.pause(length=1)
                    if re.search(r'press\s*1', speech_lower):
                        response.play(digits='1')
                        print(f"🔢 Sent DTMF 1 to confirm {awaiting_conf}")
                    else:
                        speak_with_tts(response, 'Yes.')
                        print(f"🗣️ Spoke 'Yes' to confirm {awaiting_conf}")
                    response.pause(length=1)
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='2',
                        timeout=30,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/ivr-navigate')
                    return Response(str(response), mimetype='text/xml')
                else:
                    # IVR moved on — clear the flag so it doesn't persist
                    state['awaiting_credential_confirmation'] = None
            # ── End credential confirmation check ────────────────────────────

            # ── NPI / Tax ID auto-response ──────────────────────────────────
            # If the insurance provider is flagged as asking for NPI or Tax ID
            # during IVR navigation, detect the request and respond automatically.
            ivr_asks_npi = state.get('ivr_asks_npi', False)
            ivr_asks_tax_id = state.get('ivr_asks_tax_id', False)

            if ivr_asks_npi or ivr_asks_tax_id:
                npi_keywords = ['npi', 'national provider', 'provider number', 'provider identification']
                tax_keywords = ['tax id', 'tax identification', r'\bein\b', 'employer identification',
                                'federal tax', 'tax number', 'taxpayer']

                npi_detected = ivr_asks_npi and any(re.search(kw, speech_lower) for kw in npi_keywords)
                tax_detected = ivr_asks_tax_id and any(re.search(kw, speech_lower) for kw in tax_keywords)

                if npi_detected or tax_detected:
                    print(f"📋 IVR requesting credentials - NPI: {npi_detected}, Tax ID: {tax_detected}")
                    cred_types = []
                    if npi_detected: cred_types.append('NPI')
                    if tax_detected: cred_types.append('Tax ID')
                    _log_ivr_event(
                        state.get('db_request_id') or call_id,
                        'ivr_credential_sent',
                        transcript=speech_result,
                        action_taken=f"sent {'+'.join(cred_types)} via IVR",
                        metadata={'npi_detected': npi_detected, 'tax_detected': tax_detected}
                    )

                    # ── Helper: extract "press X" digit for a credential type ──
                    # When the IVR says "press 1 for NPI, press 2 for provider ID...",
                    # we need to press the selection digit BEFORE entering the actual number.
                    def _extract_selection_digit(speech, keywords):
                        """Find the DTMF digit associated with a keyword in a menu prompt.
                        Looks for patterns like 'npi number press 1' or 'press 1 for npi'."""
                        for kw in keywords:
                            # Pattern: "keyword ... press X" (keyword before press)
                            m = re.search(rf'{kw}[\w\s,.]{{0,30}}press[\s,]*(\d)', speech)
                            if m:
                                return m.group(1)
                            # Pattern: "press X ... keyword" (press before keyword)
                            m = re.search(rf'press[\s,]*(\d)[\w\s,.]{{0,30}}{kw}', speech)
                            if m:
                                return m.group(1)
                        return None

                    # Check if this is a multi-option menu (has multiple "press X" options)
                    press_options = re.findall(r'press[\s,]*\d', speech_lower)
                    is_selection_menu = len(press_options) > 1

                    # Pause 2 seconds before responding
                    response.pause(length=2)

                    if npi_detected and state.get('npi'):
                        # If it's a selection menu, press the digit for NPI first
                        if is_selection_menu:
                            select_digit = _extract_selection_digit(speech_lower, npi_keywords)
                            if select_digit:
                                response.play(digits=select_digit)
                                print(f"🔢 Pressed {select_digit} to select NPI entry")
                                response.pause(length=3)

                        method = state.get('ivr_npi_method', 'speech')
                        if method == 'dtmf':
                            npi_digits = state['npi'] + state.get('ivr_npi_suffix', '')
                            response.play(digits=npi_digits)
                            print(f"🔢 Sent NPI via DTMF: {npi_digits}")
                        else:
                            npi_spoken = format_npi_for_speech(state['npi'])
                            speak_with_tts(response, npi_spoken)
                            print(f"🔢 Spoke NPI: {npi_spoken}")
                        state['awaiting_credential_confirmation'] = 'npi'
                        state['last_credential_sent'] = state['npi']

                    elif tax_detected and state.get('tax_id'):
                        # If it's a selection menu, press the digit for Tax ID first
                        if is_selection_menu:
                            select_digit = _extract_selection_digit(speech_lower, tax_keywords)
                            if select_digit:
                                response.play(digits=select_digit)
                                print(f"🔢 Pressed {select_digit} to select Tax ID entry")
                                response.pause(length=3)

                        method = state.get('ivr_tax_id_method', 'speech')
                        clean_tax = re.sub(r'\D', '', state['tax_id'])
                        digits_to_send = state.get('ivr_tax_id_digits_to_send')
                        selected_tax = clean_tax

                        if isinstance(digits_to_send, int):
                            if 1 <= digits_to_send <= len(clean_tax):
                                selected_tax = clean_tax[-digits_to_send:]
                            else:
                                print(f"⚠️ Invalid ivr_tax_id_digits_to_send={digits_to_send}; falling back to full Tax ID")

                        if method == 'dtmf':
                            tax_digits = selected_tax + state.get('ivr_tax_id_suffix', '')
                            response.play(digits=tax_digits)
                            print(f"🔢 Sent Tax ID via DTMF: {tax_digits}")
                        else:
                            tax_spoken = '. '.join(list(selected_tax)) + '.'
                            speak_with_tts(response, tax_spoken)
                            print(f"🔢 Spoke Tax ID: {tax_spoken}")
                        state['awaiting_credential_confirmation'] = 'tax_id'
                        state['last_credential_sent'] = selected_tax

                    # Continue listening for more IVR prompts
                    response.pause(length=1)
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='2',
                        timeout=30,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/ivr-navigate')
                    return Response(str(response), mimetype='text/xml')
            # ── End NPI / Tax ID auto-response ──────────────────────────────

            for pattern in ivr_patterns:
                pattern_level = pattern['menu_level']
                # Check patterns at current level or next level
                if pattern_level == current_level + 1:
                    detected_phrase = pattern.get('detected_phrase', '').lower()
                    if detected_phrase and detected_phrase in speech_lower:
                        matched_pattern = pattern
                        print(f"✅ Matched IVR pattern: '{detected_phrase}' at level {pattern_level}")
                        break

            # If no exact match, try flexible matching for common IVR menu patterns
            if not matched_pattern and speech_lower:
                # Check all remaining patterns at or above current level (not just next level)
                # so same-level patterns aren't skipped when current_level starts at 0.
                next_level_patterns = [p for p in ivr_patterns if p['menu_level'] == current_level + 1]

                for pattern in next_level_patterns:
                    _, action_value = normalize_ivr_action(
                        pattern.get('preferred_action', 'dtmf'),
                        pattern.get('action_value', '')
                    )
                    action_value_lower = action_value.lower() if action_value else ''
                    # Try to detect menu options like "press 1" or "press 2"
                    if action_value_lower and f"press {action_value_lower}" in speech_lower:
                        matched_pattern = pattern
                        print(f"✅ Flexible match: detected 'press {action_value}' at level {pattern['menu_level']}")
                        break
                    # Try "option 1" or "dial 1"
                    elif action_value_lower and (f"option {action_value_lower}" in speech_lower or f"dial {action_value_lower}" in speech_lower):
                        matched_pattern = pattern
                        print(f"✅ Flexible match: detected option/dial {action_value} at level {pattern['menu_level']}")
                        break
                    # Try "say provider" or "say credentialing" (speech-based IVR prompts)
                    elif action_value_lower and f"say {action_value_lower}" in speech_lower:
                        matched_pattern = pattern
                        print(f"✅ Flexible match: detected 'say {action_value}' at level {pattern['menu_level']}")
                        break

            if matched_pattern:
                # Execute the matched action
                action, action_value = normalize_ivr_action(
                    matched_pattern.get('preferred_action', 'dtmf'),
                    matched_pattern.get('action_value', '')
                )

                print(f"🎮 Executing IVR action: {action} = {action_value}")
                _log_ivr_event(
                    state.get('db_request_id') or call_id,
                    'ivr_menu_matched',
                    transcript=speech_result,
                    action_taken=f"{action}:{action_value}",
                    metadata={'menu_level': matched_pattern['menu_level'],
                              'detected_phrase': matched_pattern.get('detected_phrase', '')}
                )

                # Wait 2 seconds after the menu prompt finishes before responding.
                # This lets the IVR complete its full prompt before DTMF/speech is sent.
                response.pause(length=2)

                if action == 'dtmf' and action_value:
                    # Send DTMF tones
                    response.play(digits=action_value)
                    print(f"📲 Sending DTMF: {action_value}")
                elif action == 'speech' and action_value:
                    # Say a phrase
                    speak_with_tts(response, action_value)
                    print(f"🗣️ Saying: {action_value}")

                # Update the current menu level and reset the unmatched counter
                # so subsequent human detection reacts quickly after navigation.
                state['current_menu_level'] = matched_pattern['menu_level']
                state['ivr_unmatched_count'] = 0

                # Check if we have more IVR levels to navigate
                remaining_patterns = [p for p in ivr_patterns if p['menu_level'] > state['current_menu_level']]

                if remaining_patterns:
                    # Continue IVR navigation
                    response.pause(length=1)
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='2',
                        timeout=30,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/ivr-navigate')
                else:
                    # All IVR levels done - enter silent listening mode
                    print(f"✅ IVR navigation complete - entering silent listening mode")
                    state['call_state'] = CallState.WAITING_FOR_HUMAN
                    state['wait_start_time'] = utcnow_iso()

                    # Listen silently for human speech (no Say/Play first)
                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/wait-for-human',
                        method='POST',
                        speech_timeout='2',
                        timeout=15,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/voice/human')
            else:
                # No match - might be IVR announcement, human, or different IVR prompt
                # Track how many times we've heard unmatched speech
                if 'ivr_unmatched_count' not in state:
                    state['ivr_unmatched_count'] = 0
                state['ivr_unmatched_count'] += 1

                print(f"⚠️ No IVR pattern matched (attempt {state['ivr_unmatched_count']})")

                # Load detection phrases from call state (pre-loaded at call start).
                # If not cached (e.g. server restart), use hardcoded defaults only —
                # never hit the DB in the webhook hot path to avoid latency.
                det = state.get('detection_phrases')
                if not det:
                    det = _get_detection_defaults()
                    state['detection_phrases'] = det
                definitive_ivr_indicators = det['ivr_definitive']
                passive_ivr_indicators = det['ivr_passive']
                human_indicators = det['human']
                simple_greetings = det['simple_greeting']

                matched_human = [ind for ind in human_indicators if ind in speech_lower]
                matched_greeting = [g for g in simple_greetings if g in speech_lower and len(speech_result) < 50]
                is_human = bool(matched_human or matched_greeting)

                is_definitive_ivr = any(ind in speech_lower for ind in definitive_ivr_indicators)
                is_passive_ivr    = any(ind in speech_lower for ind in passive_ivr_indicators)
                # Passive indicators only block human detection when no strong human
                # indicator is present. e.g. "Thank you for calling X. How can I help?"
                # contains 'thank you for calling' (passive) + 'how can i help' (human)
                # → is_ivr_system = False → human detected correctly.
                is_ivr_system = is_definitive_ivr or (is_passive_ivr and not is_human)

                # DECISION — priority order matters
                if (
                    state.get('call_mode') == 'agent'
                    and (
                        'connect you with someone' in speech_lower
                        or 'connecting you with someone' in speech_lower
                        or 'please hold while i connect' in speech_lower
                        or 'please hold while we connect' in speech_lower
                    )
                ):
                    print("☎️ Transfer/hold detected in agent mode - waiting for human representative")
                    state['call_state'] = CallState.WAITING_FOR_HUMAN
                    state['wait_start_time'] = utcnow_iso()
                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/wait-for-human',
                        method='POST',
                        speech_timeout='2',
                        timeout=15,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/wait-for-human')
                elif is_ivr_system:
                    # Definitely still in IVR — decrement counter so false positives
                    # don't accumulate toward the human-mode fallback threshold.
                    state['ivr_unmatched_count'] = max(state['ivr_unmatched_count'] - 1, 0)
                    print(f"🤖 IVR system speech detected (not a human), heard: {speech_result[:80]}...")
                    print(f"🔄 Continuing to listen for IVR menu options...")
                    _log_ivr_event(
                        state.get('db_request_id') or call_id,
                        'ivr_system_speech',
                        transcript=speech_result,
                        metadata={'menu_level': current_level}
                    )
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='2',
                        timeout=30,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/ivr-navigate')
                elif is_human and state['ivr_unmatched_count'] >= 1:
                    # Respond immediately on the first human signal.
                    # IVR counter-indicators already filter out pure IVR speech, so
                    # threshold=1 is safe — waiting longer just causes the agent to hang up.
                    print(f"👤 Human detected (clear indicators, attempt {state['ivr_unmatched_count']}) - starting conversation")
                    _log_ivr_event(
                        state.get('db_request_id') or call_id,
                        'human_detected',
                        transcript=speech_result,
                        metadata={'detection': 'clear_indicators', 'attempt': state['ivr_unmatched_count'],
                                  'matched_phrases': matched_human + matched_greeting}
                    )
                    state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                    if state.get('call_mode') == 'agent':
                        print(f"🔀 Real Human Agent mode - redirecting to bridge-agent")
                        response.redirect(f'{callback_base}/webhook/bridge-agent')
                    else:
                        response.redirect(f'{callback_base}/webhook/voice/human')
                elif state['ivr_unmatched_count'] >= 8:
                    # Many unmatched attempts with no IVR counter-signal — assume human or unknown system
                    print(f"👤 Switching to human mode after {state['ivr_unmatched_count']} unmatched attempts")
                    _log_ivr_event(
                        state.get('db_request_id') or call_id,
                        'human_detected',
                        transcript=speech_result,
                        metadata={'detection': 'fallback_threshold', 'attempt': state['ivr_unmatched_count'],
                                  'matched_phrases': matched_human + matched_greeting}
                    )
                    state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                    if state.get('call_mode') == 'agent':
                        print(f"🔀 Real Human Agent mode - redirecting to bridge-agent")
                        response.redirect(f'{callback_base}/webhook/bridge-agent')
                    else:
                        response.redirect(f'{callback_base}/webhook/voice/human')
                else:
                    # Unrecognised speech — keep listening for the IVR menu
                    print(f"🔄 Continuing to listen for IVR menu options...")
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='2',
                        timeout=30,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/ivr-navigate')
        else:
            # No call state found - go to human mode
            response.redirect(f'{callback_base}/webhook/voice/human')

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in ivr_navigate_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        speak_with_tts(response, "Sorry, there was a technical error. Goodbye.")
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/wait-for-human', methods=['POST'])
def wait_for_human_webhook():
    """
    Silent listening phase after IVR - detect when actual human speaks
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()

        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult', '')

        # Get the base URL
        callback_base = os.getenv('CALLBACK_URL', '').replace('/webhook/voice', '')
        if not callback_base:
            callback_base = f"https://{request.host}"

        print(f"🎧 Wait-for-human - CallSID: {call_sid}")
        print(f"📝 Heard: {speech_result}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                break

        if call_info:
            _set_log_context(call_info['call_id'])

        if not call_info:
            # No state found - proceed to human mode
            print(f"⚠️ No call state found - proceeding to human mode")
            response.redirect(f'{callback_base}/webhook/voice/human')
            return Response(str(response), mimetype='text/xml')

        state = call_info['state']
        speech_lower = speech_result.lower()

        # Track wait attempts
        if 'wait_attempts' not in state:
            state['wait_attempts'] = 0
        state['wait_attempts'] += 1

        # Empty/very short speech might be hold music blips or IVR fragments
        # Threshold < 4 so single words like "Hello" (5 chars) still pass
        is_very_short = len(speech_result.strip()) < 4

        # ── NPI / Tax ID auto-response during wait phase ─────────────────────────
        # Some IVRs ask for NPI after routing to the provider queue. Respond with
        # DTMF and continue listening rather than treating it as a human signal.
        npi_keywords = ['npi', 'national provider', 'provider number', 'provider identification']
        tax_keywords = ['tax id', 'tax identification', r'\bein\b', 'employer identification',
                        'federal tax', 'tax number', 'taxpayer']
        npi_detected_wait = any(re.search(kw, speech_lower) for kw in npi_keywords)
        tax_detected_wait = any(re.search(kw, speech_lower) for kw in tax_keywords)

        if (npi_detected_wait or tax_detected_wait) and speech_lower:
            print(f"📋 IVR credential prompt detected during wait phase - NPI: {npi_detected_wait}, Tax ID: {tax_detected_wait}")

            # Helper: extract "press X" digit for a credential type from menu speech
            def _extract_selection_digit_wait(speech, keywords):
                for kw in keywords:
                    m = re.search(rf'{kw}[\w\s,.]{{0,30}}press[\s,]*(\d)', speech)
                    if m:
                        return m.group(1)
                    m = re.search(rf'press[\s,]*(\d)[\w\s,.]{{0,30}}{kw}', speech)
                    if m:
                        return m.group(1)
                return None

            press_options = re.findall(r'press[\s,]*\d', speech_lower)
            is_selection_menu = len(press_options) > 1

            response.pause(length=1)
            if npi_detected_wait and state.get('npi'):
                # If it's a selection menu, press the digit for NPI first
                if is_selection_menu:
                    select_digit = _extract_selection_digit_wait(speech_lower, npi_keywords)
                    if select_digit:
                        response.play(digits=select_digit)
                        print(f"🔢 Pressed {select_digit} to select NPI entry (wait phase)")
                        response.pause(length=3)

                npi_method = state.get('ivr_npi_method', 'dtmf')
                if npi_method == 'dtmf':
                    npi_digits = state['npi'] + state.get('ivr_npi_suffix', '')
                    response.play(digits=npi_digits)
                    print(f"🔢 Sent NPI via DTMF during wait: {npi_digits}")
                else:
                    npi_spoken = format_npi_for_speech(state['npi'])
                    speak_with_tts(response, npi_spoken)
                    print(f"🗣️ Spoke NPI during wait: {npi_spoken}")
            if tax_detected_wait and state.get('tax_id'):
                # If it's a selection menu, press the digit for Tax ID first
                if is_selection_menu:
                    select_digit = _extract_selection_digit_wait(speech_lower, tax_keywords)
                    if select_digit:
                        response.play(digits=select_digit)
                        print(f"🔢 Pressed {select_digit} to select Tax ID entry (wait phase)")
                        response.pause(length=3)

                tax_method = state.get('ivr_tax_id_method', 'dtmf')
                clean_tax = re.sub(r'\D', '', state['tax_id'])
                digits_to_send = state.get('ivr_tax_id_digits_to_send')
                selected_tax = clean_tax[-digits_to_send:] if isinstance(digits_to_send, int) and 1 <= digits_to_send <= len(clean_tax) else clean_tax
                if tax_method == 'dtmf':
                    tax_digits = selected_tax + state.get('ivr_tax_id_suffix', '')
                    response.play(digits=tax_digits)
                    print(f"🔢 Sent Tax ID via DTMF during wait: {tax_digits}")
                else:
                    speak_with_tts(response, '. '.join(list(selected_tax)) + '.')
                    print(f"🗣️ Spoke Tax ID during wait: {selected_tax}")
            response.pause(length=1)
            gather = Gather(
                input='speech dtmf',
                action=f'{callback_base}/webhook/wait-for-human',
                method='POST',
                speech_timeout='2',
                timeout=15,
                language='en-US'
            )
            response.append(gather)
            response.redirect(f'{callback_base}/webhook/wait-for-human')
            return Response(str(response), mimetype='text/xml')
        # ── End NPI / Tax ID auto-response ───────────────────────────────────────

        # Track total wait time (needed by both AI classification and keyword logic)
        if 'wait_start_time' not in state:
            from datetime import datetime
            state['wait_start_time'] = utcnow_iso()

        # Calculate how long we've been waiting
        from datetime import datetime
        wait_start = datetime.fromisoformat(state['wait_start_time'])
        wait_duration_minutes = (utcnow() - wait_start).total_seconds() / 60

        # ── AI-assisted classification (agent mode only) ──────────────────────────
        # Uses gpt-4o-mini to classify each speech chunk as human/IVR/hold/silence.
        # Only runs for agent-transfer calls — adds ~0.5-1s latency per webhook but
        # catches edge cases that keyword lists miss (e.g. "Thank you for calling…
        # My name is Mary" which previously scored as queue/automated).
        if state.get('call_mode') == 'agent' and speech_result and not is_very_short:
            ai_context = (
                f"Insurance: {state.get('insurance_name', 'unknown')}. "
                f"We just navigated the IVR and are waiting for a live provider services representative."
            )
            ai_result = _ai_classify_speech(speech_result, ai_context)
            if ai_result:
                ai_type = ai_result.get('type', '')
                ai_conf = float(ai_result.get('confidence', 0.0))
                print(f"🤖 AI classification: {ai_type} (confidence={ai_conf:.2f}) — {ai_result.get('reasoning', '')}")

                if ai_type == 'human_speech' and ai_conf >= 0.65:
                    print(f"✅ AI confirmed human after {wait_duration_minutes:.1f} min — bridging to agent")
                    state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                    response.redirect(f'{callback_base}/webhook/bridge-agent')
                    return Response(str(response), mimetype='text/xml')

                elif ai_type in ('ivr_menu', 'hold_music') and ai_conf >= 0.65:
                    print(f"📞 AI confirmed automated system ({ai_type}) — continuing to wait")
                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/wait-for-human',
                        method='POST',
                        speech_timeout='2',
                        timeout=15,
                        language='en-US'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/voice/human')
                    return Response(str(response), mimetype='text/xml')
                # Low confidence or 'silence' → fall through to keyword logic below
        # ── End AI classification ─────────────────────────────────────────────────

        # PRIORITY 0 (strong human override): Phrases that only a live person says.
        # These beat queue/hold detection because automated systems never introduce
        # themselves by name or offer direct personal assistance mid-recording.
        strong_human_indicators = [
            'my name is',
            'i can help you',
            'how can i assist',
            'may i have your',
            'can i have your',
        ]
        is_strong_human = any(ind in speech_lower for ind in strong_human_indicators)

        # PRIORITY 1: Detect automated queue/hold systems (check FIRST before human detection)
        # These are definitive indicators that it's NOT a human
        queue_system_indicators = [
            'all representatives are busy',
            'all agents are busy',
            'all of our representatives',
            'currently assisting other callers',
            'your call will be answered in the order',
            'please stay on the line',
            'continue to hold',
            'press 1 to continue holding',
            'press 2 if you prefer a call back',
            'press 3 to leave a message',
            'your position in queue',
            'estimated wait time',
            'currently experiencing high call volume',
            'you have reached out to',  # "you have reached out to X department"
            'thank you for calling',
            'this call is being recorded',
            'for quality assurance purposes'
        ]
        is_queue_system = any(indicator in speech_lower for indicator in queue_system_indicators)

        # PRIORITY 2: Detect transfer/hold messages (keep waiting)
        transfer_indicators = [
            'please wait',
            'please hold',
            'connecting you',
            'transferring you',
            'one moment',
            'your call is important'
        ]
        is_transfer_message = any(indicator in speech_lower for indicator in transfer_indicators)

        # PRIORITY 3: Detect human greeting/response (only if NOT queue/automated)
        # These ONLY indicate human if no queue indicators present
        human_indicators = [
            'how can i help',
            'how may i help',
            'may i help you',
            'speaking',
            'who am i speaking with',
            'may i ask who',
            'this is',  # Only when NOT part of automated message
            'good morning',
            'good afternoon',
            'good evening'
        ]
        # Simple greetings only count as human if message is SHORT (not part of long automated message)
        simple_greetings = ['hello', 'hi', 'credentialing department']

        is_human = (
            any(indicator in speech_lower for indicator in human_indicators) or
            (any(greeting in speech_lower for greeting in simple_greetings) and len(speech_result) < 50)
        )

        # DECISION LOGIC - Priority order matters!

        # PRIORITY 0: Strong human override — beats queue/hold detection.
        # "My name is Mary" + "thank you for calling" → still a human.
        if is_strong_human and not is_very_short:
            print(f"✅ Strong human signal detected after {wait_duration_minutes:.1f} minutes!")
            print(f"   Heard: {speech_result[:100]}...")
            state['call_state'] = CallState.SPEAKING_WITH_HUMAN
            if state.get('call_mode') == 'agent':
                print(f"🔀 Real Human Agent mode - redirecting to bridge-agent")
                response.redirect(f'{callback_base}/webhook/bridge-agent')
            else:
                response.redirect(f'{callback_base}/webhook/voice/human')

        # PRIORITY 1: Detected queue system or transfer - definitely NOT human yet
        elif is_queue_system or is_transfer_message:
            if wait_duration_minutes < 10:
                # Still within acceptable wait time - keep waiting
                print(f"📞 Queue/automated system detected (attempt {state['wait_attempts']}, {wait_duration_minutes:.1f} min)")
                print(f"   Heard: {speech_result[:100]}...")
                print(f"   Continuing to wait for human...")
                gather = Gather(
                    input='speech',
                    action=f'{callback_base}/webhook/wait-for-human',
                    method='POST',
                    speech_timeout='2',
                    timeout=15,
                    language='en-US'
                )
                response.append(gather)
                response.redirect(f'{callback_base}/webhook/voice/human')
            else:
                # Waited too long in queue - hang up
                print(f"⏰ Queue wait time exceeded ({wait_duration_minutes:.1f} minutes) - hanging up")
                state['call_state'] = CallState.FAILED
                state['failure_reason'] = f'Long queue - waited {wait_duration_minutes:.1f} minutes'
                speak_with_tts(response, "The wait time is longer than expected. I'll try calling back later. Goodbye.")
                response.hangup()

        # PRIORITY 2: Detected human (and NO queue/transfer indicators)
        elif is_human and not is_very_short:
            # Confident this is a real human - proceed with disclosure (or bridge agent)
            print(f"✅ Human detected after {wait_duration_minutes:.1f} minutes! Starting conversation")
            print(f"   Heard: {speech_result[:100]}...")
            state['call_state'] = CallState.SPEAKING_WITH_HUMAN
            if state.get('call_mode') == 'agent':
                print(f"🔀 Real Human Agent mode - redirecting to bridge-agent")
                response.redirect(f'{callback_base}/webhook/bridge-agent')
            else:
                response.redirect(f'{callback_base}/webhook/voice/human')

        # PRIORITY 3: Timeout checks (no clear human or queue detected)
        elif wait_duration_minutes >= 10:
            # Waited too long without detecting human - give up
            print(f"⏰ Maximum wait time exceeded ({wait_duration_minutes:.1f} minutes) without human detection")
            state['call_state'] = CallState.FAILED
            state['failure_reason'] = f'No human detected after {wait_duration_minutes:.1f} minutes'
            speak_with_tts(response, "I've been unable to reach a representative. I'll try calling back later. Goodbye.")
            response.hangup()

        elif state['wait_attempts'] >= 60:
            # Fallback: Too many attempts without clear indicators
            print(f"⏰ Maximum attempts exceeded ({state['wait_attempts']} attempts, {wait_duration_minutes:.1f} min)")
            state['call_state'] = CallState.FAILED
            state['failure_reason'] = f"Timeout after {state['wait_attempts']} attempts"
            speak_with_tts(response, "I've been unable to reach a representative. I'll try calling back later. Goodbye.")
            response.hangup()

        # PRIORITY 4: Default - unclear what we heard, keep waiting
        else:
            print(f"🔄 Unclear signal (attempt {state['wait_attempts']}, {wait_duration_minutes:.1f} min)")
            print(f"   Heard: {speech_result[:100] if speech_result else '(silence)'}...")
            print(f"   Continuing to listen...")
            gather = Gather(
                input='speech',
                action=f'{callback_base}/webhook/wait-for-human',
                method='POST',
                speech_timeout='2',
                timeout=15,
                language='en-US'
            )
            response.append(gather)
            # If Gather times out, try speaking anyway
            response.redirect(f'{callback_base}/webhook/voice/human')

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in wait_for_human_webhook: {e}")
        import traceback
        traceback.print_exc()
        # On error, proceed to human mode
        response = VoiceResponse()
        response.redirect(f'{callback_base}/webhook/voice/human')
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/speech', methods=['POST'])
def speech_webhook():
    """
    Handle speech input from Twilio Gather - Uses GPT-4 for intelligent responses
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()

        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult', '')

        # Get the base URL
        callback_base = os.getenv('CALLBACK_URL', '').replace('/webhook/voice', '')
        if not callback_base:
            callback_base = f"https://{request.host}"

        print(f"\n{'='*60}")
        print(f"🎤 [SPEECH] CallSID: {call_sid}")
        print(f"👤 REP SAID: \"{speech_result}\"")
        print(f"{'='*60}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                # Add to transcript
                state['transcript'].append({
                    'speaker': 'insurance',
                    'text': speech_result,
                    'timestamp': utcnow_iso()
                })
                break

        if call_info:
            _set_log_context(call_info['call_id'])

        # Use GPT-4 to generate intelligent response
        if call_info:
            state = call_info['state']

            # Check if we're hearing IVR menu options instead of human speech
            # This can happen if we switched to human mode too early
            speech_lower = speech_result.lower()
            is_ivr_menu = any(indicator in speech_lower for indicator in IVR_MENU_INDICATORS)

            if is_ivr_menu:
                print(f"⚠️ Detected IVR menu in human mode - switching back to IVR navigation")
                # Redirect back to IVR navigation
                state['call_state'] = CallState.IVR_NAVIGATION
                response.redirect(f'{callback_base}/webhook/ivr-navigate')
                return Response(str(response), mimetype='text/xml')

            # Track disclosure acknowledgment and questions asked
            disclosure_acknowledged = state.get('disclosure_acknowledged', False)
            questions = state.get('questions', [])
            questions_asked_count = state.get('questions_asked_count', 0)

            # Determine the stage based on disclosure and questions progress
            if not disclosure_acknowledged:
                stage = "awaiting_disclosure_confirmation"
            elif questions_asked_count >= len(questions):
                stage = "wrapping_up"
            else:
                # We're in initial contact until first question is asked
                stage = f"initial_contact_ask_question_{questions_asked_count + 1}"

            ai_response = smart_agent.generate_response(
                speech=speech_result,
                state=state,
                stage=stage,
                current_question_index=questions_asked_count
            )

            print(f"🤖 AGENT SAID: \"{ai_response.get('response', '')}\"")
            print(f"   Action: {ai_response.get('action')} | Stage: {stage} | Questions: {questions_asked_count}/{len(questions)}")
            extracted = ai_response.get('extracted_info', {})
            if any(v for v in extracted.values() if v):
                print(f"   📋 Extracted: {extracted}")

            # Detect if human acknowledged the disclosure
            if not state.get('disclosure_acknowledged', False):
                speech_lower = speech_result.lower()
                confirmation_keywords = ['yes', 'correct', 'speaking', 'speak', 'this is', 'how can', 'what', 'help', 'thank', 'my name']
                if any(keyword in speech_lower for keyword in confirmation_keywords):
                    state['disclosure_acknowledged'] = True
                    print(f"✅ Disclosure acknowledged by human")

            # Server-side refusal detection — override AI action to end_call
            # Catches cases where the human explicitly refuses to work with AI
            ai_refusal_phrases = [
                'do not work with ai', 'don\'t work with ai', 'cannot work with ai',
                'can\'t work with ai', 'not work with ai', 'we don\'t work with',
                'we do not work with', 'not able to work with ai', 'refuse to work',
                'cannot assist ai', 'can\'t assist ai', 'policy not to',
                'we don\'t speak with ai', 'we do not speak with ai',
            ]
            if any(phrase in speech_lower for phrase in ai_refusal_phrases):
                state['refusal_count'] = state.get('refusal_count', 0) + 1
                print(f"⛔ AI refusal detected (count: {state['refusal_count']})")
            # Also increment if AI itself keeps repeating (question_asked=True) with no progress
            if state.get('questions_asked_count', 0) > len(questions) + 3:
                state['refusal_count'] = state.get('refusal_count', 0) + 1
            if state.get('refusal_count', 0) >= 1:
                print(f"⛔ Forcing end_call due to refusal")
                ai_response['action'] = 'end_call'
                ai_response['response'] = "I understand. I apologize for the inconvenience. I'll have our team follow up another way. Thank you for your time. Goodbye."

            # Update questions asked count if a question was asked
            if ai_response.get('question_asked'):
                state['questions_asked_count'] = questions_asked_count + 1
                print(f"✅ Question asked! New count: {state['questions_asked_count']}")

            # Add AI response to transcript
            state['transcript'].append({
                'speaker': 'agent',
                'text': ai_response.get('response', ''),
                'timestamp': utcnow_iso()
            })

            # Store any extracted info
            if ai_response.get('extracted_info'):
                extracted = ai_response['extracted_info']
                if 'status' in extracted:
                    state['credentialing_status'] = extracted['status']
                if 'reference_number' in extracted:
                    state['reference_number'] = extracted['reference_number']
                if 'turnaround_days' in extracted:
                    state['turnaround_days'] = extracted['turnaround_days']
                if 'missing_documents' in extracted:
                    state['missing_documents'] = extracted['missing_documents']
                state['notes'] += f" {json.dumps(extracted)}"

            # Determine next action based on AI decision
            action = ai_response.get('action', 'continue')
            finalize = action == 'end_call'
            request_id = state.get('db_request_id') or state.get('call_id')

            # Persist conversation + progress in a single DB connection
            try:
                from credentialing_agent import DatabaseManager
                db = DatabaseManager()
                db.save_conversation(call_info['call_id'], 'representative', speech_result, state.get('db_request_id'))
                db.save_conversation(call_info['call_id'], 'agent', ai_response.get('response', ''), state.get('db_request_id'))
                if request_id:
                    if finalize:
                        db.save_final_results(request_id, state)
                    elif ai_response.get('extracted_info'):
                        db.update_call_progress(request_id, state)
                db.close()
            except Exception as db_err:
                print(f"WARN: Could not persist conversation/progress: {db_err}")

            if finalize:
                _enqueue_call_knowledge_ingestion(
                    call_id=call_info['call_id'],
                    request_id=request_id,
                    state=state,
                )

            if action == 'end_call':
                speak_with_tts(response, ai_response.get('response', 'Thank you for your time. Goodbye.'))
                # Give the other party a short window (8s) in case they need to interject before we disconnect
                response.pause(length=8)
                response.hangup()
            elif action == 'request_transfer':
                questions_asked_count = state.get('questions_asked_count', 0)
                total_questions = len(state.get('questions', []))

                # Block transfer until all questions are asked
                if questions_asked_count < total_questions:
                    # Politely refuse and continue with questions
                    remaining_questions = total_questions - questions_asked_count
                    next_question = state.get('questions', [])[questions_asked_count]
                    response_text = f"I understand you'd like to transfer me, but I need to ask {remaining_questions} quick question{'s' if remaining_questions > 1 else ''} first. {next_question}"

                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/speech/followup',
                        method='POST',
                        speech_timeout='auto',
                        language='en-US'
                    )
                    speak_with_tts(response, response_text, gather=gather)
                    response.append(gather)
                    speak_with_tts(response, "Are you still there?")
                else:
                    # All questions asked - allow transfer
                    speak_with_tts(response, ai_response.get('response', 'Yes, please transfer me to the credentialing department.'))
                    response.pause(length=30)  # Wait for transfer
                    # After pause, gather again
                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/speech',
                        method='POST',
                        speech_timeout='auto',
                        language='en-US'
                    )
                    speak_with_tts(response, "Hello? Is anyone there?", gather=gather)
                    response.append(gather)
            else:
                # Continue conversation
                gather = Gather(
                    input='speech',
                    action=f'{callback_base}/webhook/speech/followup',
                    method='POST',
                    speech_timeout='auto',
                    timeout=5,  # Wait 5 seconds for response (dead air detection)
                    language='en-US'
                )
                speak_with_tts(response, ai_response.get('response', 'Could you please repeat that?'), gather=gather)
                response.append(gather)

                # Dead air detected after 5 seconds - redirect to dead air handler
                response.redirect(f'{callback_base}/webhook/speech/dead-air')
        else:
            # No call state found, use fallback
            gather = Gather(
                input='speech',
                action=f'{callback_base}/webhook/speech',
                method='POST',
                speech_timeout='auto',
                language='en-US'
            )
            speak_with_tts(response, "I apologize, could you please repeat that?", gather=gather)
            response.append(gather)

            # Fallback redirect
            response.redirect(f'{callback_base}/webhook/speech')

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in speech_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        speak_with_tts(response, "Sorry, there was a technical error. Goodbye.")
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/speech/followup', methods=['POST'])
def speech_followup_webhook():
    """
    Handle follow-up speech input - Uses GPT-4 for intelligent responses
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()

        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult', '')

        # Get the base URL
        callback_base = os.getenv('CALLBACK_URL', '').replace('/webhook/voice', '')
        if not callback_base:
            callback_base = f"https://{request.host}"

        print(f"\n{'='*60}")
        print(f"🎤 [FOLLOW-UP] CallSID: {call_sid}")
        print(f"👤 REP SAID: \"{speech_result}\"")
        print(f"{'='*60}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                # Add to transcript
                state['transcript'].append({
                    'speaker': 'insurance',
                    'text': speech_result,
                    'timestamp': utcnow_iso()
                })
                break

        if call_info:
            _set_log_context(call_info['call_id'])

        if call_info:
            state = call_info['state']

            # Check if we're hearing IVR menu options instead of human speech
            # This can happen if we switched to human mode too early
            speech_lower = speech_result.lower()
            is_ivr_menu = any(indicator in speech_lower for indicator in IVR_MENU_INDICATORS)

            if is_ivr_menu:
                print(f"⚠️ Detected IVR menu in human mode - switching back to IVR navigation")
                # Redirect back to IVR navigation
                state['call_state'] = CallState.IVR_NAVIGATION
                response.redirect(f'{callback_base}/webhook/ivr-navigate')
                return Response(str(response), mimetype='text/xml')

            # Track disclosure acknowledgment and questions asked
            disclosure_acknowledged = state.get('disclosure_acknowledged', False)
            questions = state.get('questions', [])
            questions_asked_count = state.get('questions_asked_count', 0)

            # Determine stage - check disclosure first, then questions
            if not disclosure_acknowledged:
                stage = "awaiting_disclosure_confirmation"
            elif questions_asked_count >= len(questions):
                stage = "wrapping_up"
            else:
                stage = f"asking_question_{questions_asked_count + 1}_of_{len(questions)}"

            # Use GPT-4 to generate intelligent response
            ai_response = smart_agent.generate_response(
                speech=speech_result,
                state=state,
                stage=stage,
                current_question_index=questions_asked_count
            )

            print(f"🤖 AGENT SAID: \"{ai_response.get('response', '')}\"")
            print(f"   Action: {ai_response.get('action')} | Stage: {stage} | Questions: {questions_asked_count}/{len(questions)}")
            extracted = ai_response.get('extracted_info', {})
            if any(v for v in extracted.values() if v):
                print(f"   📋 Extracted: {extracted}")

            # Detect if human acknowledged the disclosure
            if not state.get('disclosure_acknowledged', False):
                speech_lower = speech_result.lower()
                confirmation_keywords = ['yes', 'correct', 'speaking', 'speak', 'this is', 'how can', 'what', 'help', 'thank', 'my name']
                if any(keyword in speech_lower for keyword in confirmation_keywords):
                    state['disclosure_acknowledged'] = True
                    print(f"✅ Disclosure acknowledged by human")

            # Server-side refusal detection — override AI action to end_call
            ai_refusal_phrases = [
                'do not work with ai', 'don\'t work with ai', 'cannot work with ai',
                'can\'t work with ai', 'not work with ai', 'we don\'t work with',
                'we do not work with', 'not able to work with ai', 'refuse to work',
                'cannot assist ai', 'can\'t assist ai', 'policy not to',
                'we don\'t speak with ai', 'we do not speak with ai',
            ]
            if any(phrase in speech_lower for phrase in ai_refusal_phrases):
                state['refusal_count'] = state.get('refusal_count', 0) + 1
                print(f"⛔ AI refusal detected (count: {state['refusal_count']})")
            if state.get('questions_asked_count', 0) > len(questions) + 3:
                state['refusal_count'] = state.get('refusal_count', 0) + 1
            if state.get('refusal_count', 0) >= 1:
                print(f"⛔ Forcing end_call due to refusal")
                ai_response['action'] = 'end_call'
                ai_response['response'] = "I understand. I apologize for the inconvenience. I'll have our team follow up another way. Thank you for your time. Goodbye."

            # Update questions asked count if a question was asked
            if ai_response.get('question_asked'):
                state['questions_asked_count'] = questions_asked_count + 1
                print(f"✅ Question asked! New count: {state['questions_asked_count']}")

            # Add AI response to transcript
            state['transcript'].append({
                'speaker': 'agent',
                'text': ai_response.get('response', ''),
                'timestamp': utcnow_iso()
            })

            # Store any extracted info
            if ai_response.get('extracted_info'):
                for key, value in ai_response['extracted_info'].items():
                    state['notes'] += f" {key}: {value}."

                # Check for specific fields
                extracted = ai_response['extracted_info']
                if 'status' in extracted:
                    state['credentialing_status'] = extracted['status']
                if 'reference_number' in extracted:
                    state['reference_number'] = extracted['reference_number']
                if 'turnaround_days' in extracted:
                    state['turnaround_days'] = extracted['turnaround_days']
                if 'missing_documents' in extracted:
                    state['missing_documents'] = extracted['missing_documents']

            # Determine next action - only end when AI explicitly decides to
            action = ai_response.get('action', 'continue')
            finalize = action == 'end_call'
            request_id = state.get('db_request_id') or state.get('call_id')

            if finalize:
                state['call_state'] = CallState.COMPLETING

            # Persist conversation + progress in a single DB connection
            try:
                from credentialing_agent import DatabaseManager
                db = DatabaseManager()
                db.save_conversation(call_info['call_id'], 'representative', speech_result, state.get('db_request_id'))
                db.save_conversation(call_info['call_id'], 'agent', ai_response.get('response', ''), state.get('db_request_id'))
                if request_id:
                    if finalize:
                        db.save_final_results(request_id, state)
                    elif ai_response.get('extracted_info'):
                        db.update_call_progress(request_id, state)
                db.close()
            except Exception as db_err:
                print(f"WARN: Could not persist conversation/progress: {db_err}")

            if finalize:
                _enqueue_call_knowledge_ingestion(
                    call_id=call_info['call_id'],
                    request_id=request_id,
                    state=state,
                )
                _flush_call_logs(call_info['call_id'], state.get('call_sid'))

            if action == 'end_call':
                speak_with_tts(response, ai_response.get('response', 'Thank you very much for your help. Have a great day. Goodbye.'))
                # Wait 8 seconds after the AI stops talking before ending the call
                response.pause(length=8)
                response.hangup()
            elif action == 'request_transfer':
                questions_asked_count = state.get('questions_asked_count', 0)
                total_questions = len(state.get('questions', []))

                # Block transfer until all questions are asked
                if questions_asked_count < total_questions:
                    # Politely refuse and continue with questions
                    remaining_questions = total_questions - questions_asked_count
                    next_question = state.get('questions', [])[questions_asked_count]
                    response_text = f"I understand you'd like to transfer me, but I need to ask {remaining_questions} quick question{'s' if remaining_questions > 1 else ''} first. {next_question}"

                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/speech/followup',
                        method='POST',
                        speech_timeout='auto',
                        language='en-US'
                    )
                    speak_with_tts(response, response_text, gather=gather)
                    response.append(gather)
                    speak_with_tts(response, "Are you still there?")
                else:
                    # All questions asked - allow transfer
                    speak_with_tts(response, ai_response.get('response', 'Yes, please transfer me to the credentialing department.'))
                    response.pause(length=30)  # Wait for transfer
                    gather = Gather(
                        input='speech',
                        action=f'{callback_base}/webhook/speech/followup',
                        method='POST',
                        speech_timeout='auto',
                        language='en-US'
                    )
                    speak_with_tts(response, "Hello? Is anyone there?", gather=gather)
                    response.append(gather)
            else:
                # Continue conversation
                gather = Gather(
                    input='speech',
                    action=f'{callback_base}/webhook/speech/followup',
                    method='POST',
                    speech_timeout='auto',
                    timeout=5,  # Wait 5 seconds for response (dead air detection)
                    language='en-US'
                )
                speak_with_tts(response, ai_response.get('response', 'Thank you. Do you have any other information?'), gather=gather)
                response.append(gather)

                # Dead air detected after 5 seconds - redirect to dead air handler
                response.redirect(f'{callback_base}/webhook/speech/dead-air')
        else:
            # No call state - try to recover
            gather = Gather(
                input='speech',
                action=f'{callback_base}/webhook/speech/followup',
                method='POST',
                speech_timeout='auto',
                timeout=5,  # 5 second dead air detection
                language='en-US'
            )
            speak_with_tts(response, "I apologize, could you please repeat that?", gather=gather)
            response.append(gather)
            response.redirect(f'{callback_base}/webhook/speech/dead-air')

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in speech_followup_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        speak_with_tts(response, "Sorry, there was a technical error. Goodbye.")
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/speech/dead-air', methods=['POST'])
def speech_dead_air_webhook():
    """
    Handle dead air detection - triggered after 5 seconds of silence.
    Asks if there's anything else, then waits 10 more seconds before ending.
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()
        callback_base = os.getenv('CALLBACK_URL', '').replace('/webhook/voice', '')
        if not callback_base:
            callback_base = f"https://{request.host}"
        call_sid = request.values.get('CallSid')

        # Set log context if we can resolve call_id from call_sid
        for _cid, _st in list(call_states.items()):
            if _st.get('call_sid') == call_sid:
                _set_log_context(_cid)
                break

        print(f"🔇 Dead air detected for call {call_sid} - asking if anything else needed")

        # Ask if there's anything else (this is triggered after 5 sec dead air)
        gather = Gather(
            input='speech',
            action=f'{callback_base}/webhook/speech/followup',
            method='POST',
            speech_timeout='auto',
            timeout=10,  # Wait 10 seconds for response before ending call
            language='en-US'
        )
        speak_with_tts(response, "Is there anything else I can help you with?", gather=gather)
        response.append(gather)

        # If still no response after 10 seconds, end the call
        speak_with_tts(response, "Thank you for your time. Have a great day. Goodbye.")
        response.hangup()

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in speech_dead_air_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        speak_with_tts(response, "Thank you for your time. Goodbye.")
        response.hangup()
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/status', methods=['POST'])
def status_webhook():
    """
    Twilio webhook for call status updates
    """
    call_sid = request.values.get('CallSid')
    call_status = request.values.get('CallStatus')
    
    print(f"Call {call_sid} status: {call_status}")
    
    # Update call state
    if call_sid in call_states:
        state = call_states[call_sid]

        if call_status == 'completed':
            state['call_state'] = CallState.COMPLETING
        elif call_status == 'failed':
            state['call_state'] = CallState.FAILED
            state['error_message'] = request.values.get('ErrorMessage', 'Unknown error')

    # Release Twilio number on terminal statuses (safety net)
    terminal_statuses = {'completed', 'failed', 'busy', 'no-answer', 'canceled'}
    if call_status in terminal_statuses:
        _release_twilio_number(call_sid=call_sid)

    # Flush per-call log buffer on terminal statuses
    if call_status in terminal_statuses:
        _call_id_for_flush = None
        with call_state_lock:
            state_by_sid = call_states_by_sid.get(call_sid)
            if state_by_sid:
                _call_id_for_flush = state_by_sid.get('call_id')
        if not _call_id_for_flush:
            for _cid, _st in list(call_states.items()):
                if _st.get('call_sid') == call_sid:
                    _call_id_for_flush = _cid
                    break
        if _call_id_for_flush:
            _flush_call_logs(_call_id_for_flush, call_sid)

    return jsonify({'success': True}), 200


@app.route('/webhook/voice/status', methods=['POST'])
def voice_status_webhook():
    """
    Alias for status webhook - Twilio may call this URL
    """
    return status_webhook()


@app.route('/webhook/transcription', methods=['POST'])
def transcription_webhook():
    """
    Webhook to receive real-time transcription from Deepgram
    This would be called by your transcription service
    """
    try:
        data = request.json
        call_sid = data.get('call_sid')
        transcript = data.get('transcript')
        is_final = data.get('is_final', False)
        confidence = data.get('confidence', 0.0)
        
        if not is_final:
            # Ignore interim results
            return jsonify({'success': True}), 200
        
        # Find the associated call state
        if call_sid in call_states:
            state = call_states[call_sid]
            
            # Add to transcript
            state['transcript'].append({
                'text': transcript,
                'timestamp': utcnow_iso(),
                'confidence': confidence
            })
            print(f"📡 DEEPGRAM [{confidence:.2f}]: \"{transcript}\"")

            # Trigger state graph to process new transcript
            # This would involve resuming the agent's state machine
            # Implementation depends on your LangGraph checkpoint strategy
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        print(f"Transcription webhook error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/webhook/recording-status', methods=['POST'])
def recording_status_webhook():
    """
    Twilio webhook for recording status updates.
    Called when a call recording is completed or fails.
    """
    try:
        from credentialing_agent import DatabaseManager

        recording_sid = request.values.get('RecordingSid')
        recording_url = request.values.get('RecordingUrl')
        recording_status = request.values.get('RecordingStatus')
        recording_duration = request.values.get('RecordingDuration')
        call_sid = request.values.get('CallSid')
        callback_call_id = request.args.get('call_id')
        callback_request_id = request.args.get('request_id')
        normalized_recording_status = 'failed' if recording_status == 'absent' else recording_status

        logger.info(
            f"Recording webhook: {recording_sid} status={normalized_recording_status} for call {call_sid} "
            f"(call_id={callback_call_id}, request_id={callback_request_id})"
        )

        # Find call_id and request_id from in-memory state first.
        # Conference recording callbacks may omit CallSid, so avoid resolving
        # state by None and fall back to callback identifiers.
        state = None
        if call_sid:
            with call_state_lock:
                state = call_states_by_sid.get(call_sid)
            if state is None:
                state = get_state_by_sid(call_sid)
        elif callback_call_id:
            with call_state_lock:
                state = call_states.get(callback_call_id)

        call_id = callback_call_id or (state.get('call_id') if state else call_sid or recording_sid)
        request_id = callback_request_id or (state.get('db_request_id') if state else None)
        effective_call_sid = call_sid or (state.get('call_sid') if state else None) or f"recording:{recording_sid}"

        # Determine recording type from in-memory call state
        recording_type = 'ai'
        if state and state.get('call_mode') == 'agent':
            recording_type = 'agent'

        # Save to database
        db = DatabaseManager()
        try:
            # If in-memory state is gone (e.g. server restarted between call end and webhook),
            # attempt to recover request_id by matching call_sid against conversation_history
            # or call_metrics, which persist across restarts.
            if request_id is None:
                try:
                    with db.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT request_id FROM call_metrics
                            WHERE call_id = %s
                            LIMIT 1
                            """,
                            (call_sid,),
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            request_id = str(row[0])
                            logger.info(
                                f"Recovered request_id {request_id} from call_metrics for call_sid {call_sid}"
                            )
                except Exception as lookup_err:
                    logger.warning(f"Could not recover request_id from DB: {lookup_err}")

            db.save_recording(
                call_id=call_id,
                call_sid=effective_call_sid,
                recording_sid=recording_sid,
                recording_url=recording_url,
                recording_duration=int(recording_duration or 0),
                recording_status=normalized_recording_status,
                request_id=request_id,
                recording_type=recording_type,
            )
            logger.info(
                f"Saved recording {recording_sid} for call {call_id} "
                f"(request_id={request_id}, status={normalized_recording_status})"
            )

            # Trigger Q&A extraction in background if recording is completed.
            # extract_qa_async queries conversation_history by call_id, so always
            # pass call_id here — it resolves request_id itself via a DB JOIN.
            if normalized_recording_status == 'completed':
                from qa_extractor import extract_qa_async
                threading.Thread(
                    target=extract_qa_async,
                    args=(call_id,),
                    kwargs={'request_id': request_id},
                    daemon=True,
                ).start()
                logger.info(f"Triggered Q&A extraction for call {call_id}")

                if recording_type == 'ai':
                    _enqueue_call_knowledge_ingestion(
                        call_id=call_id,
                        request_id=request_id,
                        state=state,
                    )
                    logger.info(f"Triggered knowledge ingestion for call {call_id}")

                # Transcribe agent-transferred recordings via Whisper
                if recording_type == 'agent' and recording_url:
                    from transcribe_recording import transcribe_agent_recording
                    threading.Thread(
                        target=transcribe_agent_recording,
                        args=(call_id, recording_url, request_id),
                        daemon=True,
                    ).start()
                    logger.info(f"Triggered Whisper transcription for agent call {call_id}")

        except Exception as e:
            logger.error(f"Failed to save recording: {e}")
        finally:
            db.close()

        return jsonify({'success': True}), 200

    except Exception as e:
        logger.error(f"Recording webhook error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/call-recording/<call_id>', methods=['GET'])
@agent_or_above
def get_call_recording(call_id: str):
    """
    Get recording metadata for a call.
    """
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        with db.conn.cursor() as cur:
            cur.execute(
                "SELECT initiated_by::text FROM credentialing_requests WHERE id = %s",
                (call_id,),
            )
            owner_row = cur.fetchone()
        denial = _deny_if_call_not_owned(owner_row[0] if owner_row else None)
        if denial:
            db.close()
            return denial
        recording = db.get_recording(call_id=call_id)
        db.close()

        if not recording:
            return jsonify({
                'success': False,
                'error': 'Recording not found'
            }), 404

        return jsonify({
            'success': True,
            'recording': {
                'recording_sid': recording['recording_sid'],
                'recording_url': recording['recording_url'],
                'duration': recording['recording_duration'],
                'format': recording['recording_format'],
                'status': recording['recording_status'],
                'recording_type': recording.get('recording_type', 'ai'),
                'created_at': serialize_timestamp(recording['created_at']),
            }
        }), 200

    except Exception as e:
        logger.error(f"Get recording error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/call-recording/<call_id>/stream', methods=['GET'])
@agent_or_above
def stream_call_recording(call_id: str):
    """
    Stream call recording audio.
    Proxies the audio from Twilio to avoid CORS issues.
    """
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        with db.conn.cursor() as cur:
            cur.execute(
                "SELECT initiated_by::text FROM credentialing_requests WHERE id = %s",
                (call_id,),
            )
            owner_row = cur.fetchone()
        denial = _deny_if_call_not_owned(owner_row[0] if owner_row else None)
        if denial:
            db.close()
            return denial
        recording = db.get_recording(call_id=call_id)
        db.close()

        if not recording:
            return jsonify({
                'success': False,
                'error': 'Recording not found'
            }), 404

        if recording['recording_status'] != 'completed':
            return jsonify({
                'success': False,
                'error': f"Recording not ready (status: {recording['recording_status']})"
            }), 400

        # Fetch recording from Twilio
        recording_url = recording['recording_url']

        # If it's a relative URL, construct the full URL
        if not recording_url.startswith('http'):
            recording_url = f"https://api.twilio.com{recording_url}.mp3"
        elif not recording_url.endswith('.mp3'):
            recording_url = f"{recording_url}.mp3"

        # Stream from Twilio with authentication
        import requests
        from requests.auth import HTTPBasicAuth

        twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
        twilio_token = os.getenv("TWILIO_AUTH_TOKEN")

        response = requests.get(
            recording_url,
            auth=HTTPBasicAuth(twilio_sid, twilio_token),
            stream=True
        )

        if response.status_code != 200:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch recording from Twilio'
            }), 500

        # Stream the audio
        def generate():
            for chunk in response.iter_content(chunk_size=8192):
                yield chunk

        return Response(
            generate(),
            mimetype='audio/mpeg',
            headers={
                'Content-Disposition': f'inline; filename="call_{call_id}.mp3"',
                'Content-Type': 'audio/mpeg'
            }
        )

    except Exception as e:
        logger.error(f"Stream recording error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/call-status/<call_id>', methods=['GET'])
@admin_required
def get_call_status(call_id: str):
    """
    Get current status of a call
    """
    if call_id not in call_states:
        return jsonify({
            'success': False,
            'error': 'Call not found'
        }), 404
    
    state = call_states[call_id]
    
    return jsonify({
        'success': True,
        'call_id': call_id,
        'call_sid': state.get('call_sid'),
        'call_state': state['call_state'].value,
        'insurance_name': state['insurance_name'],
        'provider_name': state['provider_name'],
        'status': state.get('credentialing_status'),
        'reference_number': state.get('reference_number'),
        'notes': state.get('notes', '')
    }), 200


@app.route('/api/call-transcript/<call_id>', methods=['GET'])
@admin_required
def get_call_transcript(call_id: str):
    """
    Get full conversation transcript
    """
    if call_id not in call_states:
        return jsonify({
            'success': False,
            'error': 'Call not found'
        }), 404
    
    state = call_states[call_id]
    
    return jsonify({
        'success': True,
        'call_id': call_id,
        'conversation': state.get('conversation_history', []),
        'transcript': state.get('transcript', [])
    }), 200


@app.route('/api/call-transcript/<call_id>/backfill-agent-recording', methods=['POST'])
@admin_required
def backfill_agent_recording_transcript(call_id: str):
    """
    Re-run post-call transcription for the longest completed Human Agent recording.
    Useful when the recording webhook completed but no agent_transcript rows exist.
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()
        try:
            with db.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT call_id, request_id::text, recording_url, recording_sid, recording_duration
                    FROM call_recordings
                    WHERE (request_id::text = %s OR call_id = %s)
                      AND recording_status = 'completed'
                      AND recording_type = 'agent'
                    ORDER BY recording_duration DESC NULLS LAST, created_at DESC
                    LIMIT 1
                    """,
                    (call_id, call_id),
                )
                recording = cur.fetchone()
        finally:
            db.close()

        if not recording:
            return jsonify({
                'success': False,
                'error': 'No completed Human Agent recording found for this call'
            }), 404

        runtime_call_id, request_id, recording_url, recording_sid, recording_duration = recording

        def _worker():
            try:
                from transcribe_recording import transcribe_agent_recording
                transcribe_agent_recording(
                    str(runtime_call_id),
                    recording_url,
                    request_id or call_id,
                )
            except Exception as exc:
                logger.error(f"Agent recording transcript backfill failed for {call_id}: {exc}", exc_info=True)

        threading.Thread(target=_worker, daemon=True).start()

        return jsonify({
            'success': True,
            'message': 'Agent recording transcript backfill started',
            'call_id': str(runtime_call_id),
            'request_id': request_id,
            'recording_sid': recording_sid,
            'recording_duration': recording_duration,
        }), 202

    except Exception as e:
        logger.error(f"Backfill agent recording transcript error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/call-qa/extract/<call_id>', methods=['POST'])
@admin_required
def extract_call_qa(call_id: str):
    """
    Manually trigger Q&A extraction for a call.
    Useful for retries or extracting from historical calls.
    """
    try:
        data = request.json or {}
        force = data.get('force', False)

        # Check if already extracted (unless force=true)
        if not force:
            from credentialing_agent import DatabaseManager
            db = DatabaseManager()
            existing_qa = db.get_qa_pairs(call_id)
            db.close()

            if existing_qa:
                return jsonify({
                    'success': True,
                    'message': 'Q&A already extracted',
                    'qa_pairs_extracted': len(existing_qa)
                }), 200

        # Extract in background
        from qa_extractor import extract_qa_async
        threading.Thread(target=extract_qa_async, args=(call_id,), daemon=True).start()

        return jsonify({
            'success': True,
            'message': 'Q&A extraction started'
        }), 202

    except Exception as e:
        logger.error(f"Q&A extraction trigger error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/call-qa/<call_id>', methods=['GET'])
@admin_required
def get_call_qa(call_id: str):
    """
    Get Q&A pairs for a call.
    Returns questions asked and answers received with confidence scores.
    """
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        qa_pairs = db.get_qa_pairs(call_id)
        db.close()

        return jsonify({
            'success': True,
            'qa_pairs': qa_pairs
        }), 200

    except Exception as e:
        logger.error(f"Get Q&A error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/call-detail/<call_id>', methods=['GET'])
@agent_or_above
def get_call_detail(call_id: str):
    """
    Get full call details from the database (persists across server restarts).
    Combines credentialing request data with conversation history and call events.
    """
    try:
        from credentialing_agent import DatabaseManager
        from psycopg2.extras import RealDictCursor

        db = DatabaseManager()
        lookup_id = call_id
        active_call_state = None

        # If frontend passed runtime call_id, map it to request_id when available.
        if call_id in call_states:
            mapped_request_id = call_states[call_id].get('db_request_id')
            if mapped_request_id:
                lookup_id = mapped_request_id

        with call_state_lock:
            active_call_state = (
                call_states.get(call_id)
                or call_states.get(lookup_id)
                or next(
                    (
                        state for state in call_states.values()
                        if state.get('db_request_id') == lookup_id
                        or state.get('call_id') == call_id
                        or state.get('call_id') == lookup_id
                    ),
                    None,
                )
            )

        # Fetch credentialing request
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, insurance_name, provider_name, npi, tax_id,
                       address, insurance_phone, questions, status, reference_number,
                       missing_documents, turnaround_days, notes,
                       created_at, updated_at, completed_at,
                       call_mode, agent_phone, initiated_by::text,
                       human_detection_correct
                FROM credentialing_requests
                WHERE id = %s
            """, (lookup_id,))
            row = cur.fetchone()

        if not row:
            db.close()
            return jsonify({'success': False, 'error': 'Call not found'}), 404

        denial = _deny_if_call_not_owned(row[18] if len(row) > 18 else None)
        if denial:
            db.close()
            return denial

        call_data = {
            'id': str(row[0]),
            'insurance_name': row[1],
            'provider_name': row[2],
            'npi': row[3],
            'tax_id': row[4],
            'address': row[5],
            'insurance_phone': row[6],
            'questions': row[7] or [],
            'status': row[8],
            'reference_number': row[9],
            'missing_documents': row[10] or [],
            'turnaround_days': row[11],
            'notes': row[12],
            'created_at': serialize_timestamp(row[13]),
            'updated_at': serialize_timestamp(row[14]),
            'completed_at': serialize_timestamp(row[15]),
            'call_mode': row[16] or 'ai',
            'agent_phone': row[17],
            'initiated_by': row[18],
            'human_detection_correct': row[19],
        }

        # Fetch conversation history
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT speaker, message, timestamp
                FROM conversation_history
                WHERE call_id = %s OR request_id = %s
                ORDER BY timestamp ASC
            """, (lookup_id, lookup_id))
            conversations = []
            for r in cur.fetchall():
                conversations.append({
                    'speaker': r[0],
                    'message': r[1],
                    'timestamp': serialize_timestamp(r[2]),
                })

        # Fetch call events
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT event_type, transcript, action_taken, confidence, timestamp, metadata
                FROM call_events
                WHERE call_id = %s
                ORDER BY timestamp ASC
            """, (lookup_id,))
            events = []
            for r in cur.fetchall():
                events.append({
                    'event_type': r[0],
                    'transcript': r[1],
                    'action_taken': r[2],
                    'confidence': r[3],
                    'timestamp': serialize_timestamp(r[4]),
                    'metadata': r[5] or {},
                })

        # Fetch recording data.  Try lookup_id (request UUID) first; if that
        # misses, try the original runtime call_id (different from lookup_id
        # when the frontend passed a runtime UUID that was mapped to request_id).
        recording = db.get_recording(call_id=lookup_id)
        if recording is None and lookup_id != call_id:
            recording = db.get_recording(call_id=call_id)

        # Fetch Q&A pairs. Try the request UUID first, then fall back to the
        # runtime call_id used by extraction/storage for in-flight calls.
        qa_pairs = db.get_qa_pairs(call_id=lookup_id)
        if not qa_pairs and lookup_id != call_id:
            qa_pairs = db.get_qa_pairs(call_id=call_id)

        # Enrich conversation with Q&A flags
        qa_map = {qa['id']: qa for qa in qa_pairs}
        for conv in conversations:
            # Check if this message is part of a Q&A
            for qa in qa_pairs:
                if qa.get('conversation_snippet'):
                    try:
                        snippet = qa['conversation_snippet'] if isinstance(qa['conversation_snippet'], list) else json.loads(qa['conversation_snippet'])
                        for msg in snippet:
                            if (msg.get('speaker') == conv['speaker'] and
                                msg.get('text') == conv['message']):
                                conv['related_qa_id'] = str(qa['id'])
                                conv['is_question'] = msg['speaker'] == 'agent'
                                conv['is_answer'] = msg['speaker'] != 'agent'
                                break
                    except:
                        pass

        # Fetch IVR patterns for this insurance
        ivr_patterns = []
        insurance_name = call_data.get('insurance_name')
        if insurance_name:
            try:
                with db.conn.cursor() as cur:
                    cur.execute("""
                        SELECT menu_level, detected_phrase, preferred_action, action_value
                        FROM ivr_knowledge
                        WHERE insurance_name ILIKE %s
                        ORDER BY menu_level ASC
                    """, (f"%{insurance_name}%",))
                    for r in cur.fetchall():
                        ivr_patterns.append({
                            'menu_level': r[0],
                            'detected_phrase': r[1],
                            'preferred_action': r[2],
                            'action_value': r[3],
                        })
            except Exception:
                pass  # IVR patterns are optional

        db.close()

        # Prepare recording data.  'available' is True only when the recording
        # has finished processing so the audio stream endpoint can serve it.
        recording_data = None
        if recording:
            recording_status = recording.get('recording_status') or 'processing'
            if recording_status == 'absent':
                recording_status = 'failed'
            recording_data = {
                'available': recording_status == 'completed',
                'url': f"/api/call-recording/{lookup_id}/stream",
                'duration': recording.get('recording_duration'),
                'status': recording_status,
                'recording_type': recording.get('recording_type', 'ai'),
                'created_at': serialize_timestamp(recording.get('created_at')),
            }
        elif active_call_state and row[15] is None:
            recording_data = {
                'available': False,
                'url': f"/api/call-recording/{lookup_id}/stream",
                'duration': None,
                'status': 'pending',
                'created_at': None,
            }

        # Prepare Q&A pairs data
        qa_pairs_data = []
        for qa in qa_pairs:
            qa_pairs_data.append({
                'id': str(qa['id']),
                'question_index': qa['question_index'],
                'question_text': qa['question_text'],
                'answer_text': qa.get('answer_text'),
                'confidence': qa.get('confidence', 0.0),
                'extracted_at': serialize_timestamp(qa.get('extracted_at')),
                'verified': qa.get('verified', False),
                'conversation_snippet': qa.get('conversation_snippet') if isinstance(qa.get('conversation_snippet'), list) else (json.loads(qa.get('conversation_snippet')) if qa.get('conversation_snippet') else []),
            })

        return jsonify({
            'success': True,
            'data': {
                **call_data,
                'conversation': conversations,
                'events': events,
                'ivr_patterns': ivr_patterns,
                'recording': recording_data,
                'qa_pairs': qa_pairs_data,
            }
        }), 200

    except Exception as e:
        print(f"Error fetching call detail: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# Self-Learning Human Detection: Feedback + CRUD
# =============================================================================

@app.route('/api/call/<call_id>/human-detection-feedback', methods=['POST'])
@admin_required
def submit_human_detection_feedback(call_id: str):
    """
    Submit feedback on whether human detection was correct for a call.
    If incorrect, triggers auto-review of the transcript to learn new phrases.
    """
    try:
        data = request.json or {}
        correct = data.get('correct')
        if correct is None:
            return jsonify({'success': False, 'error': 'correct (boolean) is required'}), 400

        from credentialing_agent import DatabaseManager
        db = DatabaseManager()

        # Save feedback to credentialing_requests
        with db.conn.cursor() as cur:
            cur.execute("""
                UPDATE credentialing_requests
                SET human_detection_correct = %s, human_detection_feedback_at = NOW()
                WHERE id = %s
            """, (correct, call_id))
        db.conn.commit()

        result = {'success': True, 'correct': correct, 'new_phrases': []}

        if correct:
            # Positive feedback: boost confidence of phrases that matched
            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT metadata FROM call_events
                    WHERE call_id = %s AND event_type = 'human_detected'
                    ORDER BY timestamp DESC LIMIT 1
                """, (call_id,))
                row = cur.fetchone()
                if row and row[0] and row[0].get('matched_phrases'):
                    for phrase in row[0]['matched_phrases']:
                        cur.execute("""
                            UPDATE human_detection_phrases
                            SET times_correct = times_correct + 1,
                                times_seen = times_seen + 1,
                                confidence = LEAST(1.0, (times_correct + 1)::float / GREATEST(times_seen + 1, 1)::float),
                                updated_at = NOW()
                            WHERE phrase = %s AND is_active = TRUE
                        """, (phrase,))
            db.conn.commit()
        else:
            # Negative feedback: auto-review the transcript
            # Fetch conversation and events for analysis
            with db.conn.cursor() as cur:
                cur.execute("""
                    SELECT speaker, message, timestamp
                    FROM conversation_history
                    WHERE call_id = %s OR request_id = %s
                    ORDER BY timestamp ASC
                """, (call_id, call_id))
                conversations = [{'speaker': r[0], 'message': r[1]} for r in cur.fetchall()]

                cur.execute("""
                    SELECT event_type, transcript, action_taken, metadata, timestamp
                    FROM call_events
                    WHERE call_id = %s
                    ORDER BY timestamp ASC
                """, (call_id,))
                events = [{'event_type': r[0], 'transcript': r[1], 'action_taken': r[2],
                           'metadata': r[3] or {}} for r in cur.fetchall()]

                # Get insurance name for this call
                cur.execute("SELECT insurance_name FROM credentialing_requests WHERE id = %s", (call_id,))
                ins_row = cur.fetchone()
                insurance_name = ins_row[0] if ins_row else None

            # Determine what went wrong
            human_events = [e for e in events if e['event_type'] == 'human_detected']
            ivr_events = [e for e in events if e['event_type'] == 'ivr_system_speech']

            # Build transcript snippet for analysis
            transcript_lines = []
            for e in events:
                if e.get('transcript'):
                    transcript_lines.append(f"[{e['event_type']}] {e['transcript']}")
            for c in conversations:
                transcript_lines.append(f"[{c['speaker']}] {c['message']}")

            transcript_snippet = "\n".join(transcript_lines[:30])  # Limit for LLM context

            # Get current phrases for context
            current_phrases = _get_human_detection_phrases(insurance_name)

            # Determine error type
            if human_events:
                error_description = "The system FALSELY detected a human (false positive) - the speech was actually IVR."
            else:
                error_description = "The system MISSED a human (false negative) - a real person was speaking but was classified as IVR."

            # Use GPT-4o-mini to analyze and extract new phrases
            try:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
                prompt = f"""You are analyzing a phone call transcript where human detection was INCORRECT.

Error: {error_description}

Transcript (events and conversation):
{transcript_snippet}

Current human indicator phrases: {current_phrases['human'][:20]}
Current IVR definitive phrases: {current_phrases['ivr_definitive'][:20]}
Current IVR passive phrases: {current_phrases['ivr_passive'][:15]}

Based on this transcript, extract NEW phrases that should be added to improve detection.
Rules:
- Phrases should be lowercase, 2-6 words, generic enough to apply to other calls
- Do NOT repeat existing phrases
- Only include phrases that clearly indicate human or IVR
- For false positive: suggest IVR phrases that were mistaken for human
- For false negative: suggest human phrases that were missed

Return ONLY valid JSON (no markdown):
{{
  "new_phrases": [
    {{"phrase": "example phrase", "phrase_type": "human", "confidence": 0.7}},
    {{"phrase": "press hash", "phrase_type": "ivr_definitive", "confidence": 0.9}}
  ],
  "demote_phrases": [
    {{"phrase": "some phrase", "reason": "too ambiguous"}}
  ]
}}"""

                response_text = llm.invoke(prompt).content.strip()
                # Parse JSON response
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    llm_result = json.loads(json_match.group())
                else:
                    llm_result = {'new_phrases': [], 'demote_phrases': []}

                # Save new phrases to DB
                new_phrases_saved = []
                for p in llm_result.get('new_phrases', []):
                    phrase = p.get('phrase', '').lower().strip()
                    phrase_type = p.get('phrase_type', 'human')
                    conf = min(max(p.get('confidence', 0.7), 0.3), 0.9)

                    if not phrase or phrase_type not in VALID_PHRASE_TYPES:
                        continue
                    if len(phrase) < 3 or len(phrase) > 80:
                        continue

                    try:
                        with db.conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO human_detection_phrases
                                    (phrase, phrase_type, insurance_name, source, confidence, source_call_id)
                                VALUES (%s, %s, %s, 'auto_review', %s, %s)
                                ON CONFLICT (phrase, phrase_type, COALESCE(insurance_name, ''))
                                DO UPDATE SET
                                    times_seen = human_detection_phrases.times_seen + 1,
                                    updated_at = NOW()
                            """, (phrase, phrase_type, insurance_name, conf, call_id))
                        db.conn.commit()
                        new_phrases_saved.append({'phrase': phrase, 'phrase_type': phrase_type, 'confidence': conf})
                    except Exception as insert_err:
                        print(f"WARN: Could not save phrase '{phrase}': {insert_err}")
                        db.conn.rollback()

                # Demote bad phrases
                for d in llm_result.get('demote_phrases', []):
                    phrase = d.get('phrase', '').lower().strip()
                    if phrase:
                        try:
                            with db.conn.cursor() as cur:
                                cur.execute("""
                                    UPDATE human_detection_phrases
                                    SET times_seen = times_seen + 1,
                                        confidence = GREATEST(0.0, confidence - 0.15),
                                        is_active = CASE WHEN confidence - 0.15 < 0.3 THEN FALSE ELSE is_active END,
                                        updated_at = NOW()
                                    WHERE phrase = %s
                                """, (phrase,))
                            db.conn.commit()
                        except Exception:
                            db.conn.rollback()

                result['new_phrases'] = new_phrases_saved
                result['analysis'] = error_description

            except Exception as llm_err:
                print(f"WARN: LLM auto-review failed: {llm_err}")
                import traceback
                traceback.print_exc()
                result['analysis_error'] = str(llm_err)

        db.close()
        return jsonify(result), 200

    except Exception as e:
        print(f"Error in human detection feedback: {e}")
        import traceback
        traceback.print_exc()
        try:
            db.close()
        except Exception:
            pass
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/human-detection-phrases', methods=['GET'])
@admin_required
def get_human_detection_phrases():
    """Get all human detection phrases, optionally filtered by insurance name."""
    try:
        insurance_name = request.args.get('insurance_name')
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()

        with db.conn.cursor() as cur:
            if insurance_name:
                cur.execute("""
                    SELECT id, phrase, phrase_type, insurance_name, source, confidence,
                           times_seen, times_correct, is_active, source_call_id, created_at, updated_at
                    FROM human_detection_phrases
                    WHERE insurance_name IS NULL OR insurance_name ILIKE %s
                    ORDER BY phrase_type, confidence DESC
                """, (f"%{insurance_name}%",))
            else:
                cur.execute("""
                    SELECT id, phrase, phrase_type, insurance_name, source, confidence,
                           times_seen, times_correct, is_active, source_call_id, created_at, updated_at
                    FROM human_detection_phrases
                    ORDER BY phrase_type, confidence DESC
                """)

            phrases = []
            for r in cur.fetchall():
                phrases.append({
                    'id': str(r[0]),
                    'phrase': r[1],
                    'phrase_type': r[2],
                    'insurance_name': r[3],
                    'source': r[4],
                    'confidence': r[5],
                    'times_seen': r[6],
                    'times_correct': r[7],
                    'is_active': r[8],
                    'source_call_id': r[9],
                    'created_at': r[10].isoformat() if r[10] else None,
                    'updated_at': r[11].isoformat() if r[11] else None,
                })

        db.close()
        return jsonify({'success': True, 'phrases': phrases}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/human-detection-phrases', methods=['POST'])
@admin_required
def add_human_detection_phrase():
    """Manually add a new human detection phrase."""
    data = request.json or {}
    phrase = data.get('phrase', '').lower().strip()
    phrase_type = data.get('phrase_type', 'human')
    insurance_name = data.get('insurance_name')

    if not phrase:
        return jsonify({'success': False, 'error': 'phrase is required'}), 400
    if phrase_type not in VALID_PHRASE_TYPES:
        return jsonify({'success': False, 'error': 'Invalid phrase_type'}), 400

    from credentialing_agent import DatabaseManager
    db = DatabaseManager()
    try:
        with db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO human_detection_phrases
                    (phrase, phrase_type, insurance_name, source, confidence)
                VALUES (%s, %s, %s, 'manual', 0.9)
                ON CONFLICT (phrase, phrase_type, COALESCE(insurance_name, ''))
                DO UPDATE SET
                    is_active = TRUE,
                    times_seen = human_detection_phrases.times_seen + 1,
                    updated_at = NOW()
                RETURNING id
            """, (phrase, phrase_type, insurance_name))
            row = cur.fetchone()
        db.conn.commit()

        return jsonify({'success': True, 'id': str(row[0]) if row else None}), 201
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/human-detection-phrases/<phrase_id>', methods=['DELETE'])
@admin_required
def delete_human_detection_phrase(phrase_id: str):
    """Deactivate a human detection phrase (soft delete)."""
    from credentialing_agent import DatabaseManager
    db = DatabaseManager()
    try:
        with db.conn.cursor() as cur:
            cur.execute("""
                UPDATE human_detection_phrases SET is_active = FALSE, updated_at = NOW()
                WHERE id = %s
            """, (phrase_id,))
        db.conn.commit()

        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/human-detection-phrases/<phrase_id>', methods=['PATCH'])
@admin_required
def update_human_detection_phrase(phrase_id: str):
    """Update a human detection phrase (toggle active, change type, etc.)."""
    data = request.json or {}

    updates = []
    params = []
    if 'is_active' in data:
        updates.append("is_active = %s")
        params.append(data['is_active'])
    if 'phrase_type' in data:
        updates.append("phrase_type = %s")
        params.append(data['phrase_type'])
    if 'confidence' in data:
        updates.append("confidence = %s")
        params.append(data['confidence'])

    if not updates:
        return jsonify({'success': False, 'error': 'No fields to update'}), 400

    from credentialing_agent import DatabaseManager
    db = DatabaseManager()
    try:
        updates.append("updated_at = NOW()")
        params.append(phrase_id)

        with db.conn.cursor() as cur:
            cur.execute(f"""
                UPDATE human_detection_phrases SET {', '.join(updates)}
                WHERE id = %s
            """, params)
        db.conn.commit()

        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()


@app.route('/api/metrics', methods=['GET'])
@admin_required
def get_metrics():
    """
    Get system metrics
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        # Get metrics from database
        with db.conn.cursor() as cur:
            # Success rate
            cur.execute("""
                SELECT
                    COUNT(*) as total_calls,
                    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN status IN ('pending_review', 'missing_documents') THEN 1 ELSE 0 END) as in_progress
                FROM credentialing_requests
                WHERE created_at > NOW() - INTERVAL '7 days'
            """)
            metrics = cur.fetchone()

        db.close()

        return jsonify({
            'success': True,
            'period_days': 7,
            'total_calls': metrics[0] or 0,
            'approved': metrics[1] or 0,
            'in_progress': metrics[2] or 0,
            'success_rate': round(100 * metrics[1] / metrics[0], 2) if metrics[0] and metrics[0] > 0 else 0
        }), 200

    except Exception as e:
        # Return default metrics on error (e.g., no database)
        return jsonify({
            'success': True,
            'period_days': 7,
            'total_calls': 0,
            'approved': 0,
            'in_progress': 0,
            'success_rate': 0
        }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': utcnow_iso(),
        'active_calls': len(active_calls)
    }), 200


@app.route('/api/ivr-knowledge', methods=['POST'])
@admin_required
def add_ivr_knowledge():
    """
    Add new IVR knowledge
    
    Request body:
    {
        "insurance_name": "Aetna",
        "menu_level": 1,
        "detected_phrase": "press 3 for credentialing",
        "preferred_action": "dtmf",
        "action_value": "3"
    }
    """
    try:
        from credentialing_agent import DatabaseManager, invalidate_ivr_knowledge_cache
        
        data = request.json
        preferred_action, action_value = normalize_ivr_action(
            data.get('preferred_action'),
            data.get('action_value')
        )
        db = DatabaseManager()
        
        with db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO ivr_knowledge 
                (insurance_name, menu_level, detected_phrase, preferred_action, action_value)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (
                data['insurance_name'],
                data['menu_level'],
                data['detected_phrase'],
                preferred_action,
                action_value
            ))
            
            ivr_id = cur.fetchone()[0]
            db.conn.commit()
        
        db.close()
        invalidate_ivr_knowledge_cache(data['insurance_name'])
        
        return jsonify({
            'success': True,
            'ivr_id': str(ivr_id)
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/calls', methods=['GET'])
@agent_or_above
def get_recent_calls():
    """
    Get recent credentialing calls
    """
    try:
        from credentialing_agent import DatabaseManager

        limit = request.args.get('limit', 20, type=int)
        current_user = get_current_user()
        db = DatabaseManager()

        with db.conn.cursor() as cur:
            if _is_admin_like(current_user):
                cur.execute("""
                    SELECT id, insurance_name, provider_name, npi, tax_id,
                           address, insurance_phone, status, reference_number,
                           missing_documents, turnaround_days, notes,
                           created_at, completed_at, call_mode, agent_phone, initiated_by::text
                    FROM credentialing_requests
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            else:
                cur.execute("""
                    SELECT id, insurance_name, provider_name, npi, tax_id,
                           address, insurance_phone, status, reference_number,
                           missing_documents, turnaround_days, notes,
                           created_at, completed_at, call_mode, agent_phone, initiated_by::text
                    FROM credentialing_requests
                    WHERE initiated_by = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (current_user["id"], limit))

            calls = []
            for row in cur.fetchall():
                calls.append({
                    'id': str(row[0]),
                    'insurance_name': row[1],
                    'provider_name': row[2],
                    'npi': row[3],
                    'tax_id': row[4],
                    'address': row[5],
                    'insurance_phone': row[6],
                    'status': row[7],
                    'reference_number': row[8],
                    'missing_documents': row[9] or [],
                    'turnaround_days': row[10],
                    'notes': row[11],
                    'created_at': serialize_timestamp(row[12]),
                    'completed_at': serialize_timestamp(row[13]),
                    'call_mode': row[14] or 'ai',
                    'agent_phone': row[15],
                    'initiated_by': row[16],
                })

        db.close()

        return jsonify({
            'success': True,
            'data': calls
        }), 200

    except Exception as e:
        # Return empty array on error (e.g., no database)
        return jsonify({
            'success': True,
            'data': []
        }), 200


@app.route('/api/dashboard-stats', methods=['GET'])
@admin_required
def get_dashboard_stats():
    """
    Get dashboard statistics
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            # Get stats
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE DATE(created_at) = CURRENT_DATE) as today_calls,
                    COUNT(*) FILTER (WHERE status IN ('initiated', 'pending_review')) as active,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as week_total,
                    COUNT(*) FILTER (WHERE status = 'approved' AND created_at > NOW() - INTERVAL '7 days') as week_approved
                FROM credentialing_requests
            """)
            stats = cur.fetchone()

            cur.execute("""
                SELECT COUNT(*) FROM scheduled_followups WHERE status = 'pending'
            """)
            pending_followups = cur.fetchone()[0]

        db.close()

        success_rate = round(100 * stats[3] / stats[2], 2) if stats[2] > 0 else 0

        return jsonify({
            'success': True,
            'data': {
                'total_calls_today': stats[0],
                'active_calls': stats[1],
                'success_rate_7d': success_rate,
                'avg_duration_minutes': 12,  # Placeholder
                'pending_followups': pending_followups
            }
        }), 200

    except Exception as e:
        return jsonify({
            'success': True,
            'data': {
                'total_calls_today': 0,
                'active_calls': 0,
                'success_rate_7d': 0,
                'avg_duration_minutes': 0,
                'pending_followups': 0
            }
        }), 200


@app.route('/api/knowledge/search', methods=['GET'])
@admin_required
def knowledge_search():
    """Search prior call knowledge stored in pgvector."""
    insurance = request.args.get('insurance')
    if not insurance:
        return jsonify({'success': False, 'error': 'insurance parameter is required'}), 400

    provider = request.args.get('provider')
    query = request.args.get('q')
    limit = request.args.get('limit', 5, type=int)

    try:
        kb = KnowledgeBase()
        results = kb.search(
            insurance_name=insurance,
            provider_name=provider,
            query=query,
            limit=limit
        )
        kb.close()
        return jsonify({'success': True, 'data': results}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/scheduled-followups', methods=['GET'])
@admin_required
def get_scheduled_followups():
    """
    Get all pending follow-ups
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT sf.*, cr.insurance_name, cr.provider_name
                FROM scheduled_followups sf
                JOIN credentialing_requests cr ON sf.request_id = cr.id
                WHERE sf.status = 'pending'
                ORDER BY sf.scheduled_date ASC
                LIMIT 20
            """)

            followups = []
            for row in cur.fetchall():
                followups.append({
                    'id': str(row[0]),
                    'request_id': str(row[1]),
                    'scheduled_date': serialize_timestamp(row[2]),
                    'action_type': row[3],
                    'status': row[4] if len(row) > 4 else 'pending',
                    'insurance_name': row[-2],
                    'provider_name': row[-1]
                })

        db.close()

        return jsonify({
            'success': True,
            'data': followups
        }), 200

    except Exception as e:
        return jsonify({
            'success': True,
            'data': []
        }), 200


@app.route('/api/followup/<followup_id>/execute', methods=['POST'])
@admin_required
def execute_followup(followup_id: str):
    """
    Execute a scheduled follow-up
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            # Get followup details
            cur.execute("""
                SELECT sf.*, cr.*
                FROM scheduled_followups sf
                JOIN credentialing_requests cr ON sf.request_id = cr.id
                WHERE sf.id = %s
            """, (followup_id,))

            followup = cur.fetchone()
            if not followup:
                return jsonify({
                    'success': False,
                    'error': 'Follow-up not found'
                }), 404

            # Mark as completed (in real implementation, would trigger actual call)
            cur.execute("""
                UPDATE scheduled_followups
                SET status = 'completed'
                WHERE id = %s
            """, (followup_id,))
            db.conn.commit()

        db.close()

        return jsonify({
            'success': True,
            'message': 'Follow-up executed'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/call/<call_id>/cancel', methods=['POST'])
@admin_required
def cancel_call(call_id: str):
    """
    Cancel an active call
    """
    try:
        # Check if call exists in active calls
        if call_id in active_calls:
            agent = active_calls[call_id]
            # Signal the agent to stop
            if call_id in call_states:
                call_states[call_id]['should_continue'] = False

            # Try to end the Twilio call if we have a call_sid
            if call_id in call_states and call_states[call_id].get('call_sid'):
                try:
                    twilio_client.calls(call_states[call_id]['call_sid']).update(status='completed')
                except Exception:
                    pass  # Call may have already ended

            del active_calls[call_id]

        _call_sid_for_flush = call_states.get(call_id, {}).get('call_sid') if call_id in call_states else None
        _flush_call_logs(call_id, _call_sid_for_flush)

        return jsonify({
            'success': True,
            'message': 'Call cancelled'
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ivr-knowledge/<insurance_name>', methods=['GET'])
@agent_or_above
def get_ivr_knowledge(insurance_name: str):
    """
    Get IVR knowledge for a specific insurance provider
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, insurance_name, menu_level, detected_phrase,
                       preferred_action, action_value, confidence_threshold,
                       success_rate, attempts
                FROM ivr_knowledge
                WHERE insurance_name = %s
                ORDER BY menu_level ASC
            """, (insurance_name,))

            knowledge = []
            for row in cur.fetchall():
                preferred_action, action_value = normalize_ivr_action(row[4], row[5])
                knowledge.append({
                    'id': str(row[0]),
                    'insurance_name': row[1],
                    'menu_level': row[2],
                    'detected_phrase': row[3],
                    'preferred_action': preferred_action,
                    'action_value': action_value,
                    'confidence_threshold': row[6],
                    'success_rate': row[7],
                    'attempts': row[8]
                })

        db.close()

        return jsonify({
            'success': True,
            'data': knowledge
        }), 200

    except Exception as e:
        return jsonify({
            'success': True,
            'data': []
        }), 200


# ============================================================================
# Insurance Provider CRUD Endpoints
# ============================================================================

@app.route('/api/insurance-providers', methods=['GET'])
@agent_or_above
def get_insurance_providers():
    """
    Get all insurance providers
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, insurance_name, phone_number, department,
                       best_call_times, average_wait_time_minutes, notes, last_updated,
                       ivr_asks_npi, ivr_npi_method, ivr_asks_tax_id, ivr_tax_id_method, ivr_tax_id_digits_to_send,
                       ivr_npi_suffix, ivr_tax_id_suffix
                FROM insurance_providers
                ORDER BY insurance_name ASC
            """)

            providers = []
            for row in cur.fetchall():
                providers.append({
                    'id': str(row[0]),
                    'insurance_name': row[1],
                    'phone_number': row[2],
                    'department': row[3],
                    'best_call_times': row[4],
                    'average_wait_time_minutes': row[5],
                    'notes': row[6],
                    'last_updated': row[7].isoformat() if row[7] else None,
                    'ivr_asks_npi': row[8] or False,
                    'ivr_npi_method': row[9] or 'speech',
                    'ivr_asks_tax_id': row[10] or False,
                    'ivr_tax_id_method': row[11] or 'speech',
                    'ivr_tax_id_digits_to_send': row[12],
                    'ivr_npi_suffix': row[13] or None,
                    'ivr_tax_id_suffix': row[14] or None,
                })

        db.close()

        return jsonify({
            'success': True,
            'data': providers
        }), 200

    except Exception as e:
        print(f"Error getting insurance providers: {e}")
        return jsonify({
            'success': True,
            'data': []
        }), 200


@app.route('/api/insurance-providers', methods=['POST'])
@admin_required
def add_insurance_provider():
    """
    Add a new insurance provider

    Request body:
    {
        "insurance_name": "Aetna",
        "phone_number": "+18001234567",
        "department": "Provider Services",
        "best_call_times": {"days": ["monday", "tuesday"], "hours": [9, 17]},
        "average_wait_time_minutes": 15,
        "notes": "Best to call early morning"
    }
    """
    from credentialing_agent import DatabaseManager

    data = request.json
    db = DatabaseManager()
    try:
        with db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO insurance_providers
                (insurance_name, phone_number, department, best_call_times,
                 average_wait_time_minutes, notes,
                 ivr_asks_npi, ivr_npi_method, ivr_asks_tax_id, ivr_tax_id_method, ivr_tax_id_digits_to_send,
                 ivr_npi_suffix, ivr_tax_id_suffix)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                data['insurance_name'],
                data['phone_number'],
                data.get('department'),
                json.dumps(data.get('best_call_times')) if data.get('best_call_times') else None,
                data.get('average_wait_time_minutes'),
                data.get('notes'),
                data.get('ivr_asks_npi', False),
                data.get('ivr_npi_method', 'speech'),
                data.get('ivr_asks_tax_id', False),
                data.get('ivr_tax_id_method', 'speech'),
                data.get('ivr_tax_id_digits_to_send'),
                data.get('ivr_npi_suffix'),
                data.get('ivr_tax_id_suffix'),
            ))

            provider_id = cur.fetchone()[0]
            db.conn.commit()

        return jsonify({
            'success': True,
            'id': str(provider_id),
            'message': 'Insurance provider added successfully'
        }), 201
    except Exception as e:
        print(f"Error adding insurance provider: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        db.close()


@app.route('/api/insurance-providers/<provider_id>', methods=['PUT'])
@admin_required
def update_insurance_provider(provider_id: str):
    """
    Update an existing insurance provider
    """
    from credentialing_agent import DatabaseManager, invalidate_ivr_knowledge_cache

    data = request.json
    db = DatabaseManager()
    try:
        with db.conn.cursor() as cur:
            # Read old name first so IVR rows can be kept in sync when renamed.
            cur.execute("""
                SELECT insurance_name
                FROM insurance_providers
                WHERE id = %s
            """, (provider_id,))
            existing = cur.fetchone()
            if not existing:
                return jsonify({
                    'success': False,
                    'error': 'Insurance provider not found'
                }), 404

            old_insurance_name = existing[0]
            new_insurance_name = data['insurance_name']

            cur.execute("""
                UPDATE insurance_providers
                SET insurance_name = %s,
                    phone_number = %s,
                    department = %s,
                    best_call_times = %s,
                    average_wait_time_minutes = %s,
                    notes = %s,
                    ivr_asks_npi = %s,
                    ivr_npi_method = %s,
                    ivr_asks_tax_id = %s,
                    ivr_tax_id_method = %s,
                    ivr_tax_id_digits_to_send = %s,
                    ivr_npi_suffix = %s,
                    ivr_tax_id_suffix = %s,
                    last_updated = NOW()
                WHERE id = %s
                RETURNING id
            """, (
                data['insurance_name'],
                data['phone_number'],
                data.get('department'),
                json.dumps(data.get('best_call_times')) if data.get('best_call_times') else None,
                data.get('average_wait_time_minutes'),
                data.get('notes'),
                data.get('ivr_asks_npi', False),
                data.get('ivr_npi_method', 'speech'),
                data.get('ivr_asks_tax_id', False),
                data.get('ivr_tax_id_method', 'speech'),
                data.get('ivr_tax_id_digits_to_send'),
                data.get('ivr_npi_suffix'),
                data.get('ivr_tax_id_suffix'),
                provider_id
            ))

            cur.fetchone()

            # Keep IVR knowledge visible after provider renames.
            if old_insurance_name != new_insurance_name:
                cur.execute("""
                    UPDATE ivr_knowledge
                    SET insurance_name = %s
                    WHERE insurance_name = %s
                """, (new_insurance_name, old_insurance_name))

            db.conn.commit()

        invalidate_ivr_knowledge_cache(old_insurance_name)
        if old_insurance_name != new_insurance_name:
            invalidate_ivr_knowledge_cache(new_insurance_name)

        return jsonify({
            'success': True,
            'message': 'Insurance provider updated successfully'
        }), 200
    except Exception as e:
        print(f"Error updating insurance provider: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    finally:
        db.close()


@app.route('/api/insurance-providers/<provider_id>', methods=['DELETE'])
@admin_required
def delete_insurance_provider(provider_id: str):
    """
    Delete an insurance provider
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            # First delete associated IVR knowledge
            cur.execute("""
                DELETE FROM ivr_knowledge
                WHERE insurance_name = (
                    SELECT insurance_name FROM insurance_providers WHERE id = %s
                )
            """, (provider_id,))

            # Then delete the provider
            cur.execute("""
                DELETE FROM insurance_providers
                WHERE id = %s
                RETURNING id
            """, (provider_id,))

            result = cur.fetchone()
            if not result:
                db.close()
                return jsonify({
                    'success': False,
                    'error': 'Insurance provider not found'
                }), 404

            db.conn.commit()

        db.close()

        return jsonify({
            'success': True,
            'message': 'Insurance provider deleted successfully'
        }), 200

    except Exception as e:
        print(f"Error deleting insurance provider: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ivr-knowledge/<ivr_id>', methods=['DELETE'])
@admin_required
def delete_ivr_knowledge(ivr_id: str):
    """
    Delete an IVR knowledge entry
    """
    try:
        from credentialing_agent import DatabaseManager, invalidate_ivr_knowledge_cache

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                DELETE FROM ivr_knowledge
                WHERE id = %s
                RETURNING id, insurance_name
            """, (ivr_id,))

            result = cur.fetchone()
            if not result:
                db.close()
                return jsonify({
                    'success': False,
                    'error': 'IVR knowledge entry not found'
                }), 404

            db.conn.commit()

        db.close()
        invalidate_ivr_knowledge_cache(result[1])

        return jsonify({
            'success': True,
            'message': 'IVR knowledge deleted successfully'
        }), 200

    except Exception as e:
        print(f"Error deleting IVR knowledge: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ivr-knowledge/<ivr_id>', methods=['PUT'])
@admin_required
def update_ivr_knowledge(ivr_id: str):
    """
    Update an IVR knowledge entry
    """
    try:
        from credentialing_agent import DatabaseManager, invalidate_ivr_knowledge_cache

        data = request.json
        preferred_action, action_value = normalize_ivr_action(
            data.get('preferred_action'),
            data.get('action_value')
        )
        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                UPDATE ivr_knowledge
                SET menu_level = %s,
                    detected_phrase = %s,
                    preferred_action = %s,
                    action_value = %s,
                    last_updated = NOW()
                WHERE id = %s
                RETURNING id, insurance_name
            """, (
                data['menu_level'],
                data['detected_phrase'],
                preferred_action,
                action_value,
                ivr_id
            ))

            result = cur.fetchone()
            if not result:
                db.close()
                return jsonify({
                    'success': False,
                    'error': 'IVR knowledge entry not found'
                }), 404

            db.conn.commit()

        db.close()
        invalidate_ivr_knowledge_cache(result[1])

        return jsonify({
            'success': True,
            'message': 'IVR knowledge updated successfully'
        }), 200

    except Exception as e:
        print(f"Error updating IVR knowledge: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Call Metrics API Endpoints
# ============================================================================

@app.route('/api/call-metrics/<call_id>', methods=['GET'])
@admin_required
def get_call_metrics_endpoint(call_id: str):
    """Get call metrics for a specific call."""
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        metrics = db.get_call_metrics(call_id)
        db.close()

        if not metrics:
            return jsonify({'success': False, 'error': 'Metrics not found'}), 404

        # Convert UUID and datetime for JSON serialization
        metrics['id'] = str(metrics['id'])
        if metrics.get('request_id'):
            metrics['request_id'] = str(metrics['request_id'])
        if metrics.get('created_at'):
            metrics['created_at'] = metrics['created_at'].isoformat()

        return jsonify({'success': True, 'data': metrics}), 200
    except Exception as e:
        logger.error(f"Error getting call metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/call-metrics', methods=['GET'])
@admin_required
def get_all_call_metrics():
    """Get call metrics with optional filtering."""
    try:
        from credentialing_agent import DatabaseManager

        limit = request.args.get('limit', 50, type=int)
        successful_param = request.args.get('successful')
        successful_only = None
        if successful_param is not None:
            successful_only = successful_param.lower() == 'true'

        db = DatabaseManager()
        metrics_list = db.get_all_call_metrics(limit=limit, successful_only=successful_only)
        db.close()

        # Serialize for JSON
        for m in metrics_list:
            m['id'] = str(m['id'])
            if m.get('request_id'):
                m['request_id'] = str(m['request_id'])
            if m.get('created_at'):
                m['created_at'] = m['created_at'].isoformat()

        return jsonify({'success': True, 'data': metrics_list}), 200
    except Exception as e:
        logger.error(f"Error getting all call metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# System Config API Endpoints
# ============================================================================

@app.route('/api/config', methods=['GET'])
@admin_required
def get_all_config():
    """Get all system configuration values."""
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        configs = db.get_all_configs()
        db.close()

        # Serialize for JSON
        for config in configs:
            config['id'] = str(config['id'])
            if config.get('last_updated'):
                config['last_updated'] = config['last_updated'].isoformat()

        return jsonify({'success': True, 'data': configs}), 200
    except Exception as e:
        logger.error(f"Error getting all configs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/<config_key>', methods=['GET'])
@admin_required
def get_config_value(config_key: str):
    """Get a specific configuration value."""
    try:
        from credentialing_agent import DatabaseManager
        db = DatabaseManager()
        value = db.get_config(config_key)
        db.close()

        if value is None:
            return jsonify({'success': False, 'error': 'Config not found'}), 404

        return jsonify({'success': True, 'key': config_key, 'value': value}), 200
    except Exception as e:
        logger.error(f"Error getting config {config_key}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/<config_key>', methods=['PUT'])
@admin_required
def update_config_value(config_key: str):
    """Update a configuration value."""
    try:
        from credentialing_agent import DatabaseManager

        data = request.json
        if 'value' not in data:
            return jsonify({'success': False, 'error': 'value is required'}), 400

        db = DatabaseManager()
        success = db.set_config(
            config_key,
            data['value'],
            data.get('description')
        )
        db.close()

        return jsonify({'success': success, 'message': 'Config updated'}), 200
    except Exception as e:
        logger.error(f"Error updating config {config_key}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/config/<config_key>', methods=['DELETE'])
@admin_required
def delete_config_value(config_key: str):
    """Delete a configuration value."""
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()
        success = db.delete_config(config_key)
        db.close()

        if not success:
            return jsonify({'success': False, 'error': 'Config not found'}), 404

        return jsonify({'success': True, 'message': 'Config deleted'}), 200
    except Exception as e:
        logger.error(f"Error deleting config {config_key}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# Audit Log API Endpoints
# ============================================================================

@app.route('/api/audit-logs', methods=['GET'])
@super_admin_required
def get_audit_logs_endpoint():
    """
    Get audit logs with filtering and pagination.

    Query params:
    - user_id: Filter by user
    - action: Filter by action (INSERT, UPDATE, DELETE, etc.)
    - resource_type: Filter by resource type
    - resource_id: Filter by resource ID
    - start_date: Filter from date (ISO format)
    - end_date: Filter to date (ISO format)
    - limit: Max results (default 100)
    - offset: Pagination offset (default 0)
    """
    try:
        from credentialing_agent import DatabaseManager

        # Parse query params
        user_id = request.args.get('user_id')
        action = request.args.get('action')
        resource_type = request.args.get('resource_type')
        resource_id = request.args.get('resource_id')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)

        start_date = None
        end_date = None
        if request.args.get('start_date'):
            start_date = datetime.fromisoformat(request.args.get('start_date'))
        if request.args.get('end_date'):
            end_date = datetime.fromisoformat(request.args.get('end_date'))

        db = DatabaseManager()
        logs, total_count = db.get_audit_logs(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        db.close()

        # Serialize for JSON
        for log in logs:
            log['id'] = str(log['id'])
            if log.get('timestamp'):
                log['timestamp'] = log['timestamp'].isoformat()
            if log.get('ip_address'):
                log['ip_address'] = str(log['ip_address'])

        return jsonify({
            'success': True,
            'data': logs,
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting audit logs: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/audit-logs', methods=['POST'])
@super_admin_required
def create_audit_log():
    """
    Manually create an audit log entry.

    Request body:
    {
        "action": "MANUAL_OVERRIDE",
        "resource_type": "call",
        "resource_id": "uuid-here",
        "details": {"reason": "..."},
        "user_id": "optional-user-id"
    }
    """
    try:
        from credentialing_agent import DatabaseManager

        data = request.json
        if not data.get('action'):
            return jsonify({'success': False, 'error': 'action is required'}), 400

        # Get IP address from request
        ip_address = request.remote_addr

        db = DatabaseManager()
        log_id = db.log_audit(
            action=data['action'],
            resource_type=data.get('resource_type'),
            resource_id=data.get('resource_id'),
            details=data.get('details'),
            user_id=data.get('user_id'),
            ip_address=ip_address,
        )
        db.close()

        return jsonify({
            'success': True,
            'id': log_id,
            'message': 'Audit log created'
        }), 201
    except Exception as e:
        logger.error(f"Error creating audit log: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# AI vs Real-Agent Transfer Endpoint
# ============================================================================

@app.route('/api/transfer-to-agent', methods=['POST'])
@admin_or_above
def transfer_to_agent():
    """
    Mid-call transfer from AI to a real human agent using Twilio Conference.

    Request body:
    {
        "call_id": "<runtime call UUID>",
        "agent_phone": "+15551234567"
    }

    Behaviour:
    1. Looks up the active call by call_id.
    2. Redirects the live Twilio call into a Conference room (named by call_id).
    3. Dials agent_phone into the same Conference.
    4. Sets transfer_to_agent=True in call_states so the AI polling loop exits.
    5. Persists call_mode='agent', agent_phone, and conference_sid to the DB.
    """
    try:
        data = request.json or {}
        call_id = data.get('call_id')
        agent_phone = data.get('agent_phone')

        if not call_id:
            return jsonify({'success': False, 'error': 'call_id is required'}), 400
        if not agent_phone:
            return jsonify({'success': False, 'error': 'agent_phone is required'}), 400

        # Fetch in-memory call state — try runtime call_id first, then fall back
        # to matching by DB request ID (the UI sends the DB UUID from /api/calls)
        call_state = call_states.get(call_id)
        if not call_state:
            call_state = next(
                (s for s in call_states.values() if s.get('db_request_id') == call_id),
                None
            )
        if not call_state:
            return jsonify({'success': False, 'error': 'Call not found or already ended'}), 404

        normalized_agent_phone = _normalize_us_phone(agent_phone)
        if not normalized_agent_phone:
            return jsonify({'success': False, 'error': 'agent_phone must be a valid US phone number'}), 400
        if _same_us_phone(normalized_agent_phone, call_state.get('insurance_phone')):
            return jsonify({
                'success': False,
                'error': 'Agent phone must be a staff/user phone, not the insurance phone.'
            }), 400
        agent_phone = normalized_agent_phone

        call_sid = call_state.get('call_sid')
        if not call_sid:
            return jsonify({'success': False, 'error': 'Call SID not yet available; call may still be connecting'}), 400

        from_number = call_state.get('from_number') or os.getenv("TWILIO_PHONE_NUMBER")
        conf_name = call_id  # Conference room name = call_id for uniqueness
        recording_callback_url = build_recording_status_callback(
            get_callback_base(request),
            call_id=call_id,
            request_id=call_state.get('db_request_id'),
        )

        # TwiML to redirect the insurance-side call into the conference
        insurance_leg_twiml = build_conference_twiml(
            conf_name,
            recording_callback_url=recording_callback_url,
            end_on_exit=True,
        )

        # TwiML for the agent leg (does not end conference when agent hangs up)
        agent_leg_twiml = build_conference_twiml(
            conf_name,
            recording_callback_url=recording_callback_url,
            end_on_exit=False,
        )

        # Dial the real agent first — if this fails, the insurance call is untouched
        try:
            agent_call = twilio_client.calls.create(
                to=agent_phone,
                from_=from_number,
                twiml=agent_leg_twiml,
                timeout=30,
            )
        except Exception as dial_err:
            logger.error(f"[transfer-to-agent] Failed to dial agent {agent_phone}: {dial_err}")
            return jsonify({'success': False, 'error': f'Failed to dial agent: {dial_err}'}), 500

        conference_sid = agent_call.sid  # Note: this is agent call SID, not a Twilio Conference SID
        logger.info(f"[transfer-to-agent] Dialed agent {agent_phone} into conference={conf_name}, agent_call_sid={conference_sid}")

        # Only redirect the insurance call after the agent dial succeeds
        try:
            twilio_client.calls(call_sid).update(twiml=insurance_leg_twiml)
            logger.info(f"[transfer-to-agent] Redirected call_sid={call_sid} into conference={conf_name}")
        except Exception as redirect_err:
            logger.error(f"[transfer-to-agent] Failed to redirect insurance call: {redirect_err}")
            # Cancel the agent call we just placed so it doesn't ring forever
            try:
                twilio_client.calls(agent_call.sid).update(status='canceled')
            except Exception:
                pass
            return jsonify({'success': False, 'error': f'Failed to redirect insurance call: {redirect_err}'}), 500

        # Signal the AI agent loop to stop gracefully
        call_state['transfer_to_agent'] = True
        call_state['call_mode'] = 'agent'
        call_state['agent_phone'] = agent_phone
        call_state['conference_sid'] = conference_sid  # Note: this is agent call SID, not a Twilio Conference SID

        # Persist the transfer in the DB (background thread — non-blocking)
        def _persist_transfer():
            try:
                from credentialing_agent import DatabaseManager
                db = DatabaseManager()
                request_id = call_state.get('db_request_id')
                if request_id:
                    with db.conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE credentialing_requests
                            SET call_mode = 'agent',
                                agent_phone = %s,
                                conference_sid = %s,
                                updated_at = NOW()
                            WHERE id = %s
                            """,
                            (agent_phone, conference_sid, request_id)
                        )
                        db.conn.commit()
                db.close()
            except Exception as db_err:
                logger.warning(f"[transfer-to-agent] Could not persist transfer to DB: {db_err}")

        threading.Thread(target=_persist_transfer, daemon=True).start()

        return jsonify({
            'success': True,
            'message': 'Call transferred to real agent',
            'conference_name': conf_name,
            'agent_call_sid': conference_sid,
        }), 200

    except Exception as e:
        logger.error(f"[transfer-to-agent] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================================
# WebSocket handlers for Twilio Media Streams
# ============================================================================

# Store active media stream connections
media_streams: Dict[str, dict] = {}


@socketio.on('connect', namespace='/media-stream')
def handle_connect():
    """Handle WebSocket connection from Twilio"""
    print(f"🔌 WebSocket connected: {request.sid}")


@socketio.on('disconnect', namespace='/media-stream')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f"🔌 WebSocket disconnected: {request.sid}")
    # Close any Deepgram session bound to this socket
    if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available():
        deepgram_streaming.close_session(request.sid)
    # Clean up any associated stream data
    if request.sid in media_streams:
        del media_streams[request.sid]


@socketio.on('message', namespace='/media-stream')
def handle_message(message):
    """
    Handle incoming Twilio Media Stream messages

    Twilio sends JSON messages with event types:
    - connected: WebSocket connected
    - start: Stream started (contains streamSid, callSid, etc.)
    - media: Audio payload (base64 encoded mu-law audio)
    - stop: Stream stopped
    """
    try:
        if isinstance(message, str):
            data = json.loads(message)
        else:
            data = message

        event_type = data.get('event')

        if event_type == 'connected':
            print(f"📞 Twilio stream connected: {data.get('protocol')}")

        elif event_type == 'start':
            stream_sid = data.get('streamSid')
            start_payload = data.get('start', {}) or {}
            call_sid = start_payload.get('callSid')
            custom_params = start_payload.get('customParameters', {}) or {}
            stream_call_id = custom_params.get('call_id') or call_sid
            print(f"🎙️ Stream started - StreamSID: {stream_sid}, CallSID: {call_sid}, call_id: {stream_call_id}")

            # Store stream info
            media_streams[request.sid] = {
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'call_id': stream_call_id,
                'audio_buffer': [],
                'transcript': ''
            }

            # Update call state if we have it
            for call_id, state in call_states.items():
                if state.get('call_sid') == call_sid:
                    state['current_audio_type'] = AudioType.HUMAN_SPEECH
                    print(f"📊 Updated call {call_id} audio type to HUMAN_SPEECH")
                    break

            # Open the Deepgram session for this WebSocket if streaming is on
            if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available() and call_sid:
                deepgram_streaming.open_session(
                    socket_sid=request.sid,
                    call_sid=call_sid,
                    on_final=_on_deepgram_final,
                )

        elif event_type == 'media':
            # Audio payload — base64 encoded mu-law @ 8kHz from Twilio.
            payload = data.get('media', {}).get('payload', '')
            if payload and request.sid in media_streams:
                if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available():
                    try:
                        audio_bytes = base64.b64decode(payload)
                        deepgram_streaming.feed_audio(request.sid, audio_bytes)
                    except Exception as fe:
                        logger.debug(f"Failed to forward audio chunk: {fe}")

        elif event_type == 'stop':
            stream_sid = data.get('streamSid')
            print(f"🛑 Stream stopped: {stream_sid}")
            if ENABLE_DEEPGRAM_STREAMING and deepgram_streaming.is_available():
                deepgram_streaming.close_session(request.sid)
            if request.sid in media_streams:
                del media_streams[request.sid]

    except Exception as e:
        print(f"❌ Error handling media stream message: {e}")


def send_audio_to_twilio(stream_sid: str, audio_base64: str):
    """
    Send audio back to Twilio (for TTS responses)

    Args:
        stream_sid: The Twilio stream SID
        audio_base64: Base64 encoded mu-law audio
    """
    message = {
        "event": "media",
        "streamSid": stream_sid,
        "media": {
            "payload": audio_base64
        }
    }
    socketio.emit('message', json.dumps(message), namespace='/media-stream')


# ============================================================================
# Twilio Number Management Routes (admin only)
# ============================================================================

@app.route('/api/twilio-numbers', methods=['GET'])
@admin_required
def list_twilio_numbers():
    """List all Twilio numbers with their status."""
    conn = _tp_get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, phone_number, friendly_name, is_active, "
                "       current_call_id, current_call_sid, in_use_since, "
                "       created_at, updated_at "
                "FROM twilio_numbers ORDER BY created_at ASC"
            )
            rows = cur.fetchall()
        numbers = []
        for row in rows:
            d = dict(row)
            d["id"] = str(d["id"])
            for ts_field in ("in_use_since", "created_at", "updated_at"):
                if d.get(ts_field):
                    d[ts_field] = d[ts_field].isoformat()
            numbers.append(d)
        return jsonify({"numbers": numbers}), 200
    finally:
        _tp_put_conn(conn)


@app.route('/api/twilio-numbers', methods=['POST'])
@admin_required
def add_twilio_number():
    """Add a new Twilio number to the pool."""
    data = request.get_json(silent=True) or {}
    phone_number = (data.get("phone_number") or "").strip()
    friendly_name = (data.get("friendly_name") or "").strip() or None

    if not phone_number:
        return jsonify({"error": "phone_number is required"}), 400

    # Basic E.164 validation
    if not re.match(r"^\+\d{10,15}$", phone_number):
        return jsonify({"error": "Phone number must be in E.164 format (e.g. +13513007215)"}), 400

    conn = _tp_get_conn()
    try:
        user_id = get_current_user_id()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "INSERT INTO twilio_numbers (phone_number, friendly_name, added_by) "
                "VALUES (%s, %s, %s) "
                "RETURNING id, phone_number, friendly_name, is_active, created_at",
                (phone_number, friendly_name, user_id),
            )
            row = cur.fetchone()
        conn.commit()
        d = dict(row)
        d["id"] = str(d["id"])
        if d.get("created_at"):
            d["created_at"] = d["created_at"].isoformat()
        return jsonify({"number": d}), 201
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return jsonify({"error": "This phone number already exists"}), 409
    except Exception as e:
        conn.rollback()
        logger.error("add_twilio_number error: %s", e)
        return jsonify({"error": str(e)}), 500
    finally:
        _tp_put_conn(conn)


@app.route('/api/twilio-numbers/<number_id>', methods=['PATCH'])
@admin_required
def update_twilio_number(number_id):
    """Toggle is_active for a Twilio number."""
    data = request.get_json(silent=True) or {}

    if "is_active" not in data:
        return jsonify({"error": "is_active field is required"}), 400

    conn = _tp_get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "UPDATE twilio_numbers SET is_active = %s, updated_at = NOW() "
                "WHERE id = %s "
                "RETURNING id, phone_number, friendly_name, is_active, "
                "         current_call_id, created_at, updated_at",
                (bool(data["is_active"]), number_id),
            )
            row = cur.fetchone()
        conn.commit()
        if not row:
            return jsonify({"error": "Number not found"}), 404
        d = dict(row)
        d["id"] = str(d["id"])
        for ts_field in ("created_at", "updated_at"):
            if d.get(ts_field):
                d[ts_field] = d[ts_field].isoformat()
        return jsonify({"number": d}), 200
    finally:
        _tp_put_conn(conn)


@app.route('/api/twilio-numbers/<number_id>', methods=['DELETE'])
@admin_required
def delete_twilio_number(number_id):
    """Delete a Twilio number (only if not currently in use)."""
    conn = _tp_get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Check if in use
            cur.execute(
                "SELECT current_call_id FROM twilio_numbers WHERE id = %s",
                (number_id,),
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "Number not found"}), 404
            if row["current_call_id"]:
                return jsonify({"error": "Cannot delete a number that is currently in use"}), 409

            cur.execute("DELETE FROM twilio_numbers WHERE id = %s", (number_id,))
        conn.commit()
        return jsonify({"message": "Number deleted"}), 200
    finally:
        _tp_put_conn(conn)


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == '__main__':
    # For development - use socketio.run for WebSocket support
    print("🚀 Starting server with WebSocket support...")
    socketio.run(
        app,
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true',
        allow_unsafe_werkzeug=True
    )
