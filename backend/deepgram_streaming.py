"""
Deepgram live-streaming session manager.

One DeepgramSession per Twilio Media Stream WebSocket. Forwards mu-law audio
to Deepgram's nova-2 streaming endpoint and fires `on_final` whenever Deepgram
produces a finalized transcript.

Compared to Twilio's <Gather speechTimeout="2"> (which waits 2 seconds of silence
before posting a transcript), Deepgram with endpointing=300 returns a final in
roughly 300-500 ms after the speaker stops. That is the ~1.5-2 second latency
cut you are after.

Activated by ENABLE_DEEPGRAM_STREAMING=true in the environment.
"""

import logging
import os
import threading
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEEPGRAM_IMPORT_ERROR = None
try:
    from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
    _DEEPGRAM_AVAILABLE = True
except Exception as _e:
    _DEEPGRAM_AVAILABLE = False
    _DEEPGRAM_IMPORT_ERROR = _e
    logger.warning(f"deepgram-sdk import failed; Deepgram streaming disabled. "
                   f"Reason: {type(_e).__name__}: {_e}")


class DeepgramSession:
    """One Deepgram live transcription connection, scoped to a single Twilio call."""

    def __init__(self, call_sid: str, on_final: Callable[[str, str], None]):
        self.call_sid = call_sid
        self.on_final = on_final
        self.conn = None
        self.started = False
        self._chunks_sent = 0
        self._lock = threading.Lock()

        if not _DEEPGRAM_AVAILABLE:
            return

        api_key = (os.getenv("DEEPGRAM_API_KEY") or "").strip()
        if not api_key:
            logger.error("DEEPGRAM_API_KEY missing — Deepgram session not created")
            return

        try:
            client = DeepgramClient(api_key)
            # SDK v3 exposes the websocket variant under listen.websocket
            try:
                self.conn = client.listen.websocket.v("1")
            except AttributeError:
                # Older v3 SDKs use listen.live
                self.conn = client.listen.live.v("1")

            self.conn.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.conn.on(LiveTranscriptionEvents.Error,
                         lambda *a, **kw: logger.error(f"Deepgram error [{self.call_sid}]: {a} {kw}"))
            self.conn.on(LiveTranscriptionEvents.Close,
                         lambda *a, **kw: logger.info(f"Deepgram closed [{self.call_sid}]"))
        except Exception as e:
            logger.error(f"Could not create Deepgram client: {e}", exc_info=True)
            self.conn = None

    def start(self) -> bool:
        """Open the streaming connection. Returns True on success."""
        if not self.conn:
            return False

        options = LiveOptions(
            model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
            language="en-US",
            encoding="mulaw",        # Twilio Media Streams sends mu-law @ 8kHz
            sample_rate=8000,
            channels=1,
            interim_results=True,
            smart_format=True,
            endpointing=300,         # ms of silence before flushing a final
            vad_events=True,
        )
        try:
            ok = self.conn.start(options)
            self.started = bool(ok)
            if self.started:
                logger.info(f"Deepgram session started [{self.call_sid}]")
            else:
                logger.error(f"Deepgram session failed to start [{self.call_sid}]")
            return self.started
        except Exception as e:
            logger.error(f"Deepgram start failed [{self.call_sid}]: {e}", exc_info=True)
            return False

    def send_audio(self, audio_bytes: bytes):
        """Forward one chunk of mu-law audio (~20ms of PCMU = 160 bytes)."""
        if not self.started or not self.conn:
            return
        try:
            self.conn.send(audio_bytes)
            self._chunks_sent += 1
        except Exception as e:
            logger.error(f"Deepgram send failed [{self.call_sid}]: {e}")

    def stop(self):
        """Close the connection cleanly."""
        if not self.conn:
            return
        try:
            self.conn.finish()
            logger.info(f"Deepgram session stopped [{self.call_sid}], chunks={self._chunks_sent}")
        except Exception as e:
            logger.error(f"Deepgram stop failed [{self.call_sid}]: {e}")

    def _on_transcript(self, *args, **kwargs):
        """Internal handler — pulls transcript text out of the SDK event."""
        try:
            # SDK passes the result either positionally or as `result=` kwarg
            result = kwargs.get("result")
            if result is None:
                for a in args:
                    if hasattr(a, "channel"):
                        result = a
                        break
            if result is None:
                return

            transcript = result.channel.alternatives[0].transcript
            if not transcript or not transcript.strip():
                return

            is_final = bool(getattr(result, "is_final", False))
            if not is_final:
                logger.debug(f"DG interim [{self.call_sid}]: {transcript}")
                return

            logger.info(f"DG final [{self.call_sid}]: \"{transcript}\"")
            try:
                self.on_final(self.call_sid, transcript)
            except Exception as cb_err:
                logger.error(f"on_final callback failed [{self.call_sid}]: {cb_err}", exc_info=True)
        except Exception as e:
            logger.error(f"_on_transcript error [{self.call_sid}]: {e}", exc_info=True)


# ----- Module-level registry, keyed by Socket.IO sid (one per WebSocket) -----

_sessions_lock = threading.Lock()
_sessions: Dict[str, DeepgramSession] = {}


def open_session(socket_sid: str, call_sid: str,
                 on_final: Callable[[str, str], None]) -> Optional[DeepgramSession]:
    """Open a Deepgram session for the given Twilio Media Stream socket sid."""
    with _sessions_lock:
        if socket_sid in _sessions:
            return _sessions[socket_sid]
    session = DeepgramSession(call_sid, on_final)
    if session.start():
        with _sessions_lock:
            _sessions[socket_sid] = session
        return session
    return None


def feed_audio(socket_sid: str, audio_bytes: bytes):
    """Forward a chunk of mu-law audio to the session bound to this socket."""
    with _sessions_lock:
        s = _sessions.get(socket_sid)
    if s:
        s.send_audio(audio_bytes)


def close_session(socket_sid: str):
    """Close and remove the session for this socket."""
    with _sessions_lock:
        s = _sessions.pop(socket_sid, None)
    if s:
        s.stop()


def is_available() -> bool:
    """True if the deepgram-sdk import succeeded."""
    return _DEEPGRAM_AVAILABLE
