"""
Flask API Server for Autonomous Credentialing Agent
Handles Twilio webhooks and real-time audio streaming
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables BEFORE any other imports that use them
load_dotenv()

import asyncio
import json
import uuid
from datetime import datetime
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

from credentialing_agent import (
    CredentialingAgent,
    CredentialingState,
    CallState,
    AudioType
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state management
active_calls: Dict[str, CredentialingAgent] = {}
call_states: Dict[str, CredentialingState] = {}
call_states_by_sid: Dict[str, CredentialingState] = {}
call_state_lock = threading.Lock()
audio_queues: Dict[str, Queue] = {}

# Twilio client
twilio_client = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)

# =============================================================================
# Helper utilities
# =============================================================================

def _is_public_url(url: str) -> bool:
    """Return True when the URL is not obviously localhost/loopback."""
    return not re.match(r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:?|/)", (url or "").lower())


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
    """Store call state by call_id and call_sid (if present) with a light lock."""
    with call_state_lock:
        call_states[call_id] = state
        if state.get('call_sid'):
            call_states_by_sid[state['call_sid']] = state


def bind_call_sid(call_id: str, call_sid: str):
    """Bind a call_sid to existing state for faster lookups."""
    with call_state_lock:
        if call_id in call_states:
            call_states[call_id]['call_sid'] = call_sid
            call_states_by_sid[call_sid] = call_states[call_id]


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
        # Generate audio stream
        audio_stream = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=text,
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128",
        )

        # Save to temp file
        temp_dir = os.path.join(os.getcwd(), 'temp_audio')
        os.makedirs(temp_dir, exist_ok=True)

        filename = f"tts_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join(temp_dir, filename)

        with open(filepath, 'wb') as f:
            for chunk in audio_stream:
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
7. If they need to transfer you, say "Yes, please transfer me to the credentialing department"
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
- If you just got an answer to a question, ASK THE NEXT QUESTION (if there are more)
- If stage is "asking_question_X", you MUST include question X in your response
- If stage is "wrapping_up", ALL questions have already been asked. DO NOT ask any more questions. Thank them briefly for their help and use action="end_call" to end the call. Set question_asked=false.

Current stage: {stage}

=== INFORMATION EXTRACTION ===
When the representative answers a question, extract key details into extracted_info:
- "status": credentialing status (e.g., "approved", "pending", "on hold", "denied", "in review")
- "reference": any reference or case number mentioned
- "turnaround": estimated turnaround time or days mentioned
- "missing_documents": any missing documents or requirements mentioned
- "notes": any other important details
Only include fields that were actually mentioned. Do NOT leave fields empty or as "...".

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
- action="continue" - Use this for most responses (answering verification, asking questions, etc.)
- action="end_call" - Use when stage="wrapping_up". Thank them briefly and say goodbye.
- action="request_transfer" - Use when they say they need to transfer you
- question_asked=true - Set to true ONLY when your response includes one of the credentialing questions from the list AND there are still questions remaining

Example when asking a question:
{{"response": "Thank you for verifying. Now, can you tell me the current credentialing status for this provider?", "action": "continue", "extracted_info": {{}}, "question_asked": true}}

Example when providing NPI (NOT asking a question):
{{"response": "The NPI is {npi_spoken}", "action": "continue", "extracted_info": {{}}, "question_asked": false}}

Example when wrapping up (stage="wrapping_up"):
{{"response": "Thank you so much for your help today. Have a great day. Goodbye.", "action": "end_call", "extracted_info": {{}}, "question_asked": false}}"""),
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
                recent = state['transcript'][-15:]  # Last 15 exchanges
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

        # Create initial state
        initial_state: CredentialingState = {
            'insurance_name': data['insurance_name'],
            'provider_name': data['provider_name'],
            'npi': data['npi'],
            'tax_id': data['tax_id'],
            'address': data['address'],
            'insurance_phone': data['insurance_phone'],
            'questions': data['questions'],
            'questions_asked_count': 0,  # Track how many questions have been asked
            'call_id': call_id,
            'call_sid': None,
            'db_request_id': None,
            'call_state': CallState.INITIATING,
            'transcript': [],
            'conversation_history': [],
            'ivr_knowledge': [],
            'current_menu_level': 0,
            'accumulated_ivr_speech': '',
            'ivr_no_match_count': 0,
            'ivr_zero_press_count': 0,
            'ivr_total_navigate_attempts': 0,
            'pending_ivr_match': None,
            'wait_for_agent_start_time': None,
            'wait_for_agent_attempts': 0,
            'agent_detected': False,
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
            'error_message': None
        }

        # Create DB record up front so answers can be stored during the call
        try:
            from credentialing_agent import DatabaseManager
            db = DatabaseManager()
            request_id = db.save_credentialing_request(initial_state)
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

        # Create agent
        agent = CredentialingAgent()

        # Start call in background thread
        def run_call():
            print(f"🔄 Background thread started for call: {call_id}")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                print(f"📞 Initiating Twilio call to: {data['insurance_phone']}")
                final_state = loop.run_until_complete(agent.process_call(initial_state))
                loop.close()

                # Update stored state with final state
                register_call_state(call_id, final_state)
                print(f"✅ Call completed: {call_id}")
                print(f"📊 Final status: {final_state.get('call_state')}")
            except Exception as thread_error:
                print(f"❌ Error in call thread: {thread_error}")
                import traceback
                traceback.print_exc()

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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
                speech_timeout='auto',
                timeout=15,
                language='en-US',
                speech_model='phone_call'
            )
            # Don't say anything - just listen to IVR prompts
            response.append(gather)

            # If no IVR detected, go to wait-for-agent to verify human presence
            response.redirect(f'{callback_base}/webhook/wait-for-agent')
        else:
            # No IVR patterns - go straight to human conversation
            print(f"📞 Direct Human Mode - No IVR patterns configured")
            call_info['state']['call_state'] = CallState.SPEAKING_WITH_HUMAN

            # AI Disclosure message (required for automated calls)
            disclosure = f"Hello, this is an automated AI assistant calling on behalf of {provider_name} regarding provider credentialing. Is this the credentialing department?"

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

        provider_name = "a healthcare provider"
        if call_info:
            provider_name = call_info['state'].get('provider_name', provider_name)

        # AI Disclosure message
        disclosure = f"Hello, this is an automated AI assistant calling on behalf of {provider_name} regarding provider credentialing. Is this the credentialing department?"

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
            state = call_info['state']
            ivr_patterns = state.get('ivr_knowledge', [])
            current_level = state.get('current_menu_level', 0)

            # Track total IVR navigation attempts to prevent infinite loops
            total_attempts = state.get('ivr_total_navigate_attempts', 0) + 1
            state['ivr_total_navigate_attempts'] = total_attempts
            print(f"🔢 IVR total navigate attempt: {total_attempts}")

            MAX_IVR_TOTAL_ATTEMPTS = 20
            if total_attempts > MAX_IVR_TOTAL_ATTEMPTS:
                print(f"🛑 IVR navigation exceeded max attempts ({MAX_IVR_TOTAL_ATTEMPTS}). Transitioning to wait-for-agent.")
                state['call_state'] = CallState.WAITING_FOR_AGENT
                state['ivr_no_match_count'] = 0
                state['accumulated_ivr_speech'] = ''
                state['pending_ivr_match'] = None
                response.redirect(f'{callback_base}/webhook/wait-for-agent')
                return Response(str(response), mimetype='text/xml')

            # Handle pending match: if no speech and we have a pending match,
            # the Gather timed out (menu is done talking) - NOW execute the DTMF
            pending_match = state.get('pending_ivr_match')
            if pending_match and not speech_result and not digits:
                print(f"⏱️ Menu finished (silence detected). Executing pending match: {pending_match.get('detected_phrase', '')} -> {pending_match.get('action_value', '')}")
                action = pending_match.get('preferred_action', 'dtmf')
                action_value = pending_match.get('action_value', '')

                if action == 'dtmf' and action_value:
                    response.play(digits=action_value)
                    print(f"📲 Sending DTMF: {action_value}")
                elif action == 'speech' and action_value:
                    speak_with_tts(response, action_value)
                    print(f"🗣️ Saying: {action_value}")

                state['pending_ivr_match'] = None
                state['current_menu_level'] = pending_match['menu_level']
                state['accumulated_ivr_speech'] = ''
                state['ivr_no_match_count'] = 0
                state['ivr_zero_press_count'] = 0

                # Check if we have more IVR levels to navigate
                remaining_patterns = [p for p in ivr_patterns if p['menu_level'] > state['current_menu_level']]

                if remaining_patterns:
                    response.pause(length=4)
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='auto',
                        timeout=15,
                        language='en-US',
                        speech_model='phone_call'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/wait-for-agent')
                else:
                    print(f"✅ IVR navigation complete - entering wait-for-agent loop")
                    response.pause(length=2)
                    response.redirect(f'{callback_base}/webhook/wait-for-agent')

                return Response(str(response), mimetype='text/xml')

            # If we have a pending match but more speech came in, menu is still talking
            # Clear pending match and re-evaluate with new accumulated speech
            if pending_match and speech_result:
                print(f"📢 Menu still talking (got more speech). Keeping pending match and continuing to listen.")

            # Accumulate speech across gather cycles for full IVR prompt matching
            accumulated = state.get('accumulated_ivr_speech', '')
            accumulated = (accumulated + ' ' + speech_result).strip()
            state['accumulated_ivr_speech'] = accumulated
            accumulated_lower = accumulated.lower()
            speech_lower = speech_result.lower()

            print(f"📝 Accumulated IVR speech: {accumulated}")

            # Check current speech for IVR pattern match first (fast path),
            # then fall back to accumulated speech (handles split captures)
            matched_pattern = None

            for pattern in ivr_patterns:
                if pattern['menu_level'] > current_level:
                    detected_phrase = pattern.get('detected_phrase', '').lower()
                    if detected_phrase and detected_phrase in speech_lower:
                        matched_pattern = pattern
                        print(f"✅ Matched IVR pattern (current speech): '{detected_phrase}' at level {pattern['menu_level']}")
                        break

            if not matched_pattern:
                for pattern in ivr_patterns:
                    if pattern['menu_level'] > current_level:
                        detected_phrase = pattern.get('detected_phrase', '').lower()
                        if detected_phrase and detected_phrase in accumulated_lower:
                            matched_pattern = pattern
                            print(f"✅ Matched IVR pattern (accumulated): '{detected_phrase}' at level {pattern['menu_level']}")
                            break

            if matched_pattern:
                # Store match as pending - wait for menu to finish before pressing
                print(f"🎯 Pattern matched: '{matched_pattern.get('detected_phrase', '')}' -> {matched_pattern.get('preferred_action', 'dtmf')} {matched_pattern.get('action_value', '')}")
                print(f"⏳ Storing as pending match - waiting for menu to finish before pressing")
                state['pending_ivr_match'] = matched_pattern
                state['accumulated_ivr_speech'] = ''
                state['ivr_no_match_count'] = 0

                # Keep listening - when Gather times out (menu done), redirect back here
                # which will detect pending_match + no speech = execute DTMF
                gather = Gather(
                    input='speech dtmf',
                    action=f'{callback_base}/webhook/ivr-navigate',
                    method='POST',
                    speech_timeout='auto',
                    timeout=15,
                    language='en-US',
                    speech_model='phone_call'
                )
                response.append(gather)
                # On Gather timeout (silence = menu done), redirect back to ivr-navigate
                response.redirect(f'{callback_base}/webhook/ivr-navigate')
            elif pending_match:
                # We have a pending match and menu is still talking (speech came in)
                # Keep the pending match and continue listening for silence
                print(f"⏳ Pending match exists, menu still talking. Continuing to wait for silence.")
                gather = Gather(
                    input='speech dtmf',
                    action=f'{callback_base}/webhook/ivr-navigate',
                    method='POST',
                    speech_timeout='auto',
                    timeout=15,
                    language='en-US',
                    speech_model='phone_call'
                )
                response.append(gather)
                response.redirect(f'{callback_base}/webhook/ivr-navigate')
            else:
                # No match - might be human or different IVR prompt
                # Use strict human detection (exclude IVR-like speech)
                human_indicators = [
                    'how can i help', 'how may i help', 'how can i assist',
                    'what can i do for you', 'who am i speaking with',
                    'go ahead', 'may i have your name', 'can i have your name',
                    'what is your name', 'name please'
                ]
                ivr_like_indicators = ['press', 'option', 'menu', 'para espanol', 'for english', 'dial']
                is_human = (
                    any(ind in speech_lower for ind in human_indicators) and
                    not any(ind in speech_lower for ind in ivr_like_indicators)
                )

                if is_human or len(ivr_patterns) == 0:
                    # Likely human - route through wait-for-agent to confirm
                    print(f"👤 Possible human detected during IVR - verifying via wait-for-agent")
                    state['call_state'] = CallState.WAITING_FOR_AGENT
                    state['pending_ivr_match'] = None
                    response.redirect(f'{callback_base}/webhook/wait-for-agent')
                else:
                    # Keep listening for IVR - track no-match count
                    no_match_count = state.get('ivr_no_match_count', 0) + 1
                    state['ivr_no_match_count'] = no_match_count
                    print(f"🔄 No IVR match (attempt {no_match_count}) - continuing to listen")

                    # Detect if the IVR is still playing its preamble/menu
                    ivr_preamble_indicators = [
                        'press', 'option', 'menu', 'para espanol', 'for english',
                        'dial', 'please listen', 'options have changed',
                        'to repeat', 'if you know your party', 'extension',
                        'for billing', 'for claims', 'for provider', 'for member',
                        'for pharmacy', 'for eligibility', 'for authorization',
                        'please make a selection', 'please select',
                    ]
                    speech_has_ivr_cues = any(ind in speech_lower for ind in ivr_preamble_indicators)
                    accumulated_has_ivr_cues = any(ind in accumulated_lower for ind in ivr_preamble_indicators)

                    if speech_has_ivr_cues or accumulated_has_ivr_cues:
                        # IVR menu is still speaking - reset no-match count and keep listening
                        state['ivr_no_match_count'] = 0
                        print(f"📢 IVR menu speech detected, resetting no-match count. Waiting for full menu.")
                    elif no_match_count >= 5:
                        # No IVR cues and enough failed attempts - try pressing 0
                        zero_press_count = state.get('ivr_zero_press_count', 0) + 1
                        state['ivr_zero_press_count'] = zero_press_count
                        MAX_ZERO_PRESSES = 2

                        if zero_press_count <= MAX_ZERO_PRESSES:
                            print(f"⚠️ IVR no-match limit reached ({no_match_count}), pressing 0 for operator (attempt {zero_press_count}/{MAX_ZERO_PRESSES})")
                            response.play(digits='0')
                            state['ivr_no_match_count'] = 0
                            state['accumulated_ivr_speech'] = ''
                            response.pause(length=4)
                        else:
                            # Already pressed 0 max times with no success - give up on IVR
                            print(f"🛑 Pressed 0 {MAX_ZERO_PRESSES} times with no success. Giving up on IVR, transitioning to wait-for-agent.")
                            state['call_state'] = CallState.WAITING_FOR_AGENT
                            state['ivr_no_match_count'] = 0
                            state['accumulated_ivr_speech'] = ''
                            response.redirect(f'{callback_base}/webhook/wait-for-agent')
                            return Response(str(response), mimetype='text/xml')

                    response.pause(length=1)
                    gather = Gather(
                        input='speech dtmf',
                        action=f'{callback_base}/webhook/ivr-navigate',
                        method='POST',
                        speech_timeout='auto',
                        timeout=15,
                        language='en-US',
                        speech_model='phone_call'
                    )
                    response.append(gather)
                    response.redirect(f'{callback_base}/webhook/wait-for-agent')
        else:
            # No call state found - go to wait-for-agent
            response.redirect(f'{callback_base}/webhook/wait-for-agent')

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in ivr_navigate_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        speak_with_tts(response, "Sorry, there was a technical error. Goodbye.")
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/wait-for-agent', methods=['POST'])
def wait_for_agent_webhook():
    """
    Wait-for-agent loop: listens for human speech indicators before
    starting the AI disclosure. Loops indefinitely until a human answers
    or the call is cancelled. Payors can have 2+ hour hold times.
    """
    try:
        from twilio.twiml.voice_response import Gather

        response = VoiceResponse()
        call_sid = request.values.get('CallSid')
        speech_result = request.values.get('SpeechResult', '')

        callback_base = get_callback_base(request)

        # Find call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                break

        if not call_info:
            # No state found, go to human mode as fallback
            response.redirect(f'{callback_base}/webhook/voice/human')
            return Response(str(response), mimetype='text/xml')

        state = call_info['state']

        # Initialize wait tracking on first entry
        if not state.get('wait_for_agent_start_time'):
            state['wait_for_agent_start_time'] = time.time()
            state['wait_for_agent_attempts'] = 0
            state['call_state'] = CallState.WAITING_FOR_AGENT
            print(f"⏳ Started waiting for agent - CallSID: {call_sid}")

        state['wait_for_agent_attempts'] = state.get('wait_for_agent_attempts', 0) + 1
        elapsed = time.time() - state['wait_for_agent_start_time']
        attempt = state['wait_for_agent_attempts']

        print(f"⏳ Wait-for-agent attempt {attempt} (elapsed: {elapsed:.0f}s) - CallSID: {call_sid}")

        # Analyze speech if present
        if speech_result and speech_result.strip():
            speech_lower = speech_result.lower().strip()
            print(f"🎤 Wait-for-agent speech: '{speech_result}' (elapsed: {elapsed:.0f}s)")

            # Hold/automated message indicators - keep waiting
            hold_indicators = [
                'please hold', 'please stay on the line', 'please remain on the line',
                'your call is important', 'estimated wait', 'approximate wait',
                'all representatives are busy', 'all agents are busy',
                'all of our representatives', 'all of our agents',
                'please continue to hold', 'next available',
                'in the order received', 'in the order it was received',
                'for quality assurance', 'this call may be recorded',
                'this call may be monitored', 'for training purposes',
                'thank you for your patience', 'we appreciate your patience',
                'thank you for holding', 'thank you for waiting',
                'please wait', 'one moment please',
                'your call will be answered', 'remain on the line',
                'currently experiencing high call volume',
                'higher than normal call volume',
                'press', 'for english', 'para espanol',
                'menu', 'option', 'to repeat these options',
                'if you know your party',
            ]

            # Human greeting/conversational indicators - agent answered
            human_indicators = [
                'hello', 'hi', 'hey',
                'good morning', 'good afternoon', 'good evening',
                'how can i help', 'how may i help', 'how can i assist',
                'how may i assist', 'what can i do for you',
                'speaking', 'this is', 'my name is',
                'department', 'credentialing',
                'thank you for calling',
                'who am i speaking with', 'who am i talking to',
                'may i have your', 'can i have your',
                'what is your', 'name please',
                'how are you', 'go ahead',
                'yes', 'provider services',
                'can i get your', 'do you have your',
            ]

            is_hold = any(ind in speech_lower for ind in hold_indicators)
            is_human = any(ind in speech_lower for ind in human_indicators)

            if is_hold and not is_human:
                # Definitely hold/automated message - keep waiting
                print(f"📻 Hold message detected, continuing to wait... ('{speech_result[:60]}')")
                # Fall through to re-gather below

            elif is_human and not is_hold:
                # Definitely human - proceed to conversation
                print(f"👤 Human agent confirmed! Transitioning to conversation. ('{speech_result[:60]}')")
                state['agent_detected'] = True
                state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                response.redirect(f'{callback_base}/webhook/voice/human')
                return Response(str(response), mimetype='text/xml')

            elif is_human and is_hold:
                # Matches both - "thank you for calling" can be either IVR or human
                # Use word count heuristic: short = likely human, long = likely hold message
                word_count = len(speech_lower.split())
                if word_count <= 8:
                    print(f"👤 Short mixed speech, treating as human. ('{speech_result[:60]}')")
                    state['agent_detected'] = True
                    state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                    response.redirect(f'{callback_base}/webhook/voice/human')
                    return Response(str(response), mimetype='text/xml')
                else:
                    print(f"📻 Long mixed speech, treating as hold message. ('{speech_result[:60]}')")
                    # Fall through to re-gather

            else:
                # No indicators matched - ambiguous speech
                word_count = len(speech_lower.split())
                if word_count <= 5:
                    # Short ambiguous utterance - likely human
                    print(f"👤 Short ambiguous speech, treating as human. ('{speech_result[:60]}')")
                    state['agent_detected'] = True
                    state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                    response.redirect(f'{callback_base}/webhook/voice/human')
                    return Response(str(response), mimetype='text/xml')
                else:
                    print(f"❓ Ambiguous speech, continuing to wait. ('{speech_result[:60]}')")
                    # Fall through to re-gather

        # Continue waiting - set up another gather cycle
        gather = Gather(
            input='speech',
            action=f'{callback_base}/webhook/wait-for-agent',
            method='POST',
            speech_timeout='3',
            timeout=15,
            language='en-US',
            speech_model='phone_call'
        )
        response.append(gather)

        # If gather times out (silence), loop back to keep waiting
        response.redirect(f'{callback_base}/webhook/wait-for-agent')

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in wait_for_agent_webhook: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to human mode on error
        response = VoiceResponse()
        callback_base = get_callback_base(request)
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

        print(f"🎤 Speech received - CallSID: {call_sid}")
        print(f"📝 Transcript: {speech_result}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                # Add to transcript
                state['transcript'].append({
                    'speaker': 'insurance',
                    'text': speech_result,
                    'timestamp': datetime.now().isoformat()
                })
                break

        # Use GPT-4 to generate intelligent response
        if call_info:
            state = call_info['state']

            # Track questions asked using a dedicated counter (not agent message count)
            questions = state.get('questions', [])
            questions_asked_count = state.get('questions_asked_count', 0)

            # Determine the stage based on actual questions asked, not total messages
            if questions_asked_count >= len(questions):
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

            print(f"🤖 AI Response: {ai_response}")
            print(f"📋 Stage: {stage}, Questions Asked: {questions_asked_count}/{len(questions)}")

            # Update questions asked count if a question was asked (cap at total)
            if ai_response.get('question_asked') and questions_asked_count < len(questions):
                state['questions_asked_count'] = questions_asked_count + 1
                print(f"✅ Question asked! New count: {state['questions_asked_count']}")

            # Add AI response to transcript
            state['transcript'].append({
                'speaker': 'agent',
                'text': ai_response.get('response', ''),
                'timestamp': datetime.now().isoformat()
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

            # Persist conversation + progress in a single DB connection
            try:
                from credentialing_agent import DatabaseManager
                db = DatabaseManager()
                db.save_conversation(call_info['call_id'], 'representative', speech_result)
                db.save_conversation(call_info['call_id'], 'agent', ai_response.get('response', ''))
                request_id = state.get('db_request_id') or state.get('call_id')
                if request_id:
                    if finalize:
                        db.save_final_results(request_id, state)
                    elif ai_response.get('extracted_info'):
                        db.update_call_progress(request_id, state)
                db.close()
            except Exception as db_err:
                print(f"WARN: Could not persist conversation/progress: {db_err}")

            if action == 'end_call':
                speak_with_tts(response, ai_response.get('response', 'Thank you for your time. Goodbye.'))
                # Give the other party a short window (8s) in case they need to interject before we disconnect
                response.pause(length=8)
                response.hangup()
            elif action == 'request_transfer':
                speak_with_tts(response, ai_response.get('response', 'Could you please transfer me to the credentialing department?'))
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
                # Fallback if no response after transfer
                response.redirect(f'{callback_base}/webhook/speech')
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

        print(f"🎤 Follow-up speech - CallSID: {call_sid}")
        print(f"📝 Transcript: {speech_result}")

        # Find the call state
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid:
                call_info = {'call_id': call_id, 'state': state}
                # Add to transcript
                state['transcript'].append({
                    'speaker': 'insurance',
                    'text': speech_result,
                    'timestamp': datetime.now().isoformat()
                })
                break

        if call_info:
            state = call_info['state']

            # Track questions asked using a dedicated counter
            questions = state.get('questions', [])
            questions_asked_count = state.get('questions_asked_count', 0)

            # Determine stage - ask questions if we have more to ask
            if questions_asked_count >= len(questions):
                stage = "wrapping_up"
            else:
                stage = f"asking_question_{questions_asked_count + 1}_of_{len(questions)}"

            print(f"📋 Follow-up stage: {stage}, Questions Asked: {questions_asked_count}/{len(questions)}")

            # Use GPT-4 to generate intelligent response
            ai_response = smart_agent.generate_response(
                speech=speech_result,
                state=state,
                stage=stage,
                current_question_index=questions_asked_count
            )

            print(f"🤖 AI Response: {ai_response}")

            # Update questions asked count if a question was asked (cap at total)
            if ai_response.get('question_asked') and questions_asked_count < len(questions):
                state['questions_asked_count'] = questions_asked_count + 1
                print(f"✅ Question asked! New count: {state['questions_asked_count']}")

            # Add AI response to transcript
            state['transcript'].append({
                'speaker': 'agent',
                'text': ai_response.get('response', ''),
                'timestamp': datetime.now().isoformat()
            })

            # Store any extracted info
            if ai_response.get('extracted_info'):
                for key, value in ai_response['extracted_info'].items():
                    if value:
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

            if finalize:
                state['call_state'] = CallState.COMPLETING

            # Persist conversation + progress in a single DB connection
            try:
                from credentialing_agent import DatabaseManager
                db = DatabaseManager()
                db.save_conversation(call_info['call_id'], 'representative', speech_result)
                db.save_conversation(call_info['call_id'], 'agent', ai_response.get('response', ''))
                request_id = state.get('db_request_id') or state.get('call_id')
                if request_id:
                    if finalize:
                        db.save_final_results(request_id, state)
                    elif ai_response.get('extracted_info'):
                        db.update_call_progress(request_id, state)
                db.close()
            except Exception as db_err:
                print(f"WARN: Could not persist conversation/progress: {db_err}")

            if action == 'end_call':
                speak_with_tts(response, ai_response.get('response', 'Thank you very much for your help. Have a great day. Goodbye.'))
                # Wait 8 seconds after the AI stops talking before ending the call
                response.pause(length=8)
                response.hangup()
            elif action == 'request_transfer':
                speak_with_tts(response, ai_response.get('response', 'Could you please transfer me?'))
                response.pause(length=30)
                gather = Gather(
                    input='speech',
                    action=f'{callback_base}/webhook/speech/followup',
                    method='POST',
                    speech_timeout='auto',
                    language='en-US'
                )
                speak_with_tts(response, "Hello? Is anyone there?", gather=gather)
                response.append(gather)
                # Fallback if no response after transfer
                response.redirect(f'{callback_base}/webhook/speech/followup')
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
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence
            })
            
            # Trigger state graph to process new transcript
            # This would involve resuming the agent's state machine
            # Implementation depends on your LangGraph checkpoint strategy
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        print(f"Transcription webhook error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/call-status/<call_id>', methods=['GET'])
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


@app.route('/api/call-detail/<call_id>', methods=['GET'])
def get_call_detail(call_id: str):
    """
    Get full call details from the database (persists across server restarts).
    Combines credentialing request data with conversation history and call events.
    """
    try:
        from credentialing_agent import DatabaseManager
        from psycopg2.extras import RealDictCursor

        db = DatabaseManager()

        # Fetch credentialing request
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, insurance_name, provider_name, npi, tax_id,
                       address, insurance_phone, questions, status, reference_number,
                       missing_documents, turnaround_days, notes,
                       created_at, updated_at, completed_at
                FROM credentialing_requests
                WHERE id = %s
            """, (call_id,))
            row = cur.fetchone()

        if not row:
            db.close()
            return jsonify({'success': False, 'error': 'Call not found'}), 404

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
            'created_at': row[13].isoformat() if row[13] else None,
            'updated_at': row[14].isoformat() if row[14] else None,
            'completed_at': row[15].isoformat() if row[15] else None,
        }

        # Fetch conversation history
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT speaker, message, timestamp
                FROM conversation_history
                WHERE call_id = %s OR request_id = %s
                ORDER BY timestamp ASC
            """, (call_id, call_id))
            conversations = []
            for r in cur.fetchall():
                conversations.append({
                    'speaker': r[0],
                    'message': r[1],
                    'timestamp': r[2].isoformat() if r[2] else None,
                })

        # Fetch call events
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT event_type, transcript, action_taken, confidence, timestamp, metadata
                FROM call_events
                WHERE call_id = %s
                ORDER BY timestamp ASC
            """, (call_id,))
            events = []
            for r in cur.fetchall():
                events.append({
                    'event_type': r[0],
                    'transcript': r[1],
                    'action_taken': r[2],
                    'confidence': r[3],
                    'timestamp': r[4].isoformat() if r[4] else None,
                    'metadata': r[5] or {},
                })

        db.close()

        return jsonify({
            'success': True,
            'data': {
                **call_data,
                'conversation': conversations,
                'events': events,
            }
        }), 200

    except Exception as e:
        print(f"Error fetching call detail: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/metrics', methods=['GET'])
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
        'timestamp': datetime.now().isoformat(),
        'active_calls': len(active_calls)
    }), 200


@app.route('/api/ivr-knowledge', methods=['POST'])
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
        from credentialing_agent import DatabaseManager
        
        data = request.json
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
                data['preferred_action'],
                data.get('action_value')
            ))
            
            ivr_id = cur.fetchone()[0]
            db.conn.commit()
        
        db.close()
        
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
def get_recent_calls():
    """
    Get recent credentialing calls
    """
    try:
        from credentialing_agent import DatabaseManager

        limit = request.args.get('limit', 20, type=int)
        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, insurance_name, provider_name, npi, tax_id,
                       address, insurance_phone, status, reference_number,
                       missing_documents, turnaround_days, notes,
                       created_at, completed_at
                FROM credentialing_requests
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

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
                    'created_at': row[12].isoformat() if row[12] else None,
                    'completed_at': row[13].isoformat() if row[13] else None
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
                    'scheduled_date': row[2].isoformat() if row[2] else None,
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
                knowledge.append({
                    'id': str(row[0]),
                    'insurance_name': row[1],
                    'menu_level': row[2],
                    'detected_phrase': row[3],
                    'preferred_action': row[4],
                    'action_value': row[5],
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
                       best_call_times, average_wait_time_minutes, notes, last_updated
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
                    'last_updated': row[7].isoformat() if row[7] else None
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
    try:
        from credentialing_agent import DatabaseManager

        data = request.json
        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO insurance_providers
                (insurance_name, phone_number, department, best_call_times,
                 average_wait_time_minutes, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                data['insurance_name'],
                data['phone_number'],
                data.get('department'),
                json.dumps(data.get('best_call_times')) if data.get('best_call_times') else None,
                data.get('average_wait_time_minutes'),
                data.get('notes')
            ))

            provider_id = cur.fetchone()[0]
            db.conn.commit()

        db.close()

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


@app.route('/api/insurance-providers/<provider_id>', methods=['PUT'])
def update_insurance_provider(provider_id: str):
    """
    Update an existing insurance provider
    """
    try:
        from credentialing_agent import DatabaseManager

        data = request.json
        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                UPDATE insurance_providers
                SET insurance_name = %s,
                    phone_number = %s,
                    department = %s,
                    best_call_times = %s,
                    average_wait_time_minutes = %s,
                    notes = %s,
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
                provider_id
            ))

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
            'message': 'Insurance provider updated successfully'
        }), 200

    except Exception as e:
        print(f"Error updating insurance provider: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/insurance-providers/<provider_id>', methods=['DELETE'])
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
def delete_ivr_knowledge(ivr_id: str):
    """
    Delete an IVR knowledge entry
    """
    try:
        from credentialing_agent import DatabaseManager

        db = DatabaseManager()

        with db.conn.cursor() as cur:
            cur.execute("""
                DELETE FROM ivr_knowledge
                WHERE id = %s
                RETURNING id
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
def update_ivr_knowledge(ivr_id: str):
    """
    Update an IVR knowledge entry
    """
    try:
        from credentialing_agent import DatabaseManager

        data = request.json
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
                RETURNING id
            """, (
                data['menu_level'],
                data['detected_phrase'],
                data['preferred_action'],
                data.get('action_value'),
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
            call_sid = data.get('start', {}).get('callSid')
            print(f"🎙️ Stream started - StreamSID: {stream_sid}, CallSID: {call_sid}")

            # Store stream info
            media_streams[request.sid] = {
                'stream_sid': stream_sid,
                'call_sid': call_sid,
                'audio_buffer': [],
                'transcript': ''
            }

            # Update call state if we have it
            for call_id, state in call_states.items():
                if state.get('call_sid') == call_sid:
                    state['current_audio_type'] = AudioType.HUMAN_SPEECH
                    print(f"📊 Updated call {call_id} audio type to HUMAN_SPEECH")
                    break

        elif event_type == 'media':
            # Audio payload - base64 encoded mu-law audio
            payload = data.get('media', {}).get('payload', '')
            if payload and request.sid in media_streams:
                # Decode audio (for future Deepgram integration)
                # audio_data = base64.b64decode(payload)
                # For now, just acknowledge receipt
                pass

        elif event_type == 'stop':
            stream_sid = data.get('streamSid')
            print(f"🛑 Stream stopped: {stream_sid}")

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
