"""
Autonomous AI Insurance Credentialing Voice Agent
Python Implementation with LangChain, LangSmith, and LangGraph

This module provides the core autonomous agent for handling insurance credentialing calls.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal, TypedDict, Annotated, Any
from enum import Enum

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangGraph imports for state management
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add

# LangSmith for tracing
from langsmith import Client, traceable
from langsmith.run_helpers import trace

# Telephony and Audio
from twilio.rest import Client as TwilioClient
from twilio.twiml.voice_response import VoiceResponse, Gather
from deepgram import DeepgramClient
from elevenlabs.client import ElevenLabs

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

# Environment
from dotenv import load_dotenv

load_dotenv()

# Local knowledge base (pgvector)
from knowledge_base import KnowledgeBase, redact_text


class CallState(str, Enum):
    """Enumeration of possible call states"""
    INITIATING = "initiating"
    IVR_NAVIGATION = "ivr_navigation"
    ON_HOLD = "on_hold"
    SPEAKING_WITH_HUMAN = "speaking_with_human"
    EXTRACTING_INFO = "extracting_info"
    COMPLETING = "completing"
    FAILED = "failed"


class AudioType(str, Enum):
    """Types of audio detected during call"""
    IVR_MENU = "ivr_menu"
    HOLD_MUSIC = "hold_music"
    HUMAN_SPEECH = "human_speech"
    SILENCE = "silence"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Types of actions the agent can take"""
    DTMF = "dtmf"  # Press digits
    SPEECH = "speech"  # Say something
    WAIT = "wait"  # Wait/listen
    HANGUP = "hangup"


class CredentialingState(TypedDict):
    """State graph for the credentialing call"""
    # Request information
    insurance_name: str
    provider_name: str
    npi: str
    tax_id: str
    address: str
    insurance_phone: str
    questions: List[str]
    
    # Call tracking
    call_id: Optional[str]
    call_sid: Optional[str]
    db_request_id: Optional[str]
    call_state: CallState
    
    # Real-time conversation
    transcript: Annotated[List[Dict], add]  # Running transcript
    conversation_history: Annotated[List[Dict], add]  # Full history
    
    # IVR knowledge
    ivr_knowledge: List[Dict]
    current_menu_level: int
    
    # Audio classification
    current_audio_type: AudioType
    confidence: float
    
    # Decision making
    last_action: Optional[Dict]
    retry_count: int
    
    # Results
    credentialing_status: Optional[str]
    reference_number: Optional[str]
    missing_documents: List[str]
    turnaround_days: Optional[int]
    notes: str
    
    # Control flow
    should_continue: bool
    error_message: Optional[str]

    # Timing metrics (optional)
    call_start_time: Optional[datetime]
    ivr_end_time: Optional[datetime]
    hold_end_time: Optional[datetime]
    human_start_time: Optional[datetime]


class LangSmithConfig:
    """Configuration for LangSmith tracing"""
    def __init__(self):
        self.client = Client(
            api_key=os.getenv("LANGSMITH_API_KEY"),
            api_url=os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
        )
        self.project_name = os.getenv("LANGSMITH_PROJECT", "insurance-credentialing-agent")
        
    def create_run(self, name: str, inputs: Dict, run_type: str = "chain"):
        """Create a new run in LangSmith"""
        return self.client.create_run(
            name=name,
            inputs=inputs,
            run_type=run_type,
            project_name=self.project_name
        )


class DatabaseManager:
    """Manages all database operations"""
    
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("SUPABASE_HOST"),
            database="postgres",
            user=os.getenv("SUPABASE_USER", "postgres"),
            password=os.getenv("SUPABASE_PASSWORD"),
            port=5432
        )
    
    def save_credentialing_request(self, state: CredentialingState) -> str:
        """Save initial credentialing request"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO credentialing_requests 
                (insurance_name, provider_name, npi, tax_id, address, insurance_phone, questions, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                state['insurance_name'],
                state['provider_name'],
                state['npi'],
                state['tax_id'],
                state['address'],
                state.get('insurance_phone'),
                json.dumps(state['questions']),
                'initiated'
            ))
            request_id = cur.fetchone()[0]
            self.conn.commit()
            return str(request_id)
    
    def get_ivr_knowledge(self, insurance_name: str) -> List[Dict]:
        """Retrieve IVR knowledge for an insurance provider"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM ivr_knowledge 
                WHERE insurance_name ILIKE %s
                ORDER BY success_rate DESC, menu_level ASC
            """, (f"%{insurance_name}%",))
            return [dict(row) for row in cur.fetchall()]
    
    def log_call_event(self, call_id: str, event_type: str, data: Dict):
        """Log a call event"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO call_events 
                (call_id, event_type, transcript, action_taken, confidence, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                call_id,
                event_type,
                data.get('transcript'),
                data.get('action'),
                data.get('confidence'),
                json.dumps(data.get('metadata', {}))
            ))
            self.conn.commit()
    
    def save_conversation(self, call_id: str, speaker: str, message: str):
        """Save conversation message"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_history (call_id, speaker, message)
                VALUES (%s, %s, %s)
            """, (call_id, speaker, message))
            self.conn.commit()
    
    def update_ivr_knowledge(self, ivr_id: str, success: bool):
        """Update IVR knowledge success rate"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE ivr_knowledge 
                SET attempts = attempts + 1,
                    successes = successes + CASE WHEN %s THEN 1 ELSE 0 END,
                    success_rate = (successes::float + CASE WHEN %s THEN 1 ELSE 0 END) / (attempts + 1),
                    last_updated = NOW()
                WHERE id = %s
            """, (success, success, ivr_id))
            self.conn.commit()
    
    def save_final_results(self, request_id: str, state: CredentialingState):
        """Save final credentialing results"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE credentialing_requests 
                SET status = %s,
                    reference_number = %s,
                    missing_documents = %s,
                    turnaround_days = %s,
                    notes = %s,
                    completed_at = NOW()
                WHERE id = %s
            """, (
                state.get('credentialing_status'),
                state.get('reference_number'),
                json.dumps(state.get('missing_documents', [])),
                state.get('turnaround_days'),
                state.get('notes'),
                request_id
            ))
            self.conn.commit()
    
    def update_call_progress(self, request_id: str, state: CredentialingState):
        """Persist in-progress answers so they can be viewed during the call"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE credentialing_requests 
                SET status = COALESCE(%s, status),
                    reference_number = COALESCE(%s, reference_number),
                    missing_documents = CASE WHEN %s IS NOT NULL THEN %s ELSE missing_documents END,
                    turnaround_days = COALESCE(%s, turnaround_days),
                    notes = COALESCE(%s, notes),
                    updated_at = NOW()
                WHERE id = %s
            """, (
                state.get('credentialing_status'),
                state.get('reference_number'),
                json.dumps(state.get('missing_documents', [])) if state.get('missing_documents') else None,
                json.dumps(state.get('missing_documents', [])) if state.get('missing_documents') else None,
                state.get('turnaround_days'),
                state.get('notes'),
                request_id
            ))
            self.conn.commit()
    
    # =========================================================================
    # Call Metrics Methods
    # =========================================================================

    def insert_call_metrics(
        self,
        call_id: str,
        request_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        ivr_navigation_time_seconds: Optional[int] = None,
        hold_time_seconds: Optional[int] = None,
        human_interaction_time_seconds: Optional[int] = None,
        successful: Optional[bool] = None,
        failure_reason: Optional[str] = None,
        retry_count: int = 0,
    ) -> str:
        """Insert or update call metrics record."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO call_metrics
                (call_id, request_id, duration_seconds, ivr_navigation_time_seconds,
                 hold_time_seconds, human_interaction_time_seconds, successful,
                 failure_reason, retry_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (call_id) DO UPDATE SET
                    duration_seconds = COALESCE(EXCLUDED.duration_seconds, call_metrics.duration_seconds),
                    ivr_navigation_time_seconds = COALESCE(EXCLUDED.ivr_navigation_time_seconds, call_metrics.ivr_navigation_time_seconds),
                    hold_time_seconds = COALESCE(EXCLUDED.hold_time_seconds, call_metrics.hold_time_seconds),
                    human_interaction_time_seconds = COALESCE(EXCLUDED.human_interaction_time_seconds, call_metrics.human_interaction_time_seconds),
                    successful = COALESCE(EXCLUDED.successful, call_metrics.successful),
                    failure_reason = COALESCE(EXCLUDED.failure_reason, call_metrics.failure_reason),
                    retry_count = EXCLUDED.retry_count
                RETURNING id
                """,
                (call_id, request_id, duration_seconds, ivr_navigation_time_seconds,
                 hold_time_seconds, human_interaction_time_seconds, successful,
                 failure_reason, retry_count)
            )
            result = cur.fetchone()
            self.conn.commit()
            return str(result[0])

    def update_call_metrics(self, call_id: str, **kwargs) -> bool:
        """Update specific fields of call metrics."""
        allowed_fields = {
            'duration_seconds', 'ivr_navigation_time_seconds', 'hold_time_seconds',
            'human_interaction_time_seconds', 'successful', 'failure_reason', 'retry_count'
        }

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields and v is not None}
        if not updates:
            return False

        set_clause = ", ".join([f"{k} = %s" for k in updates.keys()])
        values = list(updates.values()) + [call_id]

        with self.conn.cursor() as cur:
            cur.execute(
                f"UPDATE call_metrics SET {set_clause} WHERE call_id = %s",
                values
            )
            self.conn.commit()
            return cur.rowcount > 0

    def get_call_metrics(self, call_id: str) -> Optional[Dict]:
        """Retrieve call metrics by call_id."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM call_metrics WHERE call_id = %s",
                (call_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_all_call_metrics(self, limit: int = 50, successful_only: Optional[bool] = None) -> List[Dict]:
        """Get call metrics with optional filtering."""
        query = "SELECT * FROM call_metrics"
        params = []

        if successful_only is not None:
            query += " WHERE successful = %s"
            params.append(successful_only)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    # =========================================================================
    # System Config Methods
    # =========================================================================

    def get_config(self, config_key: str) -> Optional[Any]:
        """Get a configuration value by key."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT config_value FROM system_config WHERE config_key = %s",
                (config_key,)
            )
            row = cur.fetchone()
            if row:
                # config_value is JSONB, psycopg2 returns it as dict/list/primitive
                return row[0]
            return None

    def get_config_with_default(self, config_key: str, default: Any) -> Any:
        """Get config value with fallback default."""
        value = self.get_config(config_key)
        return value if value is not None else default

    def set_config(self, config_key: str, config_value: Any, description: Optional[str] = None) -> bool:
        """Set or update a configuration value."""
        with self.conn.cursor() as cur:
            if description:
                cur.execute(
                    """
                    INSERT INTO system_config (config_key, config_value, description, last_updated)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (config_key) DO UPDATE SET
                        config_value = EXCLUDED.config_value,
                        description = COALESCE(EXCLUDED.description, system_config.description),
                        last_updated = NOW()
                    """,
                    (config_key, json.dumps(config_value), description)
                )
            else:
                cur.execute(
                    """
                    INSERT INTO system_config (config_key, config_value, last_updated)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (config_key) DO UPDATE SET
                        config_value = EXCLUDED.config_value,
                        last_updated = NOW()
                    """,
                    (config_key, json.dumps(config_value))
                )
            self.conn.commit()
            return True

    def get_all_configs(self) -> List[Dict]:
        """Get all configuration values."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM system_config ORDER BY config_key")
            return [dict(row) for row in cur.fetchall()]

    def delete_config(self, config_key: str) -> bool:
        """Delete a configuration value."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM system_config WHERE config_key = %s", (config_key,))
            self.conn.commit()
            return cur.rowcount > 0

    # =========================================================================
    # Audit Log Methods
    # =========================================================================

    def log_audit(
        self,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> str:
        """Manually log an audit event."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit_log
                (user_id, action, resource_type, resource_id, details, ip_address)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(details) if details else None,
                    ip_address,
                )
            )
            result = cur.fetchone()
            self.conn.commit()
            return str(result[0])

    def get_audit_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple:
        """
        Query audit logs with filtering and pagination.
        Returns (logs, total_count).
        """
        conditions = []
        params = []

        if user_id:
            conditions.append("user_id = %s")
            params.append(user_id)
        if action:
            conditions.append("action = %s")
            params.append(action)
        if resource_type:
            conditions.append("resource_type = %s")
            params.append(resource_type)
        if resource_id:
            conditions.append("resource_id = %s")
            params.append(resource_id)
        if start_date:
            conditions.append("timestamp >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("timestamp <= %s")
            params.append(end_date)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM audit_log WHERE {where_clause}", params)
            total_count = cur.fetchone()[0]

        # Get paginated results
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = f"""
                SELECT * FROM audit_log
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
            """
            cur.execute(query, params + [limit, offset])
            logs = [dict(row) for row in cur.fetchall()]

        return logs, total_count

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class ConfigManager:
    """Centralized configuration management with caching."""

    _instance = None
    _cache = {}
    _cache_ttl = 300  # 5 minutes
    _cache_timestamps = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._db = None
        self._initialized = True

    def _get_db(self):
        """Get or create database connection."""
        if self._db is None:
            self._db = DatabaseManager()
        return self._db

    def get(self, key: str, default: Any = None) -> Any:
        """Get config with caching."""
        now = datetime.now().timestamp()

        # Check cache
        if key in self._cache:
            if now - self._cache_timestamps.get(key, 0) < self._cache_ttl:
                return self._cache[key]

        # Fetch from DB
        try:
            db = self._get_db()
            value = db.get_config_with_default(key, default)
            self._cache[key] = value
            self._cache_timestamps[key] = now
            return value
        except Exception:
            return default

    def invalidate_cache(self, key: Optional[str] = None):
        """Invalidate cache for key or all keys."""
        if key:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        else:
            self._cache.clear()
            self._cache_timestamps.clear()

    # Convenience properties for common configs
    @property
    def max_hold_time_minutes(self) -> int:
        return int(self.get('max_hold_time_minutes', 30))

    @property
    def max_retry_attempts(self) -> int:
        return int(self.get('max_retry_attempts', 3))

    @property
    def ai_temperature(self) -> float:
        return float(self.get('ai_temperature', 0.3))

    @property
    def disclosure_message(self) -> str:
        return str(self.get('disclosure_message',
            'Hello, this is an automated assistant calling on behalf of [provider_name].'))


class TelephonyManager:
    """Manages Twilio telephony operations"""
    
    def __init__(self):
        self.client = TwilioClient(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
        self.phone_number = os.getenv("TWILIO_PHONE_NUMBER")
    
    def initiate_call(self, to_number: str, callback_url: str) -> str:
        """Initiate outbound call"""
        call = self.client.calls.create(
            to=to_number,
            from_=self.phone_number,
            url=callback_url,
            status_callback=f"{callback_url}/status",
            record=True,
            machine_detection="Enable"
        )
        return call.sid
    
    def send_dtmf(self, call_sid: str, digits: str):
        """Send DTMF tones during call"""
        self.client.calls(call_sid).user_defined_messages.create(
            content={"digits": digits}
        )
    
    def play_audio(self, call_sid: str, audio_url: str):
        """Play audio during call"""
        self.client.calls(call_sid).update(
            url=audio_url
        )


class SpeechProcessor:
    """Handles speech-to-text and text-to-speech"""

    def __init__(self):
        self.deepgram = DeepgramClient()  # Auto-detects DEEPGRAM_API_KEY from environment
        self.elevenlabs = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    async def transcribe_stream(self, audio_stream):
        """Real-time transcription using Deepgram"""
        options = {
            "punctuate": True,
            "interim_results": True,
            "language": "en-US"
        }

        connection = self.deepgram.listen.live.v("1", options)
        return connection
    
    def generate_speech(self, text: str) -> bytes:
        """Generate speech using ElevenLabs"""
        audio = self.elevenlabs.generate(
            text=text,
            voice=self.voice_id,
            model="eleven_turbo_v2"
        )
        return audio


class AudioClassifier:
    """Classifies audio types using LLM"""

    def __init__(self, langsmith_config: LangSmithConfig):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.langsmith = langsmith_config
        
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an audio classification expert. Classify the transcript into one of these types:
            - ivr_menu: Automated menu with options (e.g., "Press 1 for...", "Say provider for...")
            - hold_music: Hold music or "please wait" messages
            - human_speech: Natural human conversation
            - silence: No meaningful audio
            
            Return JSON: {{"type": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""),
            ("user", "Transcript: {transcript}\n\nPrevious context: {context}")
        ])
    
    @traceable(name="classify_audio")
    def classify(self, transcript: str, context: str = "") -> Dict:
        """Classify audio type from transcript"""
        chain = self.classifier_prompt | self.llm | JsonOutputParser()
        
        result = chain.invoke({
            "transcript": transcript,
            "context": context
        })
        
        return result


class IVRNavigator:
    """Handles IVR menu navigation decisions"""

    def __init__(self, langsmith_config: LangSmithConfig):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.langsmith = langsmith_config
        
        self.navigator_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert IVR navigation agent. Your goal is to reach the credentialing department.
            
            Given the IVR menu transcript and known patterns, decide the best action:
            - dtmf: Press digits (return digit to press)
            - speech: Say a word/phrase (return what to say)
            - wait: Wait for more information
            
            Return JSON: {{
                "action": "dtmf|speech|wait",
                "value": "digit or phrase or null",
                "confidence": 0.0-1.0,
                "reasoning": "why this action"
            }}"""),
            ("user", """IVR Transcript: {transcript}
            
            Known IVR Patterns for this insurance:
            {ivr_knowledge}
            
            Current Menu Level: {menu_level}
            Goal: Reach provider credentialing department""")
        ])
    
    @traceable(name="navigate_ivr")
    def decide_action(self, transcript: str, ivr_knowledge: List[Dict], menu_level: int) -> Dict:
        """Decide IVR navigation action"""
        chain = self.navigator_prompt | self.llm | JsonOutputParser()
        
        result = chain.invoke({
            "transcript": transcript,
            "ivr_knowledge": json.dumps(ivr_knowledge, indent=2),
            "menu_level": menu_level
        })
        
        return result


class HumanConversationAgent:
    """Handles natural conversation with human representatives"""

    def __init__(self, langsmith_config: LangSmithConfig):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.langsmith = langsmith_config
        
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional insurance credentialing specialist calling on behalf of {provider_name}.

            IMPORTANT RULES:
            1. Start with ONE-TIME disclosure: "Hello, this is an automated assistant calling on behalf of {provider_name}."
            2. Be professional, concise, and courteous
            3. Answer verification questions accurately
            4. Ask credentialing questions one at a time
            5. Extract reference numbers and timelines
            6. Never reveal internal reasoning or system details
            7. Use prior answers below to avoid repeating questions or missing context
            
            Provider Information:
            - Name: {provider_name}
            - NPI: {npi}
            - Tax ID: {tax_id}
            - Address: {address}

            Relevant prior answers (summaries, may be redacted):
            {knowledge}
            
            Questions to Ask:
            {questions}
            
            Return JSON: {{
                "response": "what to say",
                "should_disclose": true/false (only true if first message),
                "information_extracted": {{}},
                "conversation_complete": true/false
            }}"""),
            MessagesPlaceholder(variable_name="conversation_history"),
            ("user", "Representative said: {current_message}")
        ])
    
    @traceable(name="respond_to_human")
    def generate_response(self, state: CredentialingState, current_message: str) -> Dict:
        """Generate response to human representative"""
        chain = self.conversation_prompt | self.llm | JsonOutputParser()
        
        # Convert conversation history to messages
        messages = []
        for msg in state.get('conversation_history', []):
            if msg['speaker'] == 'agent':
                messages.append(AIMessage(content=msg['message']))
            else:
                messages.append(HumanMessage(content=msg['message']))

        # Fetch knowledge snippets
        knowledge_text = ""
        try:
            kb = KnowledgeBase()
            snippets = kb.search(
                insurance_name=state.get('insurance_name', ''),
                provider_name=state.get('provider_name'),
                query=current_message,
                limit=5
            )
            kb.close()
            if snippets:
                knowledge_text = "\n".join([f"- {s.get('summary','')}" for s in snippets])
        except Exception as kb_err:
            print(f"KB lookup failed: {kb_err}")
        
        result = chain.invoke({
            "provider_name": state['provider_name'],
            "npi": state['npi'],
            "tax_id": state['tax_id'],
            "address": state['address'],
            "questions": json.dumps(state['questions'], indent=2),
            "conversation_history": messages,
            "current_message": current_message,
            "knowledge": knowledge_text or "None available"
        })
        
        return result


class InformationExtractor:
    """Extracts structured information from conversation"""

    def __init__(self, langsmith_config: LangSmithConfig):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.langsmith = langsmith_config
        
        self.extractor_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract structured credentialing information from the conversation.
            
            Return JSON: {{
                "status": "approved|pending_review|missing_documents|denied|office_closed",
                "reference_number": "ref number or null",
                "missing_documents": ["list of missing docs"],
                "turnaround_days": number or null,
                "next_action_date": "YYYY-MM-DD or null",
                "notes": "summary of important information"
            }}"""),
            ("user", "Conversation history:\n{conversation}")
        ])
    
    @traceable(name="extract_information")
    def extract(self, conversation_history: List[Dict]) -> Dict:
        """Extract structured information"""
        # Format conversation
        conversation = "\n".join([
            f"{msg['speaker']}: {msg['message']}"
            for msg in conversation_history
        ])
        
        chain = self.extractor_prompt | self.llm | JsonOutputParser()
        
        result = chain.invoke({
            "conversation": conversation
        })
        
        return result


class CredentialingAgent:
    """Main autonomous credentialing agent using LangGraph"""
    
    def __init__(self):
        self.langsmith = LangSmithConfig()
        self.db = DatabaseManager()
        self.telephony = TelephonyManager()
        self.speech = SpeechProcessor()
        
        # AI components
        self.audio_classifier = AudioClassifier(self.langsmith)
        self.ivr_navigator = IVRNavigator(self.langsmith)
        self.conversation_agent = HumanConversationAgent(self.langsmith)
        self.info_extractor = InformationExtractor(self.langsmith)
        self.kb_summarizer = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL_KB_SUMMARY", "gpt-4"),
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.kb_summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize the credentialing phone call for future reuse.
Return JSON:
{
  "summary": "2-4 bullet summary (concise)",
  "qa": [
    {"q": "...", "a": "..."},
    ...
  ]
}
Keep answers redacted of NPIs/tax IDs/phones; keep 3-5 QA pairs max."""),
            ("user", "Conversation:\n{conversation}")
        ])
        
        # Build state graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        workflow = StateGraph(CredentialingState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_call)
        workflow.add_node("classify_audio", self._classify_audio)
        workflow.add_node("navigate_ivr", self._navigate_ivr)
        workflow.add_node("handle_hold", self._handle_hold)
        workflow.add_node("converse_with_human", self._converse_with_human)
        workflow.add_node("extract_results", self._extract_results)
        workflow.add_node("finalize", self._finalize_call)
        
        # Set entry point
        workflow.set_entry_point("initialize")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "initialize",
            self._route_after_init,
            {
                "classify": "classify_audio",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "classify_audio",
            self._route_by_audio_type,
            {
                "ivr": "navigate_ivr",
                "hold": "handle_hold",
                "human": "converse_with_human",
                "classify": "classify_audio"
            }
        )
        
        workflow.add_conditional_edges(
            "navigate_ivr",
            self._check_continue,
            {
                "continue": "classify_audio",
                "complete": "extract_results",
                "error": END
            }
        )
        
        workflow.add_edge("handle_hold", "classify_audio")
        
        workflow.add_conditional_edges(
            "converse_with_human",
            self._check_conversation_complete,
            {
                "continue": "classify_audio",
                "complete": "extract_results"
            }
        )
        
        workflow.add_edge("extract_results", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile with checkpointer for persistence (using in-memory for development)
        checkpointer = MemorySaver()

        return workflow.compile(checkpointer=checkpointer)
    
    @traceable(name="initialize_call")
    def _initialize_call(self, state: CredentialingState) -> CredentialingState:
        """Initialize the call"""
        # Save request to database unless already provided (e.g., API created it)
        request_id = state.get('db_request_id')
        if not request_id:
            request_id = self.db.save_credentialing_request(state)
        
        # Get IVR knowledge
        ivr_knowledge = self.db.get_ivr_knowledge(state['insurance_name'])
        
        # Initiate Twilio call
        callback_url = os.getenv("CALLBACK_URL")
        call_sid = self.telephony.initiate_call(
            state['insurance_phone'],
            callback_url
        )
        
        state['call_sid'] = call_sid
        state['call_id'] = request_id
        state['db_request_id'] = request_id
        state['ivr_knowledge'] = ivr_knowledge
        state['current_menu_level'] = 0
        state['retry_count'] = 0
        state['call_state'] = CallState.IVR_NAVIGATION
        state['should_continue'] = True
        
        return state
    
    @traceable(name="classify_audio")
    def _classify_audio(self, state: CredentialingState) -> CredentialingState:
        """Classify current audio"""
        if not state.get('transcript'):
            state['current_audio_type'] = AudioType.SILENCE
            return state
        
        latest_transcript = state['transcript'][-1]['text']
        context = "\n".join([t['text'] for t in state['transcript'][-3:]])
        
        classification = self.audio_classifier.classify(latest_transcript, context)
        
        state['current_audio_type'] = AudioType(classification['type'])
        state['confidence'] = classification['confidence']
        
        # Log event
        self.db.log_call_event(
            state['call_id'],
            'audio_classified',
            {
                'transcript': latest_transcript,
                'classification': classification
            }
        )
        
        return state
    
    @traceable(name="navigate_ivr")
    def _navigate_ivr(self, state: CredentialingState) -> CredentialingState:
        """Navigate IVR menu"""
        latest_transcript = state['transcript'][-1]['text']
        
        action = self.ivr_navigator.decide_action(
            latest_transcript,
            state['ivr_knowledge'],
            state['current_menu_level']
        )
        
        # Execute action
        if action['action'] == ActionType.DTMF:
            self.telephony.send_dtmf(state['call_sid'], action['value'])
        elif action['action'] == ActionType.SPEECH:
            # Generate and play speech
            audio = self.speech.generate_speech(action['value'])
            # Upload audio and play (implementation depends on your setup)
            pass
        
        state['last_action'] = action
        state['current_menu_level'] += 1
        
        # Log action
        self.db.log_call_event(
            state['call_id'],
            'ivr_action',
            {
                'action': action,
                'menu_level': state['current_menu_level']
            }
        )
        
        return state
    
    @traceable(name="handle_hold")
    def _handle_hold(self, state: CredentialingState) -> CredentialingState:
        """Handle hold state - just wait"""
        state['call_state'] = CallState.ON_HOLD
        
        self.db.log_call_event(
            state['call_id'],
            'on_hold',
            {'timestamp': datetime.now().isoformat()}
        )
        
        return state
    
    @traceable(name="converse_with_human")
    def _converse_with_human(self, state: CredentialingState) -> CredentialingState:
        """Have natural conversation with human"""
        state['call_state'] = CallState.SPEAKING_WITH_HUMAN
        
        # Check if transcript is empty
        if not state.get('transcript') or len(state['transcript']) == 0:
            # No transcript yet, return state without processing
            return state
        
        latest_transcript = state['transcript'][-1]['text']
        
        response = self.conversation_agent.generate_response(state, latest_transcript)
        
        # Save to conversation history
        self.db.save_conversation(state['call_id'], 'representative', latest_transcript)
        self.db.save_conversation(state['call_id'], 'agent', response['response'])
        
        # Add to state
        state['conversation_history'].append({
            'speaker': 'representative',
            'message': latest_transcript
        })
        state['conversation_history'].append({
            'speaker': 'agent',
            'message': response['response']
        })
        
        # Generate and play speech
        audio = self.speech.generate_speech(response['response'])
        # Play audio (implementation specific)
        
        # Check if conversation is complete
        if response.get('conversation_complete'):
            state['should_continue'] = False
        
        return state
    
    @traceable(name="extract_results")
    def _extract_results(self, state: CredentialingState) -> CredentialingState:
        """Extract structured results from conversation"""
        results = self.info_extractor.extract(state['conversation_history'])
        
        state['credentialing_status'] = results['status']
        state['reference_number'] = results.get('reference_number')
        state['missing_documents'] = results.get('missing_documents', [])
        state['turnaround_days'] = results.get('turnaround_days')
        state['notes'] = results['notes']
        
        return state

    def _summarize_for_kb(self, state: CredentialingState) -> Dict:
        """Create concise summary and QA pairs for knowledge base ingestion."""
        conversation = state.get('conversation_history', [])
        if not conversation and state.get('transcript'):
            # Fallback: use transcript list of dicts with 'text'
            conversation_text = "\n".join([t.get('text', '') for t in state['transcript']])
        else:
            conversation_text = "\n".join([
                f"{msg['speaker']}: {msg.get('message', '')}"
                for msg in conversation
            ])
        chain = self.kb_summary_prompt | self.kb_summarizer | JsonOutputParser()
        try:
            return chain.invoke({"conversation": conversation_text})
        except Exception as e:
            print(f"KB summary failed: {e}")
            return {
                "summary": "Credentialing call completed; summary unavailable.",
                "qa": []
            }

    def _ingest_knowledge(self, state: CredentialingState):
        """Store knowledge in pgvector for future calls."""
        try:
            kb_data = self._summarize_for_kb(state)
            summary = redact_text(kb_data.get("summary", ""))
            qa_pairs = kb_data.get("qa", []) or []
            qa_lines = []
            for pair in qa_pairs[:5]:
                q = redact_text(pair.get("q", ""))
                a = redact_text(pair.get("a", ""))
                qa_lines.append(f"- Q: {q}\n  A: {a}")
            qa_text = "\n".join(qa_lines) if qa_lines else ""

            metadata = {
                "status": state.get("credentialing_status"),
                "reference_number": state.get("reference_number"),
                "missing_documents": state.get("missing_documents", []),
                "turnaround_days": state.get("turnaround_days"),
            }

            kb = KnowledgeBase()
            kb.upsert_entry(
                insurance_name=state.get("insurance_name"),
                provider_name=state.get("provider_name"),
                call_id=state.get("call_id"),
                request_id=state.get("db_request_id") or state.get("call_id"),
                summary=summary,
                qa_text=qa_text,
                metadata=metadata,
            )
            kb.close()
            print("Knowledge base ingestion complete.")
        except Exception as e:
            print(f"Knowledge base ingestion failed: {e}")
    
    @traceable(name="finalize_call")
    def _finalize_call(self, state: CredentialingState) -> CredentialingState:
        """Finalize and save results"""
        state['call_state'] = CallState.COMPLETING
        call_end_time = datetime.now()

        # Calculate timing metrics
        duration_seconds = None
        ivr_time = None
        hold_time = None
        human_time = None

        call_start = state.get('call_start_time')
        if call_start:
            duration_seconds = int((call_end_time - call_start).total_seconds())

        ivr_end = state.get('ivr_end_time')
        if call_start and ivr_end:
            ivr_time = int((ivr_end - call_start).total_seconds())

        hold_end = state.get('hold_end_time')
        if ivr_end and hold_end:
            hold_time = int((hold_end - ivr_end).total_seconds())

        human_start = state.get('human_start_time')
        if human_start:
            human_time = int((call_end_time - human_start).total_seconds())

        # Record call metrics
        try:
            self.db.insert_call_metrics(
                call_id=state['call_id'],
                request_id=state.get('db_request_id'),
                duration_seconds=duration_seconds,
                ivr_navigation_time_seconds=ivr_time,
                hold_time_seconds=hold_time,
                human_interaction_time_seconds=human_time,
                successful=(state.get('credentialing_status') not in ['failed', None]),
                failure_reason=state.get('error_message'),
                retry_count=state.get('retry_count', 0),
            )
            print(f"Call metrics recorded for call {state['call_id']}")
        except Exception as e:
            print(f"Failed to record call metrics: {e}")

        # Save final results
        self.db.save_final_results(state['call_id'], state)

        # Ingest into knowledge base for future calls
        self._ingest_knowledge(state)

        # Schedule follow-up if needed
        if state['credentialing_status'] in ['pending_review', 'missing_documents']:
            # Schedule follow-up logic here
            pass

        return state
    
    # Routing functions
    def _route_after_init(self, state: CredentialingState) -> str:
        """Route after initialization"""
        if state.get('error_message'):
            return "error"
        return "classify"
    
    def _route_by_audio_type(self, state: CredentialingState) -> str:
        """Route based on audio classification"""
        audio_type = state.get('current_audio_type', AudioType.UNKNOWN)

        if audio_type == AudioType.IVR_MENU:
            return "ivr"
        elif audio_type == AudioType.HOLD_MUSIC:
            return "hold"
        elif audio_type == AudioType.HUMAN_SPEECH:
            return "human"
        else:
            # Default to human to prevent infinite loop when audio type is unknown
            return "human"
    
    def _check_continue(self, state: CredentialingState) -> str:
        """Check if should continue or complete"""
        if state.get('error_message'):
            return "error"
        if not state.get('should_continue', True):
            return "complete"
        if state.get('retry_count', 0) > 3:
            return "error"
        return "continue"
    
    def _check_conversation_complete(self, state: CredentialingState) -> str:
        """Check if conversation is complete"""
        if not state.get('should_continue', True):
            return "complete"
        return "continue"
    
    async def process_call(self, initial_state: CredentialingState) -> CredentialingState:
        """Process a complete credentialing call"""
        config = {"configurable": {"thread_id": initial_state.get('call_id', 'default')}}
        
        final_state = await self.graph.ainvoke(initial_state, config=config)
        
        return final_state
    
    def close(self):
        """Cleanup resources"""
        self.db.close()


# Example usage
if __name__ == "__main__":
    # Create initial state
    initial_state: CredentialingState = {
        'insurance_name': 'Blue Cross Blue Shield',
        'provider_name': 'Dr. Jane Smith',
        'npi': '1234567890',
        'tax_id': '12-3456789',
        'address': '123 Main Street, Suite 100, Anytown, ST 12345',
        'insurance_phone': '+18001234567',
        'questions': [
            'What is the current status of my credentialing application?',
            'Are any additional documents required?',
            'What is the expected completion date?'
        ],
        'call_id': None,
        'call_sid': None,
        'db_request_id': None,
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
        'error_message': None
    }
    
    # Run agent
    agent = CredentialingAgent()
    
    # Process call (async)
    # final_state = asyncio.run(agent.process_call(initial_state))
    # print(f"Final state: {final_state}")
    
    agent.close()
