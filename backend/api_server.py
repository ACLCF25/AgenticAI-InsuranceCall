"""
Flask API Server for Autonomous Credentialing Agent
Handles Twilio webhooks and real-time audio streaming
"""

import os
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state management
active_calls: Dict[str, CredentialingAgent] = {}
call_states: Dict[str, CredentialingState] = {}
audio_queues: Dict[str, Queue] = {}

# Twilio client
twilio_client = TwilioClient(
    os.getenv("TWILIO_ACCOUNT_SID"),
    os.getenv("TWILIO_AUTH_TOKEN")
)


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

        # Check if required environment variables are set
        missing_env = []
        if not os.getenv("SUPABASE_HOST"):
            missing_env.append("SUPABASE_HOST")
        if not os.getenv("SUPABASE_PASSWORD"):
            missing_env.append("SUPABASE_PASSWORD")
        if not os.getenv("TWILIO_ACCOUNT_SID"):
            missing_env.append("TWILIO_ACCOUNT_SID")
        if not os.getenv("TWILIO_AUTH_TOKEN"):
            missing_env.append("TWILIO_AUTH_TOKEN")

        if missing_env:
            return jsonify({
                'success': False,
                'error': f'Missing required environment variables: {", ".join(missing_env)}. Please configure your .env file.'
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
            'call_id': call_id,
            'call_sid': None,
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

        # Store state immediately so frontend can track it
        call_states[call_id] = initial_state
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
                call_states[call_id] = final_state
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
        print(f"📞 Voice webhook called - CallSID: {call_sid}")

        # Get the base URL from environment or request
        callback_base = os.getenv('CALLBACK_URL', '').replace('/webhook/voice', '')
        if not callback_base:
            callback_base = f"https://{request.host}"
        print(f"📍 Callback base URL: {callback_base}")

        # Find the call state by call_id that matches this call
        call_info = None
        for call_id, state in list(call_states.items()):
            if state.get('call_sid') == call_sid or state.get('call_sid') is None:
                # Update call_sid if not set
                if state.get('call_sid') is None:
                    state['call_sid'] = call_sid
                    state['call_state'] = CallState.SPEAKING_WITH_HUMAN
                    print(f"✅ Linked CallSID {call_sid} to call_id {call_id}")
                call_info = {'call_id': call_id, 'state': state}
                break

        # Get provider name for the disclosure message
        provider_name = "a healthcare provider"
        if call_info:
            provider_name = call_info['state'].get('provider_name', provider_name)

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
        gather.say(disclosure, voice='Polly.Joanna')
        response.append(gather)

        # If no input, say goodbye
        response.say("I didn't hear a response. I'll try again later. Goodbye.", voice='Polly.Joanna')

        print(f"📤 Returning TwiML: {str(response)[:200]}...")
        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in voice_webhook: {e}")
        import traceback
        traceback.print_exc()
        # Return a simple error response
        response = VoiceResponse()
        response.say("Sorry, there was a technical error. Please try again later.", voice='Polly.Joanna')
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/speech', methods=['POST'])
def speech_webhook():
    """
    Handle speech input from Twilio Gather
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

        # Simple response logic based on what was heard
        speech_lower = speech_result.lower()

        if any(word in speech_lower for word in ['yes', 'correct', 'right', 'credentialing', 'department']):
            # They confirmed it's credentialing
            if call_info:
                questions = call_info['state'].get('questions', [])
                provider_name = call_info['state'].get('provider_name', 'the provider')
                npi = call_info['state'].get('npi', '')

                # Ask the first question
                message = f"Thank you. I'm calling to check on the credentialing status for {provider_name}, N P I number {' '.join(npi)}. "
                if questions:
                    message += questions[0] if isinstance(questions[0], str) else str(questions[0])
                else:
                    message += "Can you tell me the current status of the application?"

                gather = Gather(
                    input='speech',
                    action=f'{callback_base}/webhook/speech/followup',
                    method='POST',
                    speech_timeout='auto',
                    language='en-US'
                )
                gather.say(message, voice='Polly.Joanna')
                response.append(gather)
            else:
                response.say("Thank you. Can you tell me the status of the credentialing application?", voice='Polly.Joanna')

        elif any(word in speech_lower for word in ['no', 'wrong', 'number', 'transfer', 'hold']):
            # Wrong department or need transfer
            response.say("I apologize for the confusion. Could you please transfer me to the credentialing department?", voice='Polly.Joanna')
            response.pause(length=30)  # Wait for transfer

        else:
            # Unclear response, ask for clarification
            gather = Gather(
                input='speech',
                action=f'{callback_base}/webhook/speech',
                method='POST',
                speech_timeout='auto',
                language='en-US'
            )
            gather.say("I'm sorry, I didn't quite catch that. Is this the credentialing department?", voice='Polly.Joanna')
            response.append(gather)

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in speech_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        response.say("Sorry, there was a technical error. Goodbye.", voice='Polly.Joanna')
        return Response(str(response), mimetype='text/xml')


@app.route('/webhook/speech/followup', methods=['POST'])
def speech_followup_webhook():
    """
    Handle follow-up speech input
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

        # Extract any useful information from the response
        speech_lower = speech_result.lower()

        # Check for status information
        if call_info:
            state = call_info['state']

            # Look for status keywords
            if any(word in speech_lower for word in ['approved', 'active', 'completed', 'effective']):
                state['notes'] += f"Status: Approved/Active. "
            elif any(word in speech_lower for word in ['pending', 'processing', 'review']):
                state['notes'] += f"Status: Pending. "
            elif any(word in speech_lower for word in ['denied', 'rejected']):
                state['notes'] += f"Status: Denied. "

            # Look for timeline
            if 'days' in speech_lower or 'weeks' in speech_lower:
                state['notes'] += f"Timeline mentioned: {speech_result}. "

            # Check if there are more questions
            questions = state.get('questions', [])
            asked_count = len([t for t in state.get('transcript', []) if t.get('speaker') == 'agent'])

            if asked_count < len(questions):
                # Ask next question
                next_question = questions[asked_count] if isinstance(questions[asked_count], str) else str(questions[asked_count])
                gather = Gather(
                    input='speech',
                    action=f'{callback_base}/webhook/speech/followup',
                    method='POST',
                    speech_timeout='auto',
                    language='en-US'
                )
                gather.say(f"Thank you. {next_question}", voice='Polly.Joanna')
                response.append(gather)

                # Record that we asked
                state['transcript'].append({
                    'speaker': 'agent',
                    'text': next_question,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                # All questions asked, wrap up
                state['call_state'] = CallState.COMPLETING
                state['notes'] += f"Full response: {speech_result}"
                response.say("Thank you very much for your help. Have a great day. Goodbye.", voice='Polly.Joanna')
                response.hangup()
        else:
            response.say("Thank you for the information. Goodbye.", voice='Polly.Joanna')
            response.hangup()

        return Response(str(response), mimetype='text/xml')

    except Exception as e:
        print(f"❌ Error in speech_followup_webhook: {e}")
        import traceback
        traceback.print_exc()
        response = VoiceResponse()
        response.say("Sorry, there was a technical error. Goodbye.", voice='Polly.Joanna')
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
