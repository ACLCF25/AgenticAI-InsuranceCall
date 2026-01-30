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
from datetime import datetime
from typing import Dict, Optional
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from twilio.twiml.voice_response import VoiceResponse, Stream, Connect
from twilio.rest import Client as TwilioClient
import threading
from queue import Queue
import base64

from credentialing_agent import (
    CredentialingAgent,
    CredentialingState,
    CallState,
    AudioType
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

        # Validate required fields
        required = ['insurance_name', 'provider_name', 'npi', 'tax_id',
                   'address', 'insurance_phone', 'questions']
        for field in required:
            if field not in data:
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

        # Create initial state
        initial_state: CredentialingState = {
            'insurance_name': data['insurance_name'],
            'provider_name': data['provider_name'],
            'npi': data['npi'],
            'tax_id': data['tax_id'],
            'address': data['address'],
            'insurance_phone': data['insurance_phone'],
            'questions': data['questions'],
            'call_id': None,
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

        # Create agent
        agent = CredentialingAgent()

        # Start call in background thread
        def run_call():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            final_state = loop.run_until_complete(agent.process_call(initial_state))
            loop.close()

            # Store final state
            if final_state.get('call_id'):
                call_states[final_state['call_id']] = final_state

        thread = threading.Thread(target=run_call)
        thread.start()

        return jsonify({
            'success': True,
            'message': 'Call initiated',
            'provider': data['provider_name'],
            'insurance': data['insurance_name']
        }), 200

    except Exception as e:
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
    response = VoiceResponse()
    
    # Get call SID
    call_sid = request.values.get('CallSid')
    
    # Start media stream for real-time audio
    connect = Connect()
    stream = Stream(url=f'wss://{request.host}/media-stream')
    connect.append(stream)
    response.append(connect)
    
    # Say initial message while connecting stream
    response.say(
        "Connecting to credentialing department, please wait.",
        voice='Polly.Joanna'
    )
    
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


# WebSocket handler for media streaming (requires flask-socketio)
# This is a placeholder - you'd need to implement proper WebSocket handling
# for real-time audio streaming from Twilio

"""
@socketio.on('media')
def handle_media(data):
    # Handle incoming media stream from Twilio
    call_sid = data.get('streamSid')
    audio_data = base64.b64decode(data.get('media', {}).get('payload', ''))
    
    # Send to Deepgram for transcription
    # Process through agent
    pass
"""


if __name__ == '__main__':
    # For development
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('DEBUG', 'False').lower() == 'true'
    )
