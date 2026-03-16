---
name: Agentic Call System Architecture
description: Core architecture of the monolith backend — Flask API, Twilio telephony, Supabase DB, call flow identifiers
type: project
---

Flask-based backend at `backend/api_server.py`. Next.js frontend at `frontend/`.

**Why:** Understanding the dual-ID system is critical for any recording or call lookup work.

**How to apply:** Always account for both runtime call_id and db_request_id when querying recordings or call data.

## Key Identifier System
- `call_id`: runtime UUID generated at call start (in-memory only, not persisted to credentialing_requests)
- `db_request_id` (a.k.a. `request_id`): Postgres UUID from `credentialing_requests.id` — this is the stable, persistent identifier
- `call_sid`: Twilio SID — only known after Twilio call is placed
- `call_recordings.call_id` stores the runtime UUID; `call_recordings.request_id` stores the Postgres UUID

## Recording Flow
1. Call starts → runtime `call_id` + `db_request_id` created, stored in `call_states` in-memory dict
2. Twilio `recording-status` webhook fires → looks up state by `call_sid` from `call_states_by_sid`
3. Recording saved with `call_id = runtime_uuid`, `request_id = db_request_id`
4. `get_call_detail` uses `lookup_id = db_request_id` for all DB lookups
5. `get_recording` must search by all three: `call_id`, `request_id`, `call_sid`

## DB Schema (Supabase)
- `credentialing_requests` — main call records (id = Postgres UUID)
- `call_recordings` — Twilio recording metadata (call_id VARCHAR, request_id UUID FK, call_sid VARCHAR)
- `call_events` — timestamped events per call
- `conversation_history` — agent/rep messages
- `call_metrics` — performance data, also stores recording_sid
- `call_qa_pairs` — extracted Q&A pairs post-call

## Key API Endpoints
- `POST /api/start-call` — initiates outbound Twilio call
- `GET /api/call-detail/<call_id>` — full call detail with recording + conversation
- `GET /api/call-recording/<call_id>/stream` — proxy-streams Twilio audio
- `POST /webhook/recording-status` — Twilio callback when recording is ready
- `POST /webhook/voice` — Twilio TwiML callback

## Frontend
- `NEXT_PUBLIC_API_URL` defaults to `http://localhost:5000/api`
- Recording audio `src` = `${NEXT_PUBLIC_API_URL}/api/call-recording/{id}/stream`
- Recording visible only when `recording.available === true` (status == 'completed')
