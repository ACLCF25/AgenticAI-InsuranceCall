---
name: AI vs Real-Agent Toggle Feature
description: call_mode toggle ('ai'|'agent'), Conference-based live takeover, and transfer-to-agent endpoint — implemented 2026-03-18
type: project
---

The AI vs Real-Agent toggle feature was implemented on 2026-03-18.

**Why:** Allow operators to route a call either to the AI credentialing agent or directly to a human real agent, and support mid-call live takeover.

**How to apply:** When touching call routing, recording, or call-state logic, be aware of the `call_mode` field and `transfer_to_agent` flag.

## Schema changes (`credentialing_requests` table)
- `call_mode VARCHAR(10) DEFAULT 'ai' CHECK (call_mode IN ('ai', 'agent'))`
- `agent_phone VARCHAR(20)`
- `conference_sid VARCHAR(100)`
- Migration ALTERs provided as comments in `supabase_schema.sql` around line 248.

## API changes (`api_server.py`)

### `/api/start-call` (POST, JWT required)
- Accepts optional `call_mode` (default `'ai'`) and `agent_phone` fields.
- Stores both in `call_states[call_id]` and persists to DB via UPDATE after INSERT.
- New state keys: `call_mode`, `agent_phone`, `conference_sid`, `transfer_to_agent`.

### `/webhook/voice` (POST)
- Early-return block at the top of the AI/IVR routing logic:
  - If `call_state.get('call_mode') == 'agent'`, returns Conference TwiML immediately.
  - Conference name = `call_id` (the runtime UUID).
  - Insurance leg: `endConferenceOnExit="true"` (conference closes when insurance hangs up).
  - Agent leg: `endConferenceOnExit="false"` (agent can leave without dropping insurance).
  - Uses module-level `twilio_client` to `calls.create(twiml=...)` for the agent dial.

### `POST /api/transfer-to-agent` (JWT required)
- Body: `{ call_id, agent_phone }`
- Redirects the live Twilio call (`call_sid`) into a Conference room via `twilio_client.calls(call_sid).update(twiml=...)`.
- Dials `agent_phone` into the same conference via `twilio_client.calls.create(...)`.
- Sets `call_states[call_id]['transfer_to_agent'] = True` to stop the AI loop.
- Persists `call_mode='agent'`, `agent_phone`, `conference_sid` to DB.

## Agent changes (`credentialing_agent.py`)
- `_check_continue()`: checks `state.get('transfer_to_agent')` → returns `"complete"`.
- `_check_conversation_complete()`: same check → returns `"complete"`.
- `_route_by_audio_type()`: same check → returns `"error"` (exits graph immediately).
- All three log a message and set `state['should_continue'] = False` before exiting.
