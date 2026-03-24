---
name: AI vs Real Agent Toggle Feature
description: Call mode toggle added to start-call form; Take Over button added to active-calls table; transferToAgent API method added
type: project
---

The AI vs Real Agent toggle feature was implemented across three files:

- `frontend/types/index.ts`: Added `call_mode?: string` and `agent_phone?: string` to `CredentialingRequest`.
- `frontend/lib/api.ts`: Added `transferToAgent(callId, agentPhone)` POSTing to `/api/transfer-to-agent`.
- `frontend/components/dashboard/start-call-form.tsx`: Added `callMode` ('ai' | 'human') and `agentPhone` state. Toggle UI (two styled buttons) placed above the questions field. When 'human' is selected, an agent phone input appears. Both values are passed into `api.startCall`.
- `frontend/components/dashboard/active-calls-table.tsx`: Added "Take Over" button per row (visible only when `call_mode === 'ai'` or absent, and transfer not yet done). Button opens a Dialog (using `@/components/ui/dialog`) asking for the agent phone, then calls `api.transferToAgent`. On success, the row's status badge switches to "Agent Connected" and the button disappears. Uses `useMutation` from `@tanstack/react-query` and `toast` from `sonner`.

**Why:** Feature request to support routing calls to either the AI agent or a real human agent at call start, and to allow mid-call handoff from the dashboard.

**How to apply:** When working on call-related forms or the active calls table, these fields and patterns are already established — extend consistently.
