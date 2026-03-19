# Agent Memory Index

## Project
- [project_architecture.md](./project_architecture.md) — Core system architecture, dual call ID system, recording flow, DB schema, key API endpoints
- [project_latency_optimizations.md](./project_latency_optimizations.md) — All 8 latency improvements applied 2026-03-16: connection pools, indexes, LLM singletons, cached graph, parallel DB calls, IVR cache
- [project_ai_vs_agent_toggle.md](./project_ai_vs_agent_toggle.md) — AI vs real-agent toggle: call_mode field, Conference TwiML routing in /webhook/voice, /api/transfer-to-agent endpoint, transfer_to_agent flag in credentialing_agent.py

## Feedback
- [feedback_databasemanager_import_pattern.md](./feedback_databasemanager_import_pattern.md) — api_server.py uses inline local imports for DatabaseManager; no top-level import exists
- [feedback_extract_qa_async_call_id.md](./feedback_extract_qa_async_call_id.md) — extract_qa_async must receive call_id (not request_id); it resolves request_id internally via JOIN
