---
name: extract_qa_async must receive call_id not request_id
description: extract_qa_async queries conversation_history by call_id — passing request_id silently returns no data
type: feedback
---

`extract_qa_async(call_id)` in `qa_extractor.py` queries `conversation_history WHERE call_id = %s`. It resolves `request_id` itself via an internal DB JOIN. Always pass the internal UUID `call_id`, never `request_id`.

**Why:** A prior bug passed `request_id or call_id` to `extract_qa_async`. When `request_id` was set, the function received a `request_id` UUID, found zero rows in `conversation_history` (which is indexed on `call_id`), and logged "No conversation found" — silently skipping all Q&A extraction.

**How to apply:** Any code that calls `extract_qa_async` must pass the `call_id` field from `CredentialingState` (the internal UUID), not `db_request_id` or `request_id`. The function handles its own request_id resolution internally.
