---
name: Latency Optimization Implementation — March 2026
description: Records what latency reductions were applied to the backend, why, and the constraints enforced
type: project
---

Eight infrastructure-only latency improvements were applied on 2026-03-16.  No logic, call-flow, state-machine, or API behaviour was changed.

**Why:** Reduce per-call and per-request latency without touching business logic.  Constraint: no logic changes, only infrastructure/initialization/query improvements.

**How to apply:** When proposing future perf work, build on these foundations — pools are already wired, singletons exist, indexes are created.

## What was done

1. **DB connection pooling** (`psycopg2.pool.ThreadedConnectionPool`):
   - `credentialing_agent.py`: module-level `_db_pool` (min=5, max=20). `DatabaseManager.__init__` calls `_db_pool.getconn()`; `close()` calls `putconn()`.
   - `auth.py`: module-level `_auth_pool` (min=2, max=10). `_get_conn()` acquires from pool; `_query_one`/`_execute` call `putconn()` in finally.
   - `knowledge_base.py`: module-level `_kb_pool` (min=2, max=10). `KnowledgeBase.__init__` calls `_kb_pool.getconn()`; `close()` calls `putconn()`.

2. **DB indexes**: `backend/migrations/add_perf_indexes.sql` created with 10 `CREATE INDEX IF NOT EXISTS` statements covering `conversation_history`, `call_events`, `call_recordings`, `call_metrics`, `credentialing_requests`, `ivr_knowledge`, `call_knowledge`, `insurance_providers`.

3. **Singleton LLM clients**: Four module-level `ChatOpenAI` instances (`_llm_classifier`, `_llm_navigator`, `_llm_conversation`, `_llm_extractor`) shared across `AudioClassifier`, `IVRNavigator`, `HumanConversationAgent`, `InformationExtractor`.

4. **Cached LangGraph**: `_compiled_graph` module-level variable; `_build_graph()` returns cached graph on second+ calls.

5. **Singleton KnowledgeBase**:
   - `HumanConversationAgent.__init__` creates `self.kb`; `generate_response()` uses `self.kb.search()` directly.
   - `CredentialingAgent.__init__` creates `self.kb`; `_ingest_knowledge()` uses `self.kb.upsert_entry()`.
   - `CredentialingAgent.close()` also calls `self.kb.close()`.

6. **Production WSGI**: `gunicorn>=21.2.0` added to `requirements.txt`. Start command documented in comment: `gunicorn -w 4 --threads 4 -b 0.0.0.0:5000 --timeout 300 "api_server:app"`.

7. **Parallel DB calls** (`concurrent.futures.ThreadPoolExecutor`):
   - `_initialize_call`: when `request_id` is absent, `save_credentialing_request` + `get_ivr_knowledge` run in a 2-worker pool simultaneously.
   - `_converse_with_human`: the two `save_conversation` calls (representative + agent rows) run in a 2-worker pool simultaneously.

8. **IVR knowledge TTL cache**: `_ivr_knowledge_cache` dict + `_IVR_CACHE_TTL_SECONDS=1800` at module level. `get_ivr_knowledge()` checks cache before querying DB; cache key is `insurance_name.lower()`.
