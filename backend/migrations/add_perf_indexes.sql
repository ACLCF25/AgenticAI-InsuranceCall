-- Performance indexes for the Agentic Call backend.
-- All statements use IF NOT EXISTS so this file is safe to run multiple times.
-- Run once against the Supabase/Postgres database to apply.

-- conversation_history: most queries filter or join on call_id or request_id
CREATE INDEX IF NOT EXISTS idx_conversation_history_call_id
    ON conversation_history(call_id);

CREATE INDEX IF NOT EXISTS idx_conversation_history_request_id
    ON conversation_history(request_id);

-- call_events: always queried by call_id
CREATE INDEX IF NOT EXISTS idx_call_events_call_id
    ON call_events(call_id);

-- call_recordings: looked up by call_id and request_id (dual-ID pattern)
CREATE INDEX IF NOT EXISTS idx_call_recordings_call_id
    ON call_recordings(call_id);

CREATE INDEX IF NOT EXISTS idx_call_recordings_request_id
    ON call_recordings(request_id);

-- call_metrics: queried and joined by call_id
CREATE INDEX IF NOT EXISTS idx_call_metrics_call_id
    ON call_metrics(call_id);

-- credentialing_requests: list/filter pages always sort by status + created_at
CREATE INDEX IF NOT EXISTS idx_credentialing_requests_status
    ON credentialing_requests(status, created_at DESC);

-- ivr_knowledge: get_ivr_knowledge() filters by insurance_name ILIKE
CREATE INDEX IF NOT EXISTS idx_ivr_knowledge_insurance
    ON ivr_knowledge(insurance_name);

-- call_knowledge: search() and upsert_entry() filter by insurance_name ILIKE
CREATE INDEX IF NOT EXISTS idx_call_knowledge_insurance
    ON call_knowledge(insurance_name);

-- insurance_providers: provider lookups filter by insurance_name
CREATE INDEX IF NOT EXISTS idx_insurance_providers_name
    ON insurance_providers(insurance_name);
