-- Autonomous AI Insurance Credentialing System - Database Schema
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Enable pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "vector";

-- Credentialing Requests Table
CREATE TABLE credentialing_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    insurance_name VARCHAR(255) NOT NULL,
    provider_name VARCHAR(255) NOT NULL,
    npi VARCHAR(10) NOT NULL,
    tax_id VARCHAR(20) NOT NULL,
    address TEXT NOT NULL,
    insurance_phone VARCHAR(20),
    provider_phone VARCHAR(20),
    questions JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'initiated',
    reference_number VARCHAR(100),
    missing_documents JSONB DEFAULT '[]'::jsonb,
    turnaround_days INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_insurance_name (insurance_name),
    INDEX idx_created_at (created_at)
);

-- IVR Knowledge Base Table
CREATE TABLE ivr_knowledge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    insurance_name VARCHAR(255) NOT NULL,
    menu_level INTEGER DEFAULT 1,
    detected_phrase TEXT NOT NULL,
    preferred_action VARCHAR(20) NOT NULL, -- 'dtmf', 'speech', 'wait'
    action_value VARCHAR(100),
    confidence_threshold FLOAT DEFAULT 0.7,
    attempts INTEGER DEFAULT 0,
    successes INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    last_success TIMESTAMP,
    last_updated TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    INDEX idx_insurance_menu (insurance_name, menu_level),
    INDEX idx_success_rate (success_rate DESC),
    INDEX idx_last_updated (last_updated DESC)
);

-- Call Events Log Table
CREATE TABLE call_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'ivr_menu', 'hold', 'human_speech', 'action_taken'
    transcript TEXT,
    action_taken VARCHAR(50),
    confidence FLOAT,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    INDEX idx_call_id (call_id),
    INDEX idx_event_type (event_type),
    INDEX idx_timestamp (timestamp DESC)
);

-- Conversation History Table
CREATE TABLE conversation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id VARCHAR(100) NOT NULL,
    request_id UUID REFERENCES credentialing_requests(id),
    speaker VARCHAR(20) NOT NULL, -- 'agent', 'representative', 'ivr'
    message TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    INDEX idx_call_id (call_id),
    INDEX idx_request_id (request_id),
    INDEX idx_timestamp (timestamp)
);

-- Scheduled Follow-ups Table
CREATE TABLE scheduled_followups (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID REFERENCES credentialing_requests(id),
    scheduled_date TIMESTAMP NOT NULL,
    action_type VARCHAR(50) NOT NULL, -- 'retry_call', 'follow_up_call', 'submit_documents_then_call'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'completed', 'failed'
    created_at TIMESTAMP DEFAULT NOW(),
    executed_at TIMESTAMP,
    INDEX idx_scheduled_date (scheduled_date),
    INDEX idx_status (status),
    INDEX idx_request_id (request_id)
);

-- Call Performance Metrics Table
CREATE TABLE call_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id VARCHAR(100) NOT NULL UNIQUE,
    request_id UUID REFERENCES credentialing_requests(id),
    duration_seconds INTEGER,
    ivr_navigation_time_seconds INTEGER,
    hold_time_seconds INTEGER,
    human_interaction_time_seconds INTEGER,
    successful BOOLEAN,
    failure_reason TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_call_id (call_id),
    INDEX idx_successful (successful),
    INDEX idx_created_at (created_at DESC)
);

-- Insurance Provider Directory Table
CREATE TABLE insurance_providers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    insurance_name VARCHAR(255) NOT NULL UNIQUE,
    phone_number VARCHAR(20) NOT NULL,
    department VARCHAR(100),
    best_call_times JSONB, -- e.g., {"days": ["monday", "tuesday"], "hours": [9, 17]}
    average_wait_time_minutes INTEGER,
    typical_hold_music TEXT,
    notes TEXT,
    ivr_asks_npi BOOLEAN DEFAULT FALSE,
    ivr_npi_method VARCHAR(10) DEFAULT 'speech',  -- 'speech' or 'dtmf'
    ivr_asks_tax_id BOOLEAN DEFAULT FALSE,
    ivr_tax_id_method VARCHAR(10) DEFAULT 'speech',  -- 'speech' or 'dtmf'
    ivr_tax_id_digits_to_send INTEGER,
    CONSTRAINT chk_ivr_tax_id_digits_to_send
        CHECK (ivr_tax_id_digits_to_send IS NULL OR ivr_tax_id_digits_to_send BETWEEN 1 AND 9),
    last_updated TIMESTAMP DEFAULT NOW(),
    INDEX idx_insurance_name (insurance_name)
);

-- Existing environments migration notes:
-- ALTER TABLE insurance_providers ADD COLUMN IF NOT EXISTS ivr_tax_id_digits_to_send INTEGER;
-- ALTER TABLE insurance_providers
--   ADD CONSTRAINT chk_ivr_tax_id_digits_to_send
--   CHECK (ivr_tax_id_digits_to_send IS NULL OR ivr_tax_id_digits_to_send BETWEEN 1 AND 9);

-- System Configuration Table
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) NOT NULL UNIQUE,
    config_value JSONB NOT NULL,
    description TEXT,
    last_updated TIMESTAMP DEFAULT NOW()
);

-- Audit Log Table for Compliance
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX idx_user_id (user_id),
    INDEX idx_action (action),
    INDEX idx_timestamp (timestamp DESC)
);

-- Create Views for Analytics
CREATE VIEW credentialing_success_rate AS
SELECT 
    insurance_name,
    COUNT(*) as total_requests,
    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
    SUM(CASE WHEN status IN ('pending_review', 'missing_documents') THEN 1 ELSE 0 END) as in_progress,
    SUM(CASE WHEN status = 'denied' THEN 1 ELSE 0 END) as denied,
    ROUND(100.0 * SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) / COUNT(*), 2) as success_rate
FROM credentialing_requests
GROUP BY insurance_name;

CREATE VIEW daily_performance_metrics AS
SELECT 
    DATE(cm.created_at) as date,
    COUNT(DISTINCT cm.call_id) as total_calls,
    SUM(CASE WHEN cm.successful THEN 1 ELSE 0 END) as successful_calls,
    ROUND(AVG(cm.duration_seconds) / 60.0, 2) as avg_duration_minutes,
    ROUND(AVG(cm.hold_time_seconds) / 60.0, 2) as avg_hold_time_minutes,
    SUM(CASE WHEN cm.retry_count > 0 THEN 1 ELSE 0 END) as calls_requiring_retry
FROM call_metrics cm
GROUP BY DATE(cm.created_at)
ORDER BY date DESC;

-- Insert Sample IVR Knowledge (Common Patterns)
INSERT INTO ivr_knowledge (insurance_name, menu_level, detected_phrase, preferred_action, action_value, success_rate) VALUES
('Generic Insurance', 1, 'press 1 for provider services', 'dtmf', '1', 0.95),
('Generic Insurance', 1, 'say provider for provider services', 'speech', 'provider', 0.90),
('Generic Insurance', 2, 'press 2 for credentialing', 'dtmf', '2', 0.92),
('Generic Insurance', 0, 'please hold', 'wait', NULL, 1.0);

-- Insert Sample System Configuration
INSERT INTO system_config (config_key, config_value, description) VALUES
('max_hold_time_minutes', '30', 'Maximum time to wait on hold before hanging up'),
('max_retry_attempts', '3', 'Maximum number of retry attempts per call'),
('business_hours_start', '8', 'Business hours start (24h format)'),
('business_hours_end', '17', 'Business hours end (24h format)'),
('ai_temperature', '0.3', 'OpenAI temperature for call decision making'),
('disclosure_message', '"Hello, this is an automated assistant calling on behalf of [provider_name]."', 'Initial disclosure message');

-- Create Row Level Security Policies (Optional - enable if multi-tenant)
ALTER TABLE credentialing_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE ivr_knowledge ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_events ENABLE ROW LEVEL SECURITY;

-- Grant permissions to authenticated users
-- Adjust these based on your authentication setup
GRANT ALL ON credentialing_requests TO authenticated;
GRANT ALL ON ivr_knowledge TO authenticated;
GRANT ALL ON call_events TO authenticated;
GRANT ALL ON conversation_history TO authenticated;
GRANT ALL ON scheduled_followups TO authenticated;
GRANT ALL ON call_metrics TO authenticated;
GRANT ALL ON insurance_providers TO authenticated;

-- Knowledge base table for prior call summaries (pgvector)
CREATE TABLE call_knowledge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    insurance_name VARCHAR(255) NOT NULL,
    provider_name VARCHAR(255),
    call_id VARCHAR(100),
    request_id UUID REFERENCES credentialing_requests(id),
    summary TEXT NOT NULL,
    qa_text TEXT,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    chunk_index INTEGER DEFAULT 0,
    parent_id UUID REFERENCES call_knowledge(id),
    total_chunks INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Migration: Add chunking columns if table already exists
-- ALTER TABLE call_knowledge ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0;
-- ALTER TABLE call_knowledge ADD COLUMN IF NOT EXISTS parent_id UUID REFERENCES call_knowledge(id);
-- ALTER TABLE call_knowledge ADD COLUMN IF NOT EXISTS total_chunks INTEGER DEFAULT 1;

-- Migration: Add provider_phone column if table already exists
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS provider_phone VARCHAR(20);

-- Indexes for fast search
CREATE INDEX IF NOT EXISTS call_knowledge_insurance_idx ON call_knowledge (insurance_name);
CREATE INDEX IF NOT EXISTS call_knowledge_provider_idx ON call_knowledge (provider_name);
-- ivfflat index requires list size configured; adjust lists per data volume
CREATE INDEX IF NOT EXISTS call_knowledge_embedding_idx ON call_knowledge USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
GRANT ALL ON audit_log TO authenticated;
GRANT SELECT ON credentialing_success_rate TO authenticated;
GRANT SELECT ON daily_performance_metrics TO authenticated;

-- Functions for automatic updates
CREATE OR REPLACE FUNCTION update_ivr_success_rate()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.attempts > 0 THEN
        NEW.success_rate := NEW.successes::float / NEW.attempts::float;
    END IF;
    NEW.last_updated := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_ivr_success_rate
BEFORE UPDATE ON ivr_knowledge
FOR EACH ROW
EXECUTE FUNCTION update_ivr_success_rate();

-- Function to automatically log audit events
CREATE OR REPLACE FUNCTION log_audit_event()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (action, resource_type, resource_id, details)
    VALUES (TG_OP, TG_TABLE_NAME, NEW.id::text, to_jsonb(NEW));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add audit triggers to sensitive tables
CREATE TRIGGER trigger_audit_credentialing_requests
AFTER INSERT OR UPDATE OR DELETE ON credentialing_requests
FOR EACH ROW
EXECUTE FUNCTION log_audit_event();
