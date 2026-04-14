-- Autonomous AI Insurance Credentialing System - Database Schema
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
-- Enable pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS "vector";

-- Credentialing Requests Table
CREATE TABLE credentialing_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    initiated_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
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
    call_mode VARCHAR(10) DEFAULT 'ai' CHECK (call_mode IN ('ai', 'agent')),
    agent_phone VARCHAR(20),
    conference_sid VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
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
    timestamp TIMESTAMPTZ DEFAULT NOW(),
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
    timestamp TIMESTAMPTZ DEFAULT NOW(),
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
    ivr_npi_suffix VARCHAR(5) DEFAULT NULL,      -- NULL, '*', '#' — termination key after NPI DTMF
    ivr_tax_id_suffix VARCHAR(5) DEFAULT NULL,   -- NULL, '*', '#' — termination key after Tax ID DTMF
    CONSTRAINT chk_ivr_tax_id_digits_to_send
        CHECK (ivr_tax_id_digits_to_send IS NULL OR ivr_tax_id_digits_to_send BETWEEN 1 AND 9),
    CONSTRAINT chk_ivr_npi_suffix
        CHECK (ivr_npi_suffix IS NULL OR ivr_npi_suffix IN ('*', '#')),
    CONSTRAINT chk_ivr_tax_id_suffix
        CHECK (ivr_tax_id_suffix IS NULL OR ivr_tax_id_suffix IN ('*', '#')),
    last_updated TIMESTAMP DEFAULT NOW(),
    INDEX idx_insurance_name (insurance_name)
);

-- Existing environments migration notes:
-- ALTER TABLE insurance_providers ADD COLUMN IF NOT EXISTS ivr_tax_id_digits_to_send INTEGER;
-- ALTER TABLE insurance_providers
--   ADD CONSTRAINT chk_ivr_tax_id_digits_to_send
--   CHECK (ivr_tax_id_digits_to_send IS NULL OR ivr_tax_id_digits_to_send BETWEEN 1 AND 9);
-- ALTER TABLE insurance_providers ADD COLUMN IF NOT EXISTS ivr_npi_suffix VARCHAR(5) DEFAULT NULL;
-- ALTER TABLE insurance_providers ADD COLUMN IF NOT EXISTS ivr_tax_id_suffix VARCHAR(5) DEFAULT NULL;

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

-- Row Level Security is enabled on these tables for future multi-tenant support.
-- Currently no RLS policies are defined — this is intentional for the single-tenant
-- deployment. All authenticated users can access all rows via the GRANT statements below.
-- When multi-tenant isolation is needed, add CREATE POLICY statements here.
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
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS initiated_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;

-- Migration: Add call mode and agent transfer columns
-- Run these on existing databases that already have the credentialing_requests table.
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS call_mode VARCHAR(10) DEFAULT 'ai' CHECK (call_mode IN ('ai', 'agent'));
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS agent_phone VARCHAR(20);
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS conference_sid VARCHAR(100);

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

-- =============================================================================
-- Supabase Auth Profile Tables
-- =============================================================================

CREATE TYPE user_role AS ENUM ('super_admin', 'admin', 'agent');
CREATE TYPE approval_status AS ENUM ('pending', 'approved', 'rejected');

CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    username VARCHAR(50) NOT NULL UNIQUE,
    role user_role NOT NULL DEFAULT 'agent',
    approval_status approval_status NOT NULL DEFAULT 'pending',
    approved_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    approved_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_user_profiles_role ON user_profiles(role);
CREATE INDEX idx_user_profiles_approval_status ON user_profiles(approval_status);

CREATE OR REPLACE FUNCTION public.handle_new_auth_user()
RETURNS TRIGGER AS $$
DECLARE
    requested_username TEXT;
BEGIN
    requested_username := NULLIF(TRIM(COALESCE(NEW.raw_user_meta_data ->> 'username', '')), '');
    IF requested_username IS NULL THEN
        requested_username := split_part(NEW.email, '@', 1);
    END IF;

    INSERT INTO public.user_profiles (user_id, username, role, approval_status)
    VALUES (NEW.id, requested_username, 'agent', 'pending');

    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_auth_user();

GRANT ALL ON user_profiles TO authenticated;

-- =============================================================================
-- Legacy Custom Auth Tables (deprecated)
-- =============================================================================

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role user_role NOT NULL DEFAULT 'agent',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);

-- Token blacklist for logout/revocation
CREATE TABLE token_blacklist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    jti VARCHAR(36) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_token_blacklist_jti ON token_blacklist(jti);
CREATE INDEX idx_token_blacklist_expires ON token_blacklist(expires_at);

GRANT ALL ON users TO authenticated;
GRANT ALL ON token_blacklist TO authenticated;

-- =============================================================================
-- Twilio Phone Number Pool
-- =============================================================================

CREATE TABLE twilio_numbers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_number VARCHAR(20) NOT NULL UNIQUE,
    friendly_name VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    current_call_id VARCHAR(100),
    current_call_sid VARCHAR(100),
    in_use_since TIMESTAMP,
    added_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_twilio_numbers_available ON twilio_numbers(is_active, current_call_id);
CREATE INDEX idx_twilio_numbers_call_id ON twilio_numbers(current_call_id) WHERE current_call_id IS NOT NULL;
CREATE INDEX idx_twilio_numbers_call_sid ON twilio_numbers(current_call_sid) WHERE current_call_sid IS NOT NULL;

GRANT ALL ON twilio_numbers TO authenticated;

-- =============================================================================
-- Call Recording and Q&A Tables
-- =============================================================================

-- Call Recordings Table
CREATE TABLE call_recordings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id VARCHAR(100) NOT NULL,
    request_id UUID REFERENCES credentialing_requests(id),
    call_sid VARCHAR(100) NOT NULL,
    recording_sid VARCHAR(100) NOT NULL UNIQUE,
    recording_url TEXT NOT NULL,
    recording_duration INTEGER,  -- Duration in seconds
    recording_status VARCHAR(50) DEFAULT 'processing',  -- 'processing', 'completed', 'failed', 'expired'
    recording_format VARCHAR(10) DEFAULT 'mp3',  -- 'mp3', 'wav'
    file_size INTEGER,  -- Size in bytes
    recording_type VARCHAR(20) DEFAULT 'ai',  -- 'ai', 'agent', or 'both'
    downloaded BOOLEAN DEFAULT FALSE,  -- If stored locally/S3
    local_path TEXT,  -- Optional: local/S3 storage path
    retention_until TIMESTAMPTZ,  -- For automatic cleanup
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for call_recordings
CREATE INDEX idx_call_recordings_call_id ON call_recordings(call_id);
CREATE INDEX idx_call_recordings_recording_sid ON call_recordings(recording_sid);
CREATE INDEX idx_call_recordings_request_id ON call_recordings(request_id);
CREATE INDEX idx_call_recordings_retention ON call_recordings(retention_until);

-- Call Q&A Pairs Table
CREATE TABLE call_qa_pairs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id VARCHAR(100) NOT NULL,
    request_id UUID REFERENCES credentialing_requests(id),
    question_index INTEGER NOT NULL,  -- Which question from the questions array
    question_text TEXT NOT NULL,
    answer_text TEXT,
    confidence FLOAT DEFAULT 0.0,  -- Confidence in extraction (0.0-1.0)
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    extraction_method VARCHAR(50) DEFAULT 'gpt4',  -- 'gpt4', 'manual', 'rule-based'
    conversation_snippet JSONB,  -- Related conversation context
    verified BOOLEAN DEFAULT FALSE,  -- Manual verification flag
    notes TEXT,
    UNIQUE (call_id, question_index)
);

-- Create indexes for call_qa_pairs
CREATE INDEX idx_call_qa_pairs_call_id ON call_qa_pairs(call_id);
CREATE INDEX idx_call_qa_pairs_request_id ON call_qa_pairs(request_id);
CREATE INDEX idx_call_qa_pairs_question_index ON call_qa_pairs(question_index);

-- Enhance conversation_history table
ALTER TABLE conversation_history
    ADD COLUMN IF NOT EXISTS conversation_turn INTEGER,
    ADD COLUMN IF NOT EXISTS is_question BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS is_answer BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS related_qa_id UUID REFERENCES call_qa_pairs(id),
    ADD COLUMN IF NOT EXISTS audio_timestamp FLOAT;

-- Enhance call_metrics table
ALTER TABLE call_metrics
    ADD COLUMN IF NOT EXISTS recording_sid VARCHAR(100),
    ADD COLUMN IF NOT EXISTS recording_available BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS qa_pairs_extracted INTEGER DEFAULT 0;

-- Grant permissions
GRANT ALL ON call_recordings TO authenticated;
GRANT ALL ON call_qa_pairs TO authenticated;

-- Function to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_call_recording_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_call_recording_timestamp
BEFORE UPDATE ON call_recordings
FOR EACH ROW
EXECUTE FUNCTION update_call_recording_timestamp();

-- =============================================================================
-- Self-Learning Human Detection Phrases
-- =============================================================================

CREATE TABLE human_detection_phrases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phrase TEXT NOT NULL,
    phrase_type VARCHAR(20) NOT NULL CHECK (phrase_type IN ('human', 'ivr_definitive', 'ivr_passive', 'simple_greeting')),
    insurance_name VARCHAR(255),          -- NULL = global, non-null = insurance-specific
    source VARCHAR(50) NOT NULL DEFAULT 'manual',  -- 'manual', 'auto_review', 'feedback'
    confidence FLOAT DEFAULT 0.8,
    times_seen INTEGER DEFAULT 1,
    times_correct INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    source_call_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_hdp_unique_phrase
    ON human_detection_phrases(phrase, phrase_type, COALESCE(insurance_name, ''));
CREATE INDEX idx_hdp_type_active ON human_detection_phrases(phrase_type, is_active);
CREATE INDEX idx_hdp_insurance ON human_detection_phrases(insurance_name);

GRANT ALL ON human_detection_phrases TO authenticated;

-- Migration: Add human detection feedback columns to credentialing_requests
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS human_detection_correct BOOLEAN;
-- ALTER TABLE credentialing_requests ADD COLUMN IF NOT EXISTS human_detection_feedback_at TIMESTAMP;
