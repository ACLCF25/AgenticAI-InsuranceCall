-- Migration: Add twilio_numbers table for dynamic phone number pool
-- Run this against your Supabase PostgreSQL database

CREATE TABLE IF NOT EXISTS twilio_numbers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phone_number VARCHAR(20) NOT NULL UNIQUE,
    friendly_name VARCHAR(100),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    current_call_id VARCHAR(100),
    current_call_sid VARCHAR(100),
    in_use_since TIMESTAMP,
    added_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_twilio_numbers_available
    ON twilio_numbers(is_active, current_call_id);

CREATE INDEX IF NOT EXISTS idx_twilio_numbers_call_id
    ON twilio_numbers(current_call_id) WHERE current_call_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_twilio_numbers_call_sid
    ON twilio_numbers(current_call_sid) WHERE current_call_sid IS NOT NULL;

GRANT ALL ON twilio_numbers TO authenticated;
