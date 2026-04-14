-- Normalize all timestamp columns to TIMESTAMPTZ.
-- Existing naive values are interpreted as UTC to preserve stored wall-clock times.

-- Call-facing tables (already applied)
ALTER TABLE credentialing_requests
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC',
    ALTER COLUMN completed_at TYPE TIMESTAMPTZ USING completed_at AT TIME ZONE 'UTC';

ALTER TABLE call_events
    ALTER COLUMN timestamp TYPE TIMESTAMPTZ USING timestamp AT TIME ZONE 'UTC';

ALTER TABLE conversation_history
    ALTER COLUMN timestamp TYPE TIMESTAMPTZ USING timestamp AT TIME ZONE 'UTC';

ALTER TABLE call_recordings
    ALTER COLUMN retention_until TYPE TIMESTAMPTZ USING retention_until AT TIME ZONE 'UTC',
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';

ALTER TABLE call_qa_pairs
    ALTER COLUMN extracted_at TYPE TIMESTAMPTZ USING extracted_at AT TIME ZONE 'UTC';

-- Remaining tables
ALTER TABLE ivr_knowledge
    ALTER COLUMN last_success TYPE TIMESTAMPTZ USING last_success AT TIME ZONE 'UTC',
    ALTER COLUMN last_updated TYPE TIMESTAMPTZ USING last_updated AT TIME ZONE 'UTC';

ALTER TABLE scheduled_followups
    ALTER COLUMN scheduled_date TYPE TIMESTAMPTZ USING scheduled_date AT TIME ZONE 'UTC',
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN executed_at TYPE TIMESTAMPTZ USING executed_at AT TIME ZONE 'UTC';

ALTER TABLE call_metrics
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';

ALTER TABLE insurance_providers
    ALTER COLUMN last_updated TYPE TIMESTAMPTZ USING last_updated AT TIME ZONE 'UTC';

ALTER TABLE system_config
    ALTER COLUMN last_updated TYPE TIMESTAMPTZ USING last_updated AT TIME ZONE 'UTC';

ALTER TABLE audit_log
    ALTER COLUMN timestamp TYPE TIMESTAMPTZ USING timestamp AT TIME ZONE 'UTC';

ALTER TABLE user_profiles
    ALTER COLUMN approved_at TYPE TIMESTAMPTZ USING approved_at AT TIME ZONE 'UTC',
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';

ALTER TABLE users
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN last_login TYPE TIMESTAMPTZ USING last_login AT TIME ZONE 'UTC';

ALTER TABLE token_blacklist
    ALTER COLUMN expires_at TYPE TIMESTAMPTZ USING expires_at AT TIME ZONE 'UTC',
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';

ALTER TABLE twilio_numbers
    ALTER COLUMN in_use_since TYPE TIMESTAMPTZ USING in_use_since AT TIME ZONE 'UTC',
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';

ALTER TABLE call_knowledge
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';

ALTER TABLE human_detection_phrases
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';

ALTER TABLE credentialing_requests
    ALTER COLUMN human_detection_feedback_at TYPE TIMESTAMPTZ USING human_detection_feedback_at AT TIME ZONE 'UTC';
