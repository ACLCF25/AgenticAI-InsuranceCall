-- Normalize call-facing timestamp columns to TIMESTAMPTZ.
-- Existing naive values are interpreted as UTC to preserve stored wall-clock times.

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
