-- Add recording_type column to call_recordings to distinguish AI vs agent recordings
ALTER TABLE call_recordings
  ADD COLUMN IF NOT EXISTS recording_type VARCHAR(20) DEFAULT 'ai';

-- Add index for filtering by recording type
CREATE INDEX IF NOT EXISTS idx_call_recordings_type ON call_recordings(recording_type);
