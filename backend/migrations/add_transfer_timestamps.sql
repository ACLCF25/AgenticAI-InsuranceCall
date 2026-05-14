-- Add transfer_started_at column to track when call transfer was initiated
ALTER TABLE credentialing_requests
  ADD COLUMN IF NOT EXISTS transfer_started_at TIMESTAMPTZ;

-- Create index for efficient queries on transfer_started_at
CREATE INDEX IF NOT EXISTS idx_credentialing_requests_transfer_started_at
  ON credentialing_requests(transfer_started_at)
  WHERE transfer_started_at IS NOT NULL;

-- Add comment documenting the column
COMMENT ON COLUMN credentialing_requests.transfer_started_at IS
  'Timestamp when transfer to human agent was initiated, populated when status set to "transferred"';
