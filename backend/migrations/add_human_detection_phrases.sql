-- Self-Learning Human Detection Phrases
-- Stores learned indicator phrases from call feedback to improve future human detection.

CREATE TABLE IF NOT EXISTS human_detection_phrases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    phrase TEXT NOT NULL,
    phrase_type VARCHAR(20) NOT NULL CHECK (phrase_type IN ('human', 'ivr_definitive', 'ivr_passive', 'simple_greeting')),
    insurance_name VARCHAR(255),          -- NULL = global, non-null = insurance-specific
    source VARCHAR(50) NOT NULL DEFAULT 'manual',  -- 'manual', 'auto_review', 'feedback'
    confidence FLOAT DEFAULT 0.8,
    times_seen INTEGER DEFAULT 1,
    times_correct INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    source_call_id VARCHAR(100),          -- call that triggered the learning
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_hdp_unique_phrase
    ON human_detection_phrases(phrase, phrase_type, COALESCE(insurance_name, ''));
CREATE INDEX IF NOT EXISTS idx_hdp_type_active ON human_detection_phrases(phrase_type, is_active);
CREATE INDEX IF NOT EXISTS idx_hdp_insurance ON human_detection_phrases(insurance_name);

GRANT ALL ON human_detection_phrases TO authenticated;

-- Add feedback columns to credentialing_requests
ALTER TABLE credentialing_requests
    ADD COLUMN IF NOT EXISTS human_detection_correct BOOLEAN,
    ADD COLUMN IF NOT EXISTS human_detection_feedback_at TIMESTAMP;
