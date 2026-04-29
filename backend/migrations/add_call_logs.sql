CREATE TABLE IF NOT EXISTS call_logs (
    id            BIGSERIAL PRIMARY KEY,
    call_id       TEXT        NOT NULL,
    call_sid      TEXT,
    logged_at     TIMESTAMPTZ NOT NULL,
    level         TEXT        NOT NULL,
    logger_name   TEXT,
    function_name TEXT,
    line_number   INTEGER,
    message       TEXT        NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_call_logs_call_id   ON call_logs(call_id);
CREATE INDEX IF NOT EXISTS idx_call_logs_call_sid  ON call_logs(call_sid);
CREATE INDEX IF NOT EXISTS idx_call_logs_logged_at ON call_logs(logged_at);
