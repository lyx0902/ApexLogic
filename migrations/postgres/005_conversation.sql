CREATE TABLE conversation_turns (
    turn_seq bigserial UNIQUE,
    turn_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs(run_id),
    idempotency_key text NOT NULL UNIQUE,
    question text NOT NULL,
    request_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    answer text,
    citation_ids jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_checkpoint_id text,
    status text NOT NULL CHECK (status IN ('queued','running','completed','failed')),
    lease_owner text,
    lease_until timestamptz,
    attempts integer NOT NULL DEFAULT 0,
    last_error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX conversation_turns_run ON conversation_turns(run_id,turn_seq);
CREATE INDEX conversation_turns_claim ON conversation_turns(status,updated_at,lease_until);
