CREATE TABLE research_jobs (
    run_id text PRIMARY KEY REFERENCES runs(run_id),
    status text NOT NULL CHECK (status IN ('queued','running','completed','cancelled')),
    cancel_requested boolean NOT NULL DEFAULT false,
    lease_owner text,
    lease_until timestamptz,
    last_node text,
    last_error text,
    updated_at timestamptz NOT NULL
);
CREATE INDEX research_jobs_claim ON research_jobs(status,lease_until,updated_at);
CREATE TABLE research_outbox (
    id bigserial PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs(run_id),
    created_at timestamptz NOT NULL,
    delivered_at timestamptz
);
CREATE INDEX research_outbox_pending ON research_outbox(id) WHERE delivered_at IS NULL;
