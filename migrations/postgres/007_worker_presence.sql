CREATE TABLE worker_targets (
    manager_id text PRIMARY KEY,
    desired_count integer NOT NULL CHECK (desired_count BETWEEN 0 AND 8),
    retry_after timestamptz,
    last_error text,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE worker_instances (
    worker_id text PRIMARY KEY,
    manager_id text,
    hostname text NOT NULL,
    pid integer NOT NULL,
    phase text NOT NULL CHECK (phase IN ('starting','online')),
    drain_requested boolean NOT NULL DEFAULT false,
    started_at timestamptz NOT NULL DEFAULT now(),
    heartbeat_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX worker_instances_live ON worker_instances(heartbeat_at,manager_id);
