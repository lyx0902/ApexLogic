CREATE TABLE runs (
    run_id text PRIMARY KEY, thread_id text NOT NULL UNIQUE, topic text NOT NULL,
    status text NOT NULL, termination_reason text, created_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL, config_json jsonb NOT NULL, config_hash text NOT NULL,
    schema_version integer NOT NULL, workflow_version text NOT NULL,
    checkpoint_seen integer NOT NULL DEFAULT 0, last_error text,
    export_status text NOT NULL DEFAULT 'pending', completed_at timestamptz
);
CREATE TABLE attempts (
    attempt_id text PRIMARY KEY, run_id text NOT NULL REFERENCES runs(run_id),
    started_at timestamptz NOT NULL, ended_at timestamptz, status text NOT NULL,
    elapsed_seconds double precision, error text
);
CREATE INDEX attempts_run ON attempts(run_id,started_at);
CREATE TABLE memory_items (
    id text PRIMARY KEY, namespace text NOT NULL, url text NOT NULL,
    title text NOT NULL, content text NOT NULL, claim text NOT NULL,
    topic text NOT NULL, content_hash text NOT NULL, source_run_id text NOT NULL,
    observed_at timestamptz NOT NULL, valid_until timestamptz NOT NULL,
    status text NOT NULL DEFAULT 'active', version integer NOT NULL DEFAULT 1,
    UNIQUE(namespace,url,content_hash,version)
);
CREATE INDEX memory_scope ON memory_items(namespace,status,valid_until);
CREATE TABLE memory_embeddings (
    item_id text NOT NULL REFERENCES memory_items(id), model text NOT NULL,
    dimension integer NOT NULL CHECK(dimension>0), vector vector NOT NULL,
    PRIMARY KEY(item_id,model), CHECK(vector_dims(vector)=dimension)
);
CREATE TABLE memory_publications (
    run_id text NOT NULL, checkpoint_id text NOT NULL, namespace text NOT NULL,
    status text NOT NULL, item_ids jsonb NOT NULL, error text,
    updated_at timestamptz NOT NULL, skip_reason text,
    PRIMARY KEY(run_id,checkpoint_id,namespace)
);
CREATE TABLE memory_relations (
    left_id text NOT NULL REFERENCES memory_items(id),
    right_id text NOT NULL REFERENCES memory_items(id), relation text NOT NULL,
    reason text NOT NULL, created_at timestamptz NOT NULL,
    PRIMARY KEY(left_id,right_id,relation)
);
CREATE TABLE memory_access_log (
    query_id text PRIMARY KEY, run_id text NOT NULL, namespace text NOT NULL,
    details jsonb NOT NULL, updated_at timestamptz NOT NULL
);
CREATE TABLE memory_imports (
    source_digest text PRIMARY KEY, counts jsonb NOT NULL, imported_at timestamptz NOT NULL DEFAULT now()
);
