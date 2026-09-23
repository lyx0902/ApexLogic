ALTER TABLE conversation_turns
    ADD COLUMN result_json jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE report_versions (
    version_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs(run_id),
    operation_turn_id text NOT NULL UNIQUE REFERENCES conversation_turns(turn_id),
    parent_version_id text REFERENCES report_versions(version_id),
    kind text NOT NULL CHECK (kind IN ('update','rewrite')),
    report_markdown text NOT NULL,
    source_manifest jsonb NOT NULL DEFAULT '[]'::jsonb,
    source_checkpoint_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX report_versions_run ON report_versions(run_id,created_at);
