CREATE TABLE intent_decision_events (
    event_id text PRIMARY KEY,
    run_id text NOT NULL REFERENCES runs(run_id),
    turn_id text REFERENCES conversation_turns(turn_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    selection_mode text NOT NULL CHECK (selection_mode IN ('auto','manual')),
    decision_route text NOT NULL CHECK (decision_route IN ('rule','jev','llm','manual')),
    final_intent text NOT NULL,
    jev_provider text,
    jev_model text,
    jev_status text NOT NULL CHECK (jev_status IN
        ('not_attempted','not_sent','request_failed','http_error','invalid_response','success')),
    jev_call_succeeded boolean NOT NULL DEFAULT false,
    jev_accepted boolean NOT NULL DEFAULT false,
    jev_choice text,
    jev_confidence double precision CHECK (jev_confidence BETWEEN 0 AND 1),
    follow_up_probability double precision CHECK (follow_up_probability BETWEEN 0 AND 1),
    update_probability double precision CHECK (update_probability BETWEEN 0 AND 1),
    rewrite_probability double precision CHECK (rewrite_probability BETWEEN 0 AND 1),
    verify_probability double precision CHECK (verify_probability BETWEEN 0 AND 1),
    jev_probabilities jsonb NOT NULL DEFAULT '{}'::jsonb,
    jev_error_type text,
    jev_latency_ms double precision CHECK (jev_latency_ms >= 0),
    llm_status text NOT NULL CHECK (llm_status IN ('not_called','success','error')),
    llm_latency_ms double precision CHECK (llm_latency_ms >= 0)
);
CREATE INDEX intent_decision_events_run ON intent_decision_events(run_id,created_at DESC);
CREATE INDEX intent_decision_events_turn ON intent_decision_events(turn_id);
