ALTER TABLE intent_decision_events
    ADD COLUMN jev_http_status integer CHECK (jev_http_status BETWEEN 100 AND 599),
    ADD COLUMN jev_provider_error_type text;
