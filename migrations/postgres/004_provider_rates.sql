CREATE TABLE provider_rate_events (
    provider text NOT NULL,
    requested_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX provider_rate_events_recent ON provider_rate_events(provider, requested_at);
