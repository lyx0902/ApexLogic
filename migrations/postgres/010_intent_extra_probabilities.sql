ALTER TABLE intent_decision_events
    ADD COLUMN new_research_probability double precision CHECK (new_research_probability BETWEEN 0 AND 1),
    ADD COLUMN clarify_probability double precision CHECK (clarify_probability BETWEEN 0 AND 1);

-- Older seven-option responses already retained the full distribution in JSONB.
UPDATE intent_decision_events
SET new_research_probability = (jev_probabilities->>'new_research')::double precision,
    clarify_probability = (jev_probabilities->>'clarify')::double precision
WHERE jsonb_typeof(jev_probabilities->'new_research') = 'number'
   OR jsonb_typeof(jev_probabilities->'clarify') = 'number';
