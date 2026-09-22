"""Bounded retrieval audit; source identity survives ranking and citation renumbering."""
import hashlib
import json


def document_key(doc):
    if doc.get('memory_id'):
        return 'memory:' + doc['memory_id']
    values = [str(doc.get(k, '')) for k in ('url', 'content')]
    if not values[0]:
        values.append(str(doc.get('title', '')))
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def describe(doc):
    return {**{k: doc.get(k) for k in ('title', 'url', 'source', 'memory_id', 'cache_hit')},
            'document_key': document_key(doc), 'excerpt': str(doc.get('content', ''))[:600]}


def search_with_audit(records, provider, query, limit, search):
    record = {'provider': provider, 'query': query, 'requested_results': limit,
              'status': 'started', 'retrieved_docs': []}
    records.append(record)
    try:
        docs = search(query, max_results=limit)
        record.update(status='ok', retrieved_docs=[dict(describe(d), search_provider=provider, query=query)
                                                  for d in docs if isinstance(d, dict)])
        return docs
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__)
        raise


def annotate_selection(plan, selected):
    by_key = {document_key(d): d.get('citation_id') for d in selected}
    for sub in plan.get('sub_results', []):
        for doc in sub.get('retrieved_docs', []) + sub.get('memory_docs', []):
            # Legacy title/URL-only records cannot establish exact excerpt identity.
            if 'document_key' in doc:
                doc['selected'] = doc['document_key'] in by_key
                doc['citation_id'] = by_key.get(doc['document_key'])
