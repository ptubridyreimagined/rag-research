<!-- rag-managed: true -->
# /rag-status

Show current RAG index statistics including staleness report.

Run this command:
```bash
python rag/engine.py --status
```

Present the results clearly:
- Total files and chunks indexed
- Staleness percentage (warn if >20%)
- Embedding model and chunk strategy in use
- Suggest `/rag-reindex` if staleness is high
