<!-- rag-managed: true -->
# /rag-reindex

Reindex all stale files in the RAG index and report what changed.

Run these commands in sequence:
```bash
python rag/engine.py --staleness
python rag/engine.py --reindex
python rag/engine.py --status
```

After reindexing:
1. Report how many files were reindexed and how many were removed
2. Show the updated index stats
3. If no files were stale, confirm the index is up to date
