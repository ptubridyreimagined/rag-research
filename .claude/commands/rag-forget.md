<!-- rag-managed: true -->
# /rag-forget

Remove a file or directory from the RAG index.

Run this command:
```bash
python rag/engine.py --forget "$ARGUMENTS"
```

After removing:
1. Confirm how many chunks were removed
2. Explain that the path will not be re-indexed until the exclude patterns are updated or the file is explicitly re-added
3. Suggest adding the path to `exclude_patterns` in `.rag/config.yaml` if it should be permanently excluded
