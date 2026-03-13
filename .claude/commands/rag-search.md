<!-- rag-managed: true -->
# /rag-search

Search the RAG index and display the most relevant chunks with scores.

Run this command:
```bash
python rag/engine.py --query "$ARGUMENTS"
```

After running the search:
1. Display the results to the user with source file links
2. If scores are below 0.3, warn that confidence is low
3. If no results are found, suggest the user try a different query or run `/rag-reindex`
4. Offer to incorporate the retrieved context into the current task
