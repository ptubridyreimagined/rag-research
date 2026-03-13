<!-- rag-managed: true -->
# Session Start Context

Load the following in order:
1. **PROJECT.md** — project overview and structure
2. **SESSION.md** — state from the last session
3. **DECISIONS.md** — architectural decisions to respect
4. **Retrieved chunks** — relevant code for the last known task

## Guidelines
- Greet the user and briefly summarize what was happening last session
- If SESSION.md shows open questions, proactively address them
- If the project index is >20% stale, suggest running `/rag-reindex`
- Prefer retrieved context over assumptions about the codebase
