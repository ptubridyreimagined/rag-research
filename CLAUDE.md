# CLAUDE.md

## Project overview

This is a drop-in RAG system for Claude Code. It provides hybrid retrieval (BM25 + dense embeddings), cross-encoder reranking, and automatic context management via hooks. The system is designed to be copied into any existing project repo.

## Key architecture

- `rag/engine.py` — core module: indexing, retrieval, reranking, context assembly, CLI
- `rag/chunker.py` — semantic/fixed/hybrid chunking with AST-aware code splitting
- `rag/config.py` — loads `.rag/config.yaml`, provides `RAGConfig` dataclass
- `.claude/hooks/` — 5 hooks that fire on session start/end and file read/write
- `.claude/commands/` — 6 slash commands (`/rag-search`, `/rag-reindex`, etc.)
- `.claude/skills/RAG.md` — teaches Claude how to use the RAG system

## Commands

- `python rag/engine.py --index` — full reindex
- `python rag/engine.py --query "search terms"` — test retrieval
- `python rag/engine.py --status` — index health check
- `python rag/engine.py --reindex` — reindex only stale files
- `python rag/engine.py --staleness` — list stale files
- `python rag/engine.py --forget <path>` — remove path from index
- `python rag-setup.py` — re-run configuration wizard

## Do not modify

- `.rag/index/` — ChromaDB binary store. Never edit directly. Use `--index` or `--reindex` to rebuild.
- `.rag/chunks/chunks.json` — generated cache. Rebuilt on index.
- `.rag/logs/retrieval.log` — auto-managed, capped at 50 entries.

## Safe to edit

- `.rag/config.yaml` — all RAG settings. Re-run `python rag-setup.py` for guided changes.
- `.rag/context/PROJECT.md` — auto-generated but human-editable. Will not be overwritten unless explicitly re-generated.
- `.rag/context/SESSION.md` — updated by hooks at session end, but safe to edit mid-session.
- `.rag/context/DECISIONS.md` — append-only. Add entries via `/rag-decision` or manually.
- `.rag/prompts/` — prompt templates injected by hooks. Edit to change retrieval behavior.

## Code conventions

- Python 3.10+, no version-specific tricks
- All shell scripts must be POSIX-compatible
- Heavy imports (`chromadb`, `sentence_transformers`) are lazy-loaded — keep top-level imports light
- Every file managed by the RAG system has a `# rag-managed: true` or `<!-- rag-managed: true -->` header
- Token budget enforcement is strict — never allow context assembly to exceed `context_budget_tokens`

## Testing changes

There is no test suite. Verify changes with:
```
python rag/engine.py --index
python rag/engine.py --query "how does retrieval work"
python rag/engine.py --status
```
A successful query should return 1-5 results with scores and an assembled context token count.

## Dependencies

`sentence-transformers`, `chromadb`, `rank-bm25`, `PyYAML`. Optional: `tiktoken` for precise token counting (falls back to character-based estimation).

## Gotchas

- Changing `embedding_model` in config requires deleting `.rag/index/` and running a full `--index` — old vectors are incompatible with new model embeddings.
- On Windows, the console may fail on unicode box-drawing characters in retrieval previews. The engine handles this with an ascii fallback.
- Hook scripts use `python3` — on Windows this may need to be `python`. The hooks fall back gracefully if the command isn't found.
