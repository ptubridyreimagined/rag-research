# RAG System for Claude Code

A drop-in retrieval-augmented generation system that gives Claude Code
persistent, searchable knowledge of your entire codebase.

## Quickstart (3 commands)

```bash
git clone <your-repo> && cd <your-repo>
# Copy the rag/ directory and rag-init.sh into your project, then:
bash rag-init.sh
```

That's it. The installer detects your project type, installs dependencies,
asks 5 setup questions, builds the index, and registers Claude Code hooks.

Verify it works:
```bash
python rag/engine.py --query "main entry point"
```

## How It Works

```
  User query or file edit
         │
         ▼
  ┌─────────────┐     ┌──────────────┐
  │  BM25 Sparse │     │ Dense Embed  │
  │  (keyword)   │     │ (semantic)   │
  └──────┬───────┘     └──────┬───────┘
         │                    │
         ▼                    ▼
  ┌──────────────────────────────────┐
  │   Reciprocal Rank Fusion (RRF)   │
  │   Merge sparse + dense scores    │
  └──────────────┬───────────────────┘
                 │ top-20
                 ▼
  ┌──────────────────────────────────┐
  │   Cross-Encoder Reranking        │
  │   (ms-marco-MiniLM-L-6-v2)      │
  └──────────────┬───────────────────┘
                 │ top-3 to top-5
                 ▼
  ┌──────────────────────────────────┐
  │   Context Assembly               │
  │   Dedup → Trim → Order → Format  │
  │   Strict token budget enforced   │
  └──────────────┬───────────────────┘
                 │
                 ▼
  Injected into Claude's context window
```

**Chunking:** Files are split on structural boundaries — functions for code,
headers for markdown, paragraphs for prose. Small files (<400 tokens) are
indexed whole. Target chunk size: 256-512 tokens.

**Hybrid retrieval:** Every query runs against both a BM25 keyword index and
a dense embedding index (all-MiniLM-L6-v2). Scores are fused with RRF.

**Reranking:** Top-20 candidates are reranked with a cross-encoder for
precision. Only the top 3-5 make it into context.

**Budget enforcement:** The system never exceeds the configured token budget.
Chunks are trimmed or dropped to fit.

## Config Reference (.rag/config.yaml)

| Key | Type | Default | Effect |
|-----|------|---------|--------|
| `index_paths` | list[str] | `["."]` | Directories to index |
| `exclude_patterns` | list[str] | (common ignores) | Glob patterns to skip |
| `chunk_strategy` | str | `"semantic"` | `semantic`, `fixed`, or `hybrid` |
| `chunk_size_tokens` | int | `400` | Target tokens per chunk |
| `chunk_overlap_tokens` | int | `0` | Overlap between chunks (fixed strategy) |
| `embedding_model` | str | `"all-MiniLM-L6-v2"` | sentence-transformers model name |
| `reranker_model` | str | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` | Cross-encoder model |
| `context_budget_tokens` | int | `40000` | Max tokens for retrieved context |
| `retrieval_top_k` | int | `20` | Candidates from initial retrieval |
| `rerank_top_k` | int | `5` | Candidates after reranking |
| `inject_top_k` | int | `3` | Chunks injected into context |
| `min_similarity` | float | `0.25` | Minimum score to include a result |
| `session_type` | str | `"short"` | `short` (3 chunks) or `long` (5 chunks) |
| `watch_mode` | bool | `false` | Enable file-watch reindexing |
| `reranking_enabled` | bool | `true` | Toggle cross-encoder reranking |

### Swapping the embedding model

1. Edit `embedding_model` in `.rag/config.yaml`
2. Delete `.rag/index/` (old vectors are incompatible)
3. Run `python rag/engine.py --index` to rebuild

Good alternatives:
- `BAAI/bge-small-en-v1.5` — slightly better quality, similar speed
- `BAAI/bge-large-en-v1.5` — best quality, slower on CPU (~100ms)
- `nomic-ai/nomic-embed-text-v1.5` — strong on code

## Troubleshooting

### 1. "No relevant results found"
- **Cause:** Index is empty or query doesn't match indexed content
- **Fix:** Run `/rag-status` to check chunk count. Run `/rag-reindex` if stale.
  Try a more specific query with exact function/class names.

### 2. Retrieved context is outdated
- **Cause:** Index hasn't been updated since files changed
- **Fix:** Run `/rag-reindex`. The post-session-end hook auto-reindexes, but
  if you've made many changes mid-session, trigger it manually.

### 3. Retrieval is slow (>2 seconds)
- **Cause:** Large index + reranking overhead
- **Fix:** Set `reranking_enabled: false` in config for faster (but less
  precise) retrieval. Or reduce `retrieval_top_k` to 10.

### 4. Token budget exceeded errors
- **Cause:** Shouldn't happen — budget is strictly enforced
- **Fix:** If you see truncated context, increase `context_budget_tokens` in
  config. If Claude's total context is too large, reduce the budget.

### 5. Import errors when hooks run
- **Cause:** Python dependencies not installed or wrong Python version
- **Fix:** Run `pip install sentence-transformers chromadb rank-bm25 PyYAML`.
  Ensure `python3` points to Python 3.10+.

## Extending the System

### Adding a new hook

1. Create a script in `.claude/hooks/` (`.sh` or `.py`)
2. Add a header comment explaining the trigger and behavior
3. Register it in `.claude/settings.json` under the appropriate trigger
4. Available triggers: `PreToolUse`, `PostToolUse`, `PostSessionStart`, `PostSessionEnd`

### Adding a new command

1. Create a `.md` file in `.claude/commands/`
2. The filename becomes the command: `rag-foo.md` → `/rag-foo`
3. Use `$ARGUMENTS` to capture user input after the command name
4. Include bash code blocks that Claude will execute

### Adding a new context file

1. Create a `.md` file in `.rag/context/`
2. Add `<!-- rag-managed: true -->` as the first line
3. Reference it in `.claude/hooks/post-session-start.sh` to load at session start

## File Structure

```
.rag/
  config.yaml           # Configuration (tracked in git)
  index/                # ChromaDB store (gitignored)
  chunks/               # Chunk metadata cache (gitignored)
  context/
    PROJECT.md          # Auto-generated project summary (editable)
    SESSION.md          # Session state (updated by hooks)
    DECISIONS.md        # Architectural decisions log
  prompts/
    retrieval.md        # Injected with RAG results
    session-start.md    # Session loading instructions
  logs/
    retrieval.log       # Query → chunks log (gitignored)
```
