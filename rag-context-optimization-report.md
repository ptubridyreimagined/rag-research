# RAG Implementation Strategies for Context-Optimized LLM Workflows

**Target audience:** Developers and small teams using Claude or GPT-4 class models via API
**Focus:** Minimizing token waste, maximizing retrieval precision, CPU-friendly infrastructure

---

## 1. Chunking Strategies

**Summary:** Chunking is the highest-leverage decision in a RAG pipeline — bad boundaries produce fragments too incoherent for the LLM or chunks so large they waste context budget. Strategy choice depends on document structure and query patterns.

### Comparison

| Strategy | How it works | Retrieval precision | Coverage | Token efficiency | Best for |
|---|---|---|---|---|---|
| **Fixed-size** (e.g., 512 tokens with 50-token overlap) | Split text at token/character boundaries | Low — splits mid-thought, mid-paragraph | High — every token indexed | Medium — overlap wastes ~10% | Unstructured text with no clear sections; quick baseline |
| **Semantic** (paragraph/section-aware) | Split on natural boundaries: paragraphs, markdown headers, blank lines | High — preserves coherent units | High | High — no overlap needed | Markdown docs, codebases with docstrings, well-structured prose |
| **Hierarchical** (parent-child) | Index small chunks for retrieval, but return the parent chunk (or surrounding context) at query time | Very high — precise matching, broad context returned | Very high | Medium — parent chunks can be large | Documentation with nested structure; when you need precision + context |
| **Agentic** (LLM-assisted) | Use an LLM to decide chunk boundaries, write chunk summaries, or generate metadata | Highest — boundaries are semantically optimal | High | Low — LLM calls at index time are expensive | High-value corpora where index quality justifies compute cost |

### Decision tree

```
Is your content well-structured (markdown, code, HTML)?
├── Yes → Semantic chunking (split on headers/functions/paragraphs)
│         └── Need precise retrieval + broad context? → Add hierarchical parent-child
└── No (raw text, PDFs, transcripts)
    ├── Corpus is small + high-value? → Agentic chunking
    └── Otherwise → Fixed-size (256-512 tokens, 10-15% overlap)
```

### Key findings

- **Chunk size matters more than strategy.** 256-512 tokens is the empirical sweet spot. Below 128, chunks lose coherence; above 1024, they dilute embedding signal. (Consistent with MTEB evaluation ranges.)
- **Overlap is a band-aid.** If you need 25%+ overlap, your boundaries are wrong — switch to semantic chunking.
- **For code:** Use tree-sitter or AST parsing to chunk on function boundaries (signature + docstring + body as one unit). Index whole files under ~400 tokens. Regex splitting on `def`/`function` is brittle.
- **Hierarchical chunking is underused.** Index 128-token child chunks for retrieval, inject the 512-token parent into context — precise matching with sufficient context. Implementations: LlamaIndex `SentenceWindowNodeParser`, Langchain `ParentDocumentRetriever`.

### Recommendation

**Semantic chunking** on markdown headers or paragraph boundaries, targeting 256-512 tokens. For code, use tree-sitter for function boundaries. Add hierarchical parent-child if retrieval is precise but the LLM lacks surrounding context.

---

## 2. Embedding and Retrieval

**Summary:** Hybrid retrieval (dense + sparse) consistently outperforms either alone. Adding a cross-encoder reranker is the highest-ROI improvement (5-15% precision gain on benchmarks) if you can tolerate 50-200ms extra latency.

### Embedding model comparison

| Model | Dimensions | MTEB avg score | CPU-friendly | Cost | Notes |
|---|---|---|---|---|---|
| **OpenAI text-embedding-3-small** | 1536 (truncatable) | ~62 | Yes (API) | $0.02/1M tokens | Good default; Matryoshka support lets you truncate dims to 256-512 for storage savings |
| **OpenAI text-embedding-3-large** | 3072 (truncatable) | ~64 | Yes (API) | $0.13/1M tokens | Marginal improvement over small for most tasks |
| **Cohere embed-v3** | 1024 | ~65 | Yes (API) | Free tier available | Strong multilingual; supports `search_document` vs `search_query` input types |
| **sentence-transformers/all-MiniLM-L6-v2** | 384 | ~56 | Yes (CPU, ~15ms/query) | Free | Fastest local option; good enough for prototyping |
| **BAAI/bge-small-en-v1.5** | 384 | ~62 | Yes (CPU, ~20ms/query) | Free | Best local model for its size; competitive with API models |
| **BAAI/bge-large-en-v1.5** | 1024 | ~64 | Marginal (CPU ~100ms) | Free | Top-tier local; needs quantization for CPU comfort |
| **nomic-embed-text-v1.5** | 768 | ~62 | Yes (CPU, ~30ms) | Free | Matryoshka support; strong on code |

*Scores are approximate and vary by task. Check [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for current rankings.*

### Retrieval approach comparison

| Approach | Precision | Recall | Latency | Complexity | When to use |
|---|---|---|---|---|---|
| **Dense only** (cosine similarity on embeddings) | Medium-High | High | Low (~10ms at 100K docs) | Low | Default starting point |
| **Sparse only** (BM25) | High for keyword matches | Low for semantic queries | Low | Low | When queries contain exact terms (error messages, function names, IDs) |
| **Hybrid** (dense + BM25, score fusion) | High | Very High | Medium (~20ms) | Medium | Best general-purpose approach; recommended for production |
| **Hybrid + reranking** (cross-encoder on top-k results) | Very High | Very High | Higher (+50-200ms per query) | Medium-High | When precision matters more than latency; worth it for most RAG |

### Reranking

Cross-encoder rerankers (e.g., `ms-marco-MiniLM-L-6-v2`, Cohere Rerank, `bge-reranker-v2-m3`) score each (query, document) pair jointly — more accurate than embedding comparison, but O(n) in candidates.

**The practical pattern:** Retrieve top-20 to top-50 via hybrid (fast), rerank to top-3 to top-5 with a cross-encoder (accurate), inject only the reranked results. Keeps total latency under 300ms.

**Is it worth it?** Almost always. On BEIR benchmarks, reranking improves nDCG@10 by 10-20%. `ms-marco-MiniLM-L-6-v2` runs on CPU in ~50ms for 50 candidates and is free. Skip only if you need sub-100ms end-to-end.

### Recommendation

**`text-embedding-3-small` (OpenAI) or `bge-small-en-v1.5` (local)** for embeddings. Hybrid retrieval with BM25 + dense via RRF. Rerank with `ms-marco-MiniLM-L-6-v2`. Retrieve top-20, rerank to top-5, inject top-3.

---

## 3. Context Assembly

**Summary:** Chunk arrangement in the prompt matters as much as chunk selection. LLMs attend poorly to mid-context information (the "lost in the middle" effect), making ordering, deduplication, and trimming essential for signal-per-token efficiency.

### The lost-in-the-middle problem

Liu et al. (2023) showed that LLM performance on multi-document QA follows a U-shaped curve: information at the beginning and end of context is used reliably; information in the middle is often ignored — even well within the model's window limit.

**Mitigation strategies:**

| Strategy | Implementation | Effectiveness | Complexity |
|---|---|---|---|
| **Relevance-first ordering** | Place the most relevant chunk first, then descending | High — ensures best evidence is in the high-attention zone | Low |
| **Bookend pattern** | Place top-1 result first, top-2 result last, rest in between | High — exploits both high-attention zones | Low |
| **Compressed context** | Summarize or truncate lower-ranked chunks; keep top chunks verbatim | Very high — reduces noise, preserves signal | Medium |
| **Explicit citation prompting** | Instruct the model to "quote the source text before answering" | Medium — forces the model to attend to the context | Low |
| **Smaller context windows** | Use fewer, better chunks instead of stuffing the window | High — often the simplest and best approach | Low |

### Context assembly pipeline

```
Retrieved chunks (top-k after reranking)
    │
    ├── 1. Deduplicate
    │   └── Remove chunks with >80% token overlap (jaccard or minhash)
    │       Remove chunks from the same source if one is a superset of another
    │
    ├── 2. Trim
    │   └── If total tokens > budget, remove lowest-ranked chunks first
    │       Consider truncating the last chunk to fit exactly
    │
    ├── 3. Order
    │   └── Relevance-descending (best first) OR bookend pattern
    │       Group chunks from same document together (maintains coherence)
    │
    ├── 4. Format
    │   └── Add source metadata as lightweight headers:
    │       "[Source: filename.py, lines 45-72]"
    │       Separate chunks with "---" or XML tags
    │       DO NOT repeat the query in each chunk's context
    │
    └── 5. Budget check
        └── Total = system prompt + assembled context + conversation history + response buffer
            If over limit, reduce k or summarize older conversation turns
```

### Token budget allocation (for ~128K window, practical limit ~60K usable)

| Component | Token budget | Notes |
|---|---|---|
| System prompt | 500-2,000 | Stable; includes task instructions |
| Retrieved context | 2,000-8,000 | 3-5 chunks × 400-600 tokens each |
| Conversation history | 2,000-10,000 | Recent turns; summarize older ones |
| Response buffer | 2,000-4,000 | Space for the model's answer |
| **Total target** | **~8,000-20,000** | Stay well under window limit |

**Critical insight:** Don't fill the context window. **3-5 high-quality chunks outperform 15-20 mediocre ones** — lower-ranked chunks introduce noise that degrades answers.

### Recommendation

Retrieve top-20, rerank to top-5, inject top-3. Relevance-descending order. Deduplicate via substring containment (if A ⊂ B, drop A). Format with XML tags: `<context source="file:lines">...</context>`. Hard budget: 4,000-6,000 tokens for retrieval context.

---

## 4. Index Architecture

**Summary:** Index choice is less critical than chunking and retrieval strategy — most vector stores perform identically under 1M vectors. Choose based on operational simplicity, not theoretical throughput.

### Index algorithm comparison

| Algorithm | How it works | Query speed (1M vectors) | Index build time | Memory | Recall@10 | When to use |
|---|---|---|---|---|---|---|
| **Flat (brute force)** | Exact cosine/dot product on all vectors | ~50ms (CPU) | Instant | O(n×d) | 100% | Under 50K vectors; prototyping |
| **HNSW** | Approximate nearest neighbors via navigable small-world graph | ~1-5ms | Minutes | 1.5-2× vectors | 95-99% | 50K-10M vectors; best general-purpose |
| **IVF** (Inverted File) | Cluster vectors, search only nearest clusters | ~5-20ms | Minutes | O(n×d) + centroids | 90-98% | 1M+ vectors; memory-constrained |
| **IVF-PQ** | IVF + Product Quantization (compressed vectors) | ~2-10ms | Minutes-hours | 0.1-0.25× vectors | 85-95% | 10M+ vectors; very memory-constrained |

### Vector store comparison

| Store | Deployment | Persistence | Filtering | Hybrid search | Scaling ceiling | Best for |
|---|---|---|---|---|---|---|
| **ChromaDB** | Embedded (pip install) | SQLite + Parquet | Basic metadata | No native BM25 | ~500K vectors | Prototyping; single-developer projects |
| **Qdrant** | Embedded or server | Disk-based, memory-mapped | Rich filtering, payload indexes | Sparse vectors + dense (v1.7+) | Multi-million | Best balance of features and simplicity |
| **Weaviate** | Docker/Cloud | Persistent | GraphQL-based | BM25 + vector (native) | Multi-million | When you want hybrid search built-in |
| **FAISS** | Library (no server) | Manual save/load | None built-in | None built-in | Billions (with GPU) | Maximum throughput; batch operations; research |
| **pgvector** | PostgreSQL extension | PostgreSQL | Full SQL | Pair with `pg_trgm` or external | ~1M (CPU); improving | When you already run PostgreSQL; want single data store |
| **LanceDB** | Embedded | Lance columnar format | Rich filtering | No native | Multi-million | Embedded use with good filtering; serverless |

### Decision tree

```
How many vectors will you have?
├── Under 50K → ChromaDB (simplest) or pgvector (if you already use Postgres)
├── 50K - 1M
│   ├── Need hybrid search? → Qdrant (sparse+dense) or Weaviate (native BM25)
│   └── Don't need hybrid? → Any of the above
└── Over 1M
    ├── Have GPU? → FAISS with IVF-PQ
    └── CPU only? → Qdrant (memory-mapped) or Weaviate
```

### Operational notes

- **ChromaDB:** Fastest to prototype (no Docker/server). Limited filtering, no native hybrid search.
- **Qdrant:** Best DX for production. Embedded mode for dev, server for prod. Native sparse vectors (v1.7+) enable hybrid retrieval without a separate BM25 index. Rust-based, fast on CPU.
- **pgvector:** Compelling if you already run PostgreSQL. `halfvec` (v0.7+) halves memory; HNSW (v0.5+) is competitive at sub-1M scale. No native hybrid search.
- **FAISS:** A library, not a database — no persistence, filtering, or server. Use for batch processing or research only.

### Recommendation

**Qdrant embedded mode** (`pip install qdrant-client`). Native dense + sparse vectors for hybrid retrieval in one store, rich metadata filtering, scales to millions via memory-mapping. Same API for embedded and server modes. Alternative: **pgvector** with HNSW if you already run PostgreSQL (up to ~500K vectors).

---

## 5. Multi-Session State

**Summary:** Different state types have different lifetimes, access patterns, and staleness tolerances. Mixing them into a single store creates maintenance nightmares. Separate by tier.

### State taxonomy

| State type | Lifetime | Storage | Access pattern | Example |
|---|---|---|---|---|
| **Corpus knowledge** | Persistent | Vector index | Semantic similarity search | Documentation, codebase, reference materials |
| **Session memory** | Multi-session, decaying | Structured file (JSON/SQLite) | Key-value or chronological | "User prefers TypeScript", "Last session debugged auth module" |
| **Conversation history** | Single session (with summarization) | In-memory → vector index for long-term | Recent: full text; old: summarized | Current chat turns |
| **Task state** | Single session | Structured file | Direct lookup | Current plan, todo list, progress |
| **Derived facts** | Persistent, versioned | Structured store + vector index | Hybrid (exact + semantic) | "The auth module uses JWT with RS256", "API rate limit is 100/min" |

### Architecture pattern: three-tier memory

```
┌─────────────────────────────────────────────┐
│  Tier 1: Working Memory (in-context)         │
│  - Current conversation turns                │
│  - Active task state                         │
│  - Recently retrieved chunks                 │
│  Token budget: 10-20K tokens                 │
├─────────────────────────────────────────────┤
│  Tier 2: Session Store (structured file)     │
│  - User preferences and corrections          │
│  - Session summaries                         │
│  - Derived facts with timestamps             │
│  Format: JSON/SQLite with TTL               │
├─────────────────────────────────────────────┤
│  Tier 3: Corpus Index (vector store)         │
│  - Documentation, code, reference materials  │
│  - Conversation summaries (long-term)        │
│  - Indexed with source + timestamp metadata  │
│  Store: Qdrant / pgvector / Chroma          │
└─────────────────────────────────────────────┘
```

### Handling index staleness

| Strategy | Implementation | Latency | Complexity | When to use |
|---|---|---|---|---|
| **Full reindex** | Rebuild entire index on change | High (minutes) | Low | Corpus under 10K chunks; infrequent changes |
| **Incremental update** | Track file modification times; re-embed changed files | Low | Medium | Default for most projects |
| **Content-hash dedup** | Hash chunk content; skip unchanged chunks during reindex | Low | Medium | Reduce redundant embedding API calls |
| **Versioned chunks** | Store chunk version/timestamp; filter or boost by recency at query time | Low | Medium-High | When old and new versions coexist (e.g., API docs) |
| **Watch + queue** | File watcher triggers background re-embedding | Very low | High | Real-time freshness requirements |

**Practical pattern for codebase indexing:**
1. Store `(file_path, content_hash, last_modified)` in a metadata table
2. On reindex: scan files, compare hashes, re-embed only changed files
3. Delete vectors for removed files
4. Run reindex as a pre-session hook or on-demand command

### What goes where

| Information | Store in... | Why |
|---|---|---|
| "User corrected my approach to error handling" | Session store (structured) | Needs exact recall, not fuzzy search |
| "The codebase uses FastAPI with SQLAlchemy" | Derived facts (structured + indexed) | Exact fact, but should be discoverable via semantic search |
| "Contents of `src/auth/middleware.py`" | Corpus index (vector) | Source document; retrieved by similarity |
| "Last session we were debugging the auth flow" | Session store (structured) | Session continuity; chronological access |
| "The 3 most relevant code chunks for the current query" | Working memory (in-context) | Active use; ephemeral |
| Full conversation transcript from 2 weeks ago | Archived (not indexed unless summarized) | Too large for vector index; summarize key decisions |

### Recommendation

Use the three-tier pattern. **Tier 1:** Sliding window — last 3-5 turns verbatim, older turns summarized to 1-2 sentences. **Tier 2:** JSON file per project (preferences, derived facts, session summaries). **Tier 3:** Qdrant corpus index with content-hash incremental updates, triggered manually or pre-session. A well-structured JSON file outperforms a database for single-developer workflows.

---

## 6. Failure Modes

**Summary:** RAG systems fail silently — irrelevant retrieval still produces fluent, confident answers. Detecting degradation requires explicit monitoring across three categories: retrieval failures (wrong chunks), representation failures (bad embeddings, stale indexes), and assembly failures (poor context construction).

### Failure mode catalog

| Failure mode | What happens | Detection | Mitigation |
|---|---|---|---|
| **Hallucinated retrieval** | Model ignores context, generates from parametric knowledge | Answer contains info absent from retrieved chunks | Constrain with "answer ONLY from provided context"; require citations |
| **Embedding drift** | Old vectors incompatible after model change | Similarity distributions shift; quality drops suddenly | Re-embed entire corpus on model change. Version model ID in metadata. Never mix models. |
| **Stale chunks** | Index contradicts current document state | Track file mod times; monitor staleness ratio | Incremental reindexing (Section 5). Alert when staleness > 20%. |
| **Chunk boundary artifacts** | Key info split across chunks, neither sufficient | Manual review of retrieval failures | Semantic or hierarchical chunking; overlap as fallback |
| **Query-document mismatch** | Query phrasing differs from document phrasing | BM25 finds it but dense doesn't (or vice versa) | Hybrid retrieval. Query expansion. HyDE for hard cases. |
| **Over-retrieval** | Low-relevance chunks dilute signal | Quality degrades as k increases; model hedges | Reduce k. Minimum similarity threshold. Rerank with hard cutoff. |
| **Degenerate queries** | Short/ambiguous queries yield poor embeddings | Low max similarity across all results | Detect low-confidence retrieval; fall back to clarification or BM25 |
| **Index corruption** | Partial writes or concurrent access | Vector count mismatch; empty/nonsensical results | Use stores with WAL (Qdrant, pgvector). Regular backup. Startup integrity checks. |
| **Token budget overflow** | Context + history exceeds window | API errors or truncation | Hard budget enforcement. Aggressive summarization near limits. |

### Monitoring checklist

Lightweight checks that catch most problems without a full observability platform:

1. **Retrieval relevance:** Log top-3 chunks + similarity scores per query. Review weekly; investigate if mean similarity drops.
2. **Staleness ratio:** `(modified files since last index) / (total indexed files)`. Alert if > 20%.
3. **Answer grounding:** Spot-check whether answers are derivable from retrieved context (automatable with LLM-as-judge).
4. **Model version:** Store embedding model ID with every vector. Block queries on model mismatch.
5. **Chunk count sanity:** Compare post-reindex count to expected. Large deviations signal parsing bugs.

### The most dangerous failure

**Silent retrieval degradation** — plausible-looking but wrong chunks — is hardest to detect because the LLM confidently synthesizes answers from bad context. Most common when the corpus has similar documents (e.g., multiple API versions), chunks are too small to disambiguate, or the embedding model conflates related-but-distinct concepts.

**Mitigation:** Include source metadata in prompts. Require citations. Set a minimum similarity threshold (e.g., 0.7 for `text-embedding-3-small`) — below it, return "no relevant context found" rather than bad context.

### Recommendation

Day-one essentials: (1) Log retrieval results with similarity scores. (2) Staleness check on startup. (3) Minimum similarity threshold with explicit "no context found" handling. These three catch 80% of real-world RAG failures.

---

## 7. Recommended Baseline Stack

A specific, opinionated set of choices a developer can implement in a weekend.

### The stack

| Component | Choice | Justification |
|---|---|---|
| **Embedding model** | `text-embedding-3-small` at 512 dims | Best cost/quality ratio. Matryoshka truncation saves 67% storage, ~2% quality loss. Local alt: `bge-small-en-v1.5` (384d, CPU ~15ms). |
| **Chunking** | Semantic, 256-512 tokens | `tree-sitter` for code, header-split for markdown. No overlap needed. |
| **Vector store** | Qdrant embedded (`qdrant-client`) | Zero infra. Native sparse + dense. Scales via memory-mapping. Same API embedded → server. |
| **Sparse retrieval** | BM25 via Qdrant sparse vectors | Catches keyword matches dense misses. Essential for code queries. |
| **Score fusion** | Reciprocal Rank Fusion (RRF) | Parameter-free, well-studied. `score = Σ 1/(k + rank_i)`, k=60. |
| **Reranker** | `ms-marco-MiniLM-L-6-v2` | Free, CPU ~50ms/20 candidates, 10-15% precision gain. |
| **Retrieval pipeline** | Top-20 hybrid → rerank → top-3 | 3 chunks × 400 tokens = ~1,200 tokens of context. |
| **Context assembly** | Relevance-descending, XML tags, 6K budget | Mitigates lost-in-the-middle. Budget prevents noise injection. |
| **Session memory** | JSON file per project | Simple, readable, versionable. No DB needed for single-dev. |
| **Staleness** | Content-hash incremental reindex | `sha256` on content before embedding. Manual or pre-session trigger. |
| **LLM** | Sonnet/GPT-4o (cost) or Opus/GPT-4 (quality) | Sonnet/4o handle RAG well. Upgrade for complex multi-hop reasoning. |

### Implementation sketch

```
project/
├── index.py          # Chunking + embedding + Qdrant ingestion
├── retrieve.py       # Hybrid retrieval + reranking + context assembly
├── chat.py           # LLM integration with assembled context
├── reindex.py        # Incremental reindex with content hashing
├── session_store.json # Per-project session memory
└── config.py         # Model names, thresholds, token budgets
```

### Weekend implementation plan

**Saturday morning — Indexing pipeline (3-4 hours):**
1. Write a chunker that splits markdown by headers and code by functions (tree-sitter)
2. Embed chunks with `text-embedding-3-small` (or `bge-small-en-v1.5` locally)
3. Store in Qdrant embedded mode with metadata (file_path, content_hash, timestamp)
4. Add BM25 sparse vectors alongside dense vectors

**Saturday afternoon — Retrieval pipeline (2-3 hours):**
1. Implement hybrid query (dense + sparse with RRF)
2. Add cross-encoder reranking on top-20 results
3. Build context assembly: deduplicate, order, format, budget-check
4. Test with sample queries against your corpus

**Sunday morning — LLM integration (2-3 hours):**
1. Wire up the retrieval pipeline to Claude/GPT-4 API calls
2. Design the system prompt with context injection template
3. Add conversation history management (sliding window + summarization)
4. Implement the "no relevant context found" fallback

**Sunday afternoon — Robustness (2-3 hours):**
1. Add content-hash incremental reindexing
2. Add retrieval logging (query, chunks, scores)
3. Set minimum similarity threshold (start at 0.65, tune empirically)
4. Test edge cases: empty queries, no-match queries, very long documents

---

## Appendix: Areas of Active Debate

1. **Optimal chunk size.** 256-512 tokens is common advice, but some practitioners see better results at 1024+ for complex reasoning. The interaction between chunk size and context window utilization is under-studied.

2. **Late chunking.** Jina AI's approach (embed full document with a long-context model, then split contextualized embeddings) is promising but new (2024) and not widely validated.

3. **Query transformation.** HyDE, step-back prompting, and multi-query approaches show improvements in papers but add latency. Practical ROI vs. hybrid + reranking is unclear.

4. **RAG vs. long context.** As windows grow (Claude 200K, Gemini 1M+), stuffing the full corpus into context becomes viable for small corpora (<50K tokens). RAG still wins on precision for most tasks, but the crossover point is moving.

5. **Embedding fine-tuning.** Domain-specific fine-tuning improves retrieval quality but requires training data and GPU compute. Off-the-shelf models are sufficient for most small teams.

6. **GraphRAG.** Microsoft's knowledge-graph-based retrieval shows strong multi-hop reasoning results, but index-time cost is high (many LLM calls) and the approach is not yet mature for general use.

---

## References

- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023) — https://arxiv.org/abs/2307.03172
- MTEB Benchmark Leaderboard — https://huggingface.co/spaces/mteb/leaderboard
- BEIR Benchmark — https://github.com/beir-cellar/beir
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE, 2022) — https://arxiv.org/abs/2212.10496
- Microsoft GraphRAG (2024) — https://github.com/microsoft/graphrag
- Jina AI Late Chunking (2024) — https://jina.ai/news/late-chunking-in-long-context-embedding-models/
- Reciprocal Rank Fusion — Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (2009)
