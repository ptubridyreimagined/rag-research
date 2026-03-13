# rag-managed: true
"""
Core RAG engine — indexing, hybrid retrieval, reranking, context assembly.

Usage:
    python rag/engine.py --query "how is auth handled"
    python rag/engine.py --index
    python rag/engine.py --staleness
    python rag/engine.py --status
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for direct invocation (python rag/engine.py)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from rag.chunker import Chunk, chunk_file, count_tokens
from rag.config import RAGConfig, load_config

logger = logging.getLogger("rag.engine")

# ---------------------------------------------------------------------------
# Embedding wrapper — lazy-loaded to keep import fast
# ---------------------------------------------------------------------------
_embedder = None
_reranker = None


def _get_embedder(model_name: str):
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(model_name)
        _embedder.model_name = model_name
    return _embedder


def _get_reranker(model_name: str):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(model_name)
    return _reranker


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed a batch of texts. Returns list of vectors."""
    if not texts:
        return []
    model = _get_embedder(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


# ---------------------------------------------------------------------------
# BM25 sparse retrieval
# ---------------------------------------------------------------------------

def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Lightweight BM25 index over chunk documents."""

    def __init__(self):
        self._corpus: list[list[str]] = []
        self._ids: list[str] = []
        self._bm25 = None

    def build(self, ids: list[str], documents: list[str]):
        from rank_bm25 import BM25Okapi
        self._ids = ids
        self._corpus = [_tokenize_for_bm25(doc) for doc in documents]
        self._bm25 = BM25Okapi(self._corpus)

    def query(self, text: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Return (id, score) pairs sorted by BM25 score."""
        if self._bm25 is None or not self._ids:
            return []
        tokens = _tokenize_for_bm25(text)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# ChromaDB wrapper
# ---------------------------------------------------------------------------

def _get_collection(cfg: RAGConfig):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(cfg.index_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name="rag_index",
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

@dataclass
class StaleFile:
    path: str
    index_mtime: float
    file_mtime: float

    @property
    def stale_seconds(self) -> float:
        return self.file_mtime - self.index_mtime


def detect_staleness(cfg: RAGConfig) -> list[StaleFile]:
    """Compare index mtime metadata against actual file mtimes."""
    collection = _get_collection(cfg)
    stale: list[StaleFile] = []

    # Get all indexed file paths and their mtimes from metadata
    try:
        result = collection.get(include=["metadatas"])
    except Exception:
        return []

    indexed: dict[str, float] = {}
    for meta in (result.get("metadatas") or []):
        if meta and "source_file" in meta and "indexed_at" in meta:
            path = meta["source_file"]
            if path not in indexed or meta["indexed_at"] > indexed[path]:
                indexed[path] = meta["indexed_at"]

    for path, idx_time in indexed.items():
        full = cfg.project_root / path
        if full.exists():
            file_mtime = full.stat().st_mtime
            if file_mtime > idx_time:
                stale.append(StaleFile(path, idx_time, file_mtime))
        else:
            # File deleted — also stale (needs removal)
            stale.append(StaleFile(path, idx_time, 0))

    return sorted(stale, key=lambda s: s.path)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _should_exclude(path: Path, excludes: list[str], root: Path) -> bool:
    """Check if path matches any exclude pattern."""
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        # path not under root — use name-only matching
        rel = path.name
    name = path.name
    # Normalize separators for cross-platform matching
    rel = rel.replace("\\", "/")
    for pat in excludes:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat):
            return True
        if fnmatch.fnmatch(rel, f"*/{pat}/*") or fnmatch.fnmatch(rel, f"{pat}/*"):
            return True
        # Directory match
        if path.is_dir() and fnmatch.fnmatch(name, pat):
            return True
    return False


def discover_files(cfg: RAGConfig) -> list[Path]:
    """Walk index_paths respecting exclude_patterns."""
    files: list[Path] = []
    seen: set[str] = set()
    root = cfg.project_root.resolve()

    for idx_path_str in cfg.index_paths:
        idx_path = (root / idx_path_str).resolve()
        if not idx_path.exists():
            logger.warning("Index path does not exist: %s", idx_path)
            continue

        if idx_path.is_file():
            files.append(idx_path)
            continue

        for dirpath, dirnames, filenames in os.walk(idx_path):
            dp = Path(dirpath)

            # Prune excluded directories in-place
            dirnames[:] = [
                d for d in dirnames
                if not _should_exclude(dp / d, cfg.exclude_patterns, root)
            ]

            for fname in filenames:
                fp = dp / fname
                if _should_exclude(fp, cfg.exclude_patterns, root):
                    continue
                real = str(fp.resolve())
                if real not in seen:
                    seen.add(real)
                    files.append(fp)

    return sorted(files)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_repo(cfg: RAGConfig, paths: list[Path] | None = None) -> dict[str, Any]:
    """Chunk, embed, and upsert files into ChromaDB. Returns stats."""
    if paths is None:
        paths = discover_files(cfg)

    collection = _get_collection(cfg)
    now = time.time()

    all_chunks: list[Chunk] = []
    for fp in paths:
        try:
            rel = str(fp.relative_to(cfg.project_root))
        except ValueError:
            rel = str(fp)
        chunks = chunk_file(
            fp,
            strategy=cfg.chunk_strategy,
            max_tokens=cfg.chunk_size_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
        )
        # Rewrite source_file to relative path
        for c in chunks:
            c.source_file = rel
            c.id = f"{rel}:{c.start_line}-{c.end_line}:{c.content_hash}"
        all_chunks.extend(chunks)

    if not all_chunks:
        return {"files": 0, "chunks": 0, "skipped": 0}

    # Batch embed
    batch_size = 256
    total_embedded = 0

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [c.content for c in batch]
        ids = [c.id for c in batch]
        metadatas = [{
            "source_file": c.source_file,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "token_count": c.token_count,
            "content_hash": c.content_hash,
            "chunk_type": c.chunk_type,
            "indexed_at": now,
        } for c in batch]

        embeddings = embed_texts(texts, cfg.embedding_model)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_embedded += len(batch)

    # Save chunk metadata cache
    cache_path = cfg.chunks_dir / "chunks.json"
    cache_data = [{"id": c.id, **c.to_dict()} for c in all_chunks]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    return {
        "files": len(paths),
        "chunks": total_embedded,
        "index_dir": str(cfg.index_dir),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    id: str
    content: str
    source_file: str
    start_line: int
    end_line: int
    token_count: int
    score: float
    retrieval_method: str


def retrieve(query: str, cfg: RAGConfig) -> list[RetrievedChunk]:
    """Hybrid BM25 + dense retrieval with optional cross-encoder reranking."""
    collection = _get_collection(cfg)

    total_docs = collection.count()
    if total_docs == 0:
        logger.warning("Index is empty. Run indexing first.")
        return []

    # --- Dense retrieval ---
    query_embedding = embed_texts([query], cfg.embedding_model)[0]
    dense_k = min(cfg.retrieval_top_k, total_docs)
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=dense_k,
        include=["documents", "metadatas", "distances"],
    )

    dense_chunks: dict[str, RetrievedChunk] = {}
    for i, cid in enumerate(dense_results["ids"][0]):
        # ChromaDB returns distances for cosine; similarity = 1 - distance
        distance = dense_results["distances"][0][i]
        similarity = 1.0 - distance
        meta = dense_results["metadatas"][0][i]
        doc = dense_results["documents"][0][i]

        dense_chunks[cid] = RetrievedChunk(
            id=cid,
            content=doc,
            source_file=meta.get("source_file", ""),
            start_line=meta.get("start_line", 0),
            end_line=meta.get("end_line", 0),
            token_count=meta.get("token_count", count_tokens(doc)),
            score=similarity,
            retrieval_method="dense",
        )

    # --- BM25 sparse retrieval ---
    # Fetch all documents for BM25 (practical for <100K chunks)
    bm25 = BM25Index()
    all_docs = collection.get(include=["documents"])
    if all_docs["ids"] and all_docs["documents"]:
        bm25.build(all_docs["ids"], all_docs["documents"])
        bm25_results = bm25.query(query, top_k=cfg.retrieval_top_k)
    else:
        bm25_results = []

    # --- Reciprocal Rank Fusion ---
    rrf_k = 60
    rrf_scores: dict[str, float] = defaultdict(float)

    # Dense ranks
    dense_ranked = sorted(dense_chunks.values(), key=lambda x: x.score, reverse=True)
    for rank, chunk in enumerate(dense_ranked):
        rrf_scores[chunk.id] += 1.0 / (rrf_k + rank + 1)

    # BM25 ranks
    for rank, (cid, _score) in enumerate(bm25_results):
        rrf_scores[cid] += 1.0 / (rrf_k + rank + 1)

    # Merge — ensure we have content for all candidates
    bm25_ids_needed = [cid for cid, _ in bm25_results if cid not in dense_chunks]
    if bm25_ids_needed:
        extra = collection.get(ids=bm25_ids_needed, include=["documents", "metadatas"])
        for i, cid in enumerate(extra["ids"]):
            meta = extra["metadatas"][i]
            doc = extra["documents"][i]
            dense_chunks[cid] = RetrievedChunk(
                id=cid,
                content=doc,
                source_file=meta.get("source_file", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                token_count=meta.get("token_count", count_tokens(doc)),
                score=0.0,
                retrieval_method="bm25",
            )

    # Apply RRF scores
    for cid, score in rrf_scores.items():
        if cid in dense_chunks:
            dense_chunks[cid].score = score
            dense_chunks[cid].retrieval_method = "hybrid"

    # Sort by RRF score
    candidates = sorted(
        [c for c in dense_chunks.values() if c.id in rrf_scores],
        key=lambda x: x.score,
        reverse=True,
    )[:cfg.retrieval_top_k]

    # --- Cross-encoder reranking ---
    if cfg.reranking_enabled and len(candidates) > cfg.inject_top_k:
        try:
            reranker = _get_reranker(cfg.reranker_model)
            pairs = [(query, c.content) for c in candidates]
            rerank_scores = reranker.predict(pairs)
            for c, rs in zip(candidates, rerank_scores):
                c.score = float(rs)
                c.retrieval_method = "reranked"
            candidates.sort(key=lambda x: x.score, reverse=True)
        except Exception as e:
            logger.warning("Reranking failed, using RRF scores: %s", e)

    # Apply minimum similarity filter
    candidates = [c for c in candidates if c.score >= cfg.min_similarity]

    # Trim to inject_top_k
    result = candidates[:cfg.inject_top_k]

    # Log retrieval
    _log_retrieval(query, result, cfg)

    return result


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def assemble_context(
    chunks: list[RetrievedChunk],
    budget_tokens: int,
    bookend: bool = True,
) -> str:
    """
    Deduplicate, order, and trim chunks to fit the token budget.

    Ordering: if bookend=True, places most relevant first AND last
    (lost-in-the-middle mitigation). Otherwise, strict relevance-descending.
    """
    if not chunks:
        return ""

    # --- Deduplicate: drop chunks with >80% content overlap ---
    deduped: list[RetrievedChunk] = []
    seen_hashes: set[str] = set()
    for c in chunks:
        h = c.id.split(":")[-1] if ":" in c.id else c.id
        if h in seen_hashes:
            continue
        # Check substring containment
        is_subset = False
        for existing in deduped:
            if c.content in existing.content:
                is_subset = True
                break
            if existing.content in c.content:
                # Replace existing with the larger chunk
                deduped.remove(existing)
                break
        if not is_subset:
            deduped.append(c)
            seen_hashes.add(h)

    # --- Token trimming: fit within budget ---
    fitted: list[RetrievedChunk] = []
    used_tokens = 0
    overhead_per_chunk = 40  # XML tags + source metadata

    for c in deduped:
        needed = c.token_count + overhead_per_chunk
        if used_tokens + needed > budget_tokens:
            # Try to fit a truncated version
            remaining = budget_tokens - used_tokens - overhead_per_chunk
            if remaining > 50:
                # Truncate content to fit
                truncated = c.content[:remaining * 4]  # ~4 chars/token
                c.content = truncated
                c.token_count = count_tokens(truncated)
                fitted.append(c)
            break
        fitted.append(c)
        used_tokens += needed

    if not fitted:
        return ""

    # --- Ordering: bookend pattern ---
    if bookend and len(fitted) > 2:
        # Top-1 first, top-2 last, rest in middle
        ordered = [fitted[0]] + fitted[2:] + [fitted[1]]
    else:
        ordered = fitted

    # --- Format with XML tags ---
    parts: list[str] = []
    for c in ordered:
        source_label = c.source_file
        if c.start_line and c.end_line:
            source_label += f":{c.start_line}-{c.end_line}"
        parts.append(
            f'<context source="{source_label}" score="{c.score:.3f}">\n'
            f"{c.content}\n"
            f"</context>"
        )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Retrieval logging
# ---------------------------------------------------------------------------

def _log_retrieval(
    query: str,
    chunks: list[RetrievedChunk],
    cfg: RAGConfig,
):
    """Append to retrieval.log, keeping last 50 entries."""
    log_path = cfg.logs_dir / "retrieval.log"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "results": [
            {
                "source": c.source_file,
                "lines": f"{c.start_line}-{c.end_line}",
                "score": round(c.score, 4),
                "method": c.retrieval_method,
                "tokens": c.token_count,
            }
            for c in chunks
        ],
    }

    # Read existing entries
    entries: list[dict] = []
    if log_path.exists():
        try:
            for line in log_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    entries.append(json.loads(line))
        except Exception:
            entries = []

    entries.append(entry)
    entries = entries[-50:]  # Keep last 50

    with open(log_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# Project summary generator
# ---------------------------------------------------------------------------

def generate_project_summary(cfg: RAGConfig) -> str:
    """Auto-generate PROJECT.md from repo structure and key files."""
    lines = [
        "<!-- rag-managed: true -->",
        "# Project Summary",
        "",
        f"*Auto-generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "*Edit freely — this file is human-editable and will not be overwritten unless you re-run project detection.*",
        "",
    ]

    # Detect project type
    root = cfg.project_root
    project_type = "generic"
    if (root / "package.json").exists():
        project_type = "node"
    elif (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        project_type = "python"
    elif (root / "Cargo.toml").exists():
        project_type = "rust"
    elif (root / "go.mod").exists():
        project_type = "go"

    lines.append(f"## Project Type: {project_type}")
    lines.append("")

    # Key files
    key_files = ["README.md", "CLAUDE.md", "package.json", "pyproject.toml",
                 "Cargo.toml", "go.mod", "Makefile", "Dockerfile"]
    found = [f for f in key_files if (root / f).exists()]
    if found:
        lines.append("## Key Files")
        for f in found:
            lines.append(f"- `{f}`")
        lines.append("")

    # Top-level directory structure
    lines.append("## Structure")
    lines.append("```")
    try:
        entries = sorted(root.iterdir())
        for entry in entries[:30]:
            if entry.name.startswith(".") and entry.name not in (".github", ".claude"):
                continue
            indicator = "/" if entry.is_dir() else ""
            lines.append(f"  {entry.name}{indicator}")
        if len(entries) > 30:
            lines.append(f"  ... ({len(entries) - 30} more)")
    except Exception:
        lines.append("  (unable to read directory)")
    lines.append("```")
    lines.append("")

    # Read README excerpt if available
    readme = root / "README.md"
    if readme.exists():
        try:
            content = readme.read_text(encoding="utf-8", errors="replace")
            # First 500 chars
            excerpt = content[:500].strip()
            if len(content) > 500:
                excerpt += "..."
            lines.append("## README Excerpt")
            lines.append(excerpt)
            lines.append("")
        except Exception:
            pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Index stats
# ---------------------------------------------------------------------------

def get_index_stats(cfg: RAGConfig) -> dict[str, Any]:
    """Return index statistics."""
    collection = _get_collection(cfg)
    total_chunks = collection.count()

    # Get unique files
    files: set[str] = set()
    if total_chunks > 0:
        result = collection.get(include=["metadatas"])
        for meta in (result.get("metadatas") or []):
            if meta and "source_file" in meta:
                files.add(meta["source_file"])

    stale = detect_staleness(cfg)
    staleness_pct = (len(stale) / max(len(files), 1)) * 100

    return {
        "total_chunks": total_chunks,
        "total_files": len(files),
        "stale_files": len(stale),
        "staleness_pct": round(staleness_pct, 1),
        "index_dir": str(cfg.index_dir),
        "embedding_model": cfg.embedding_model,
        "chunk_strategy": cfg.chunk_strategy,
    }


# ---------------------------------------------------------------------------
# Incremental reindex
# ---------------------------------------------------------------------------

def reindex_stale(cfg: RAGConfig) -> dict[str, Any]:
    """Reindex only stale files. Returns stats."""
    stale = detect_staleness(cfg)
    if not stale:
        return {"reindexed": 0, "removed": 0}

    collection = _get_collection(cfg)
    to_reindex: list[Path] = []
    removed = 0

    for sf in stale:
        full_path = cfg.project_root / sf.path
        # Remove old chunks for this file
        try:
            existing = collection.get(
                where={"source_file": sf.path},
                include=[],
            )
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        if full_path.exists() and sf.file_mtime > 0:
            to_reindex.append(full_path)
        else:
            removed += 1

    stats = {"reindexed": 0, "removed": removed}
    if to_reindex:
        result = index_repo(cfg, paths=to_reindex)
        stats["reindexed"] = result.get("files", 0)

    return stats


# ---------------------------------------------------------------------------
# Remove from index
# ---------------------------------------------------------------------------

def forget_path(path: str, cfg: RAGConfig) -> int:
    """Remove all chunks for a given path (or prefix) from the index."""
    collection = _get_collection(cfg)
    try:
        result = collection.get(include=["metadatas"])
    except Exception:
        return 0

    to_delete: list[str] = []
    for i, meta in enumerate(result.get("metadatas") or []):
        if meta and "source_file" in meta:
            sf = meta["source_file"]
            if sf == path or sf.startswith(path + "/") or sf.startswith(path + "\\"):
                to_delete.append(result["ids"][i])

    if to_delete:
        collection.delete(ids=to_delete)
    return len(to_delete)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG Engine CLI")
    parser.add_argument("--query", "-q", type=str, help="Run retrieval query")
    parser.add_argument("--index", action="store_true", help="Run full index")
    parser.add_argument("--reindex", action="store_true", help="Reindex stale files")
    parser.add_argument("--staleness", action="store_true", help="Show stale files")
    parser.add_argument("--status", action="store_true", help="Show index stats")
    parser.add_argument("--forget", type=str, help="Remove path from index")
    parser.add_argument("--context-budget", type=int, help="Override context budget")
    parser.add_argument("--root", type=str, help="Project root directory")

    args = parser.parse_args()

    root = Path(args.root) if args.root else None
    cfg = load_config(root)

    if args.context_budget:
        cfg.context_budget_tokens = args.context_budget

    if args.index:
        print("Indexing repository...")
        stats = index_repo(cfg)
        print(f"Indexed {stats['files']} files -> {stats['chunks']} chunks")
        print(f"Index stored at: {stats['index_dir']}")

    elif args.reindex:
        print("Checking for stale files...")
        stats = reindex_stale(cfg)
        print(f"Reindexed: {stats['reindexed']} files, Removed: {stats['removed']} files")

    elif args.staleness:
        stale = detect_staleness(cfg)
        if not stale:
            print("Index is up to date.")
        else:
            print(f"Found {len(stale)} stale file(s):")
            for sf in stale:
                if sf.file_mtime == 0:
                    print(f"  DELETED  {sf.path}")
                else:
                    age = int(sf.stale_seconds)
                    print(f"  STALE    {sf.path}  ({age}s behind)")

    elif args.status:
        stats = get_index_stats(cfg)
        print(f"Files indexed:    {stats['total_files']}")
        print(f"Total chunks:     {stats['total_chunks']}")
        print(f"Stale files:      {stats['stale_files']} ({stats['staleness_pct']}%)")
        print(f"Embedding model:  {stats['embedding_model']}")
        print(f"Chunk strategy:   {stats['chunk_strategy']}")
        print(f"Index location:   {stats['index_dir']}")

    elif args.forget:
        n = forget_path(args.forget, cfg)
        print(f"Removed {n} chunk(s) matching '{args.forget}'")

    elif args.query:
        chunks = retrieve(args.query, cfg)
        if not chunks:
            print("No relevant results found.")
            sys.exit(0)

        print(f"Top {len(chunks)} results for: {args.query}\n")
        for i, c in enumerate(chunks, 1):
            source = c.source_file
            if c.start_line and c.end_line:
                source += f":{c.start_line}-{c.end_line}"
            print(f"--- [{i}] {source}  (score: {c.score:.4f}, method: {c.retrieval_method}) ---")
            # Show first 300 chars, handling encoding issues on Windows
            preview = c.content[:300]
            if len(c.content) > 300:
                preview += "..."
            try:
                print(preview)
            except UnicodeEncodeError:
                print(preview.encode("ascii", errors="replace").decode("ascii"))
            print()

        # Also print assembled context
        assembled = assemble_context(chunks, cfg.context_budget_tokens)
        token_count = count_tokens(assembled)
        print(f"=== Assembled context: {token_count} tokens ===")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
