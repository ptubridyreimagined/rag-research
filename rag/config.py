# rag-managed: true
"""Load and validate .rag/config.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Walk up from cwd to find .rag/ directory."""
    cur = Path.cwd()
    for parent in [cur, *cur.parents]:
        if (parent / ".rag" / "config.yaml").exists():
            return parent
    return cur


@dataclass
class RAGConfig:
    project_root: Path = field(default_factory=_project_root)
    index_paths: list[str] = field(default_factory=lambda: ["."])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "node_modules", "__pycache__", ".git", ".rag/index", ".rag/chunks",
        "dist", "build", ".venv", "venv", "*.pyc", "*.min.js", "*.map",
        "package-lock.json", "yarn.lock", "poetry.lock",
    ])
    chunk_strategy: str = "semantic"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 0
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    context_budget_tokens: int = 40000
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    inject_top_k: int = 3
    min_similarity: float = 0.25
    session_type: str = "short"
    watch_mode: bool = False
    reranking_enabled: bool = True

    @property
    def rag_dir(self) -> Path:
        return self.project_root / ".rag"

    @property
    def index_dir(self) -> Path:
        return self.rag_dir / "index"

    @property
    def chunks_dir(self) -> Path:
        return self.rag_dir / "chunks"

    @property
    def logs_dir(self) -> Path:
        return self.rag_dir / "logs"

    @property
    def context_dir(self) -> Path:
        return self.rag_dir / "context"

    @property
    def prompts_dir(self) -> Path:
        return self.rag_dir / "prompts"


def load_config(root: Path | None = None) -> RAGConfig:
    """Load config from .rag/config.yaml, falling back to defaults."""
    if root is None:
        root = _project_root()
    cfg_path = root / ".rag" / "config.yaml"
    cfg = RAGConfig(project_root=root)

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        for key, val in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

    # Ensure directories exist
    for d in [cfg.index_dir, cfg.chunks_dir, cfg.logs_dir,
              cfg.context_dir, cfg.prompts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return cfg


def save_config(cfg: RAGConfig) -> Path:
    """Serialize current config to .rag/config.yaml."""
    cfg_path = cfg.rag_dir / "config.yaml"
    data = {
        "index_paths": cfg.index_paths,
        "exclude_patterns": cfg.exclude_patterns,
        "chunk_strategy": cfg.chunk_strategy,
        "chunk_size_tokens": cfg.chunk_size_tokens,
        "chunk_overlap_tokens": cfg.chunk_overlap_tokens,
        "embedding_model": cfg.embedding_model,
        "reranker_model": cfg.reranker_model,
        "context_budget_tokens": cfg.context_budget_tokens,
        "retrieval_top_k": cfg.retrieval_top_k,
        "rerank_top_k": cfg.rerank_top_k,
        "inject_top_k": cfg.inject_top_k,
        "min_similarity": cfg.min_similarity,
        "session_type": cfg.session_type,
        "watch_mode": cfg.watch_mode,
        "reranking_enabled": cfg.reranking_enabled,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return cfg_path
