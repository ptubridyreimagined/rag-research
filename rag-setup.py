#!/usr/bin/env python3
# rag-managed: true
"""
Interactive RAG setup wizard — asks 3-5 questions, writes .rag/config.yaml.
Re-runnable without losing the existing index.

Usage:
    python rag-setup.py              # interactive mode
    python rag-setup.py --defaults   # accept all defaults non-interactively
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _read_gitignore(root: Path) -> list[str]:
    """Parse .gitignore for directory patterns to suggest exclusions."""
    gi = root / ".gitignore"
    if not gi.exists():
        return []
    patterns = []
    for line in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line.rstrip("/"))
    return patterns


def _detect_project_type(root: Path) -> str:
    if (root / "package.json").exists():
        return "node"
    if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
        return "python"
    if (root / "Cargo.toml").exists():
        return "rust"
    if (root / "go.mod").exists():
        return "go"
    return "generic"


def _suggest_index_paths(root: Path) -> list[str]:
    """Suggest directories likely containing source code."""
    candidates = ["src", "lib", "app", "pkg", "cmd", "internal", "api", "core"]
    found = [d for d in candidates if (root / d).is_dir()]
    if not found:
        return ["."]
    return found


def _suggest_excludes(root: Path) -> list[str]:
    """Merge default excludes with .gitignore patterns."""
    defaults = [
        "node_modules", "__pycache__", ".git", ".rag/index", ".rag/chunks",
        "dist", "build", ".venv", "venv", "*.pyc", "*.min.js", "*.map",
        "package-lock.json", "yarn.lock", "poetry.lock", "*.lock",
        ".next", ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
        "target", "*.exe", "*.dll", "*.so", "*.dylib",
    ]
    gi_patterns = _read_gitignore(root)
    combined = list(dict.fromkeys(defaults + gi_patterns))
    return combined


def ask(prompt: str, default: str, use_defaults: bool = False) -> str:
    if use_defaults:
        print(f"  {prompt} [{default}]")
        return default
    try:
        answer = input(f"  {prompt} [{default}]: ").strip()
        return answer if answer else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def main():
    parser = argparse.ArgumentParser(description="RAG Setup Wizard")
    parser.add_argument("--defaults", action="store_true", help="Accept all defaults")
    parser.add_argument("--root", type=str, help="Project root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    rag_dir = root / ".rag"
    cfg_path = rag_dir / "config.yaml"
    use_defaults = args.defaults

    project_type = _detect_project_type(root)
    suggested_paths = _suggest_index_paths(root)
    suggested_excludes = _suggest_excludes(root)

    # Load existing config if re-running
    existing: dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        print(f"\n  Found existing config at {cfg_path}")
        print("  Re-running wizard — existing index will be preserved.\n")
    else:
        print()

    print("=" * 60)
    print("  RAG Setup Wizard")
    print(f"  Project: {root.name} ({project_type})")
    print("=" * 60)
    print()

    # --- Question 1: Index paths ---
    default_paths = ", ".join(existing.get("index_paths", suggested_paths))
    print("  Q1: Which directories should be indexed?")
    print(f"      Detected source dirs: {', '.join(suggested_paths)}")
    raw_paths = ask("Directories (comma-separated)", default_paths, use_defaults)
    index_paths = [p.strip() for p in raw_paths.split(",") if p.strip()]

    # --- Question 2: Chunk strategy ---
    print()
    print("  Q2: Chunking strategy?")
    print("      semantic  — Split on code/document structure (recommended)")
    print("      fixed     — Fixed-size token windows with overlap")
    print("      hybrid    — Semantic where possible, fixed-size fallback")
    default_strategy = existing.get("chunk_strategy", "semantic")
    chunk_strategy = ask("Strategy", default_strategy, use_defaults)
    if chunk_strategy not in ("semantic", "fixed", "hybrid"):
        print(f"    Unknown strategy '{chunk_strategy}', using 'semantic'")
        chunk_strategy = "semantic"

    # --- Question 3: Context budget ---
    print()
    print("  Q3: Context window budget (tokens reserved for RAG)?")
    print("      This is how many tokens of retrieved context to inject.")
    print("      Default is 20% of 200K = 40,000 tokens.")
    default_budget = str(existing.get("context_budget_tokens", 40000))
    budget_str = ask("Token budget", default_budget, use_defaults)
    try:
        context_budget = int(budget_str)
    except ValueError:
        context_budget = 40000

    # --- Question 4: Session type ---
    print()
    print("  Q4: Typical session length?")
    print("      short   — Quick chat, single task (lighter context loading)")
    print("      long    — Multi-day project work (loads full session state)")
    default_session = existing.get("session_type", "short")
    session_type = ask("Session type", default_session, use_defaults)
    if session_type not in ("short", "long"):
        session_type = "short"

    # --- Question 5: Reranking ---
    print()
    print("  Q5: Enable cross-encoder reranking? (Higher precision, +50ms latency)")
    default_rerank = "yes" if existing.get("reranking_enabled", True) else "no"
    rerank_answer = ask("Enable reranking (yes/no)", default_rerank, use_defaults)
    reranking_enabled = rerank_answer.lower() in ("yes", "y", "true", "1")

    # --- Build config ---
    overlap = 50 if chunk_strategy == "fixed" else 0
    config = {
        "index_paths": index_paths,
        "exclude_patterns": suggested_excludes,
        "chunk_strategy": chunk_strategy,
        "chunk_size_tokens": 400,
        "chunk_overlap_tokens": overlap,
        "embedding_model": existing.get("embedding_model", "all-MiniLM-L6-v2"),
        "reranker_model": existing.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        "context_budget_tokens": context_budget,
        "retrieval_top_k": 20,
        "rerank_top_k": 5,
        "inject_top_k": 3 if session_type == "short" else 5,
        "min_similarity": 0.25,
        "session_type": session_type,
        "watch_mode": False,
        "reranking_enabled": reranking_enabled,
    }

    # --- Write config ---
    rag_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print()
    print("=" * 60)
    print(f"  Config written to: {cfg_path}")
    print()
    print("  Settings:")
    print(f"    Index paths:    {', '.join(index_paths)}")
    print(f"    Strategy:       {chunk_strategy}")
    print(f"    Budget:         {context_budget:,} tokens")
    print(f"    Session type:   {session_type}")
    print(f"    Reranking:      {'enabled' if reranking_enabled else 'disabled'}")
    print(f"    Embedding:      {config['embedding_model']}")
    print("=" * 60)
    print()
    print("  Next: run 'python rag/engine.py --index' to build the index.")
    print()


if __name__ == "__main__":
    main()
