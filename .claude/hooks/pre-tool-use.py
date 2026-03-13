#!/usr/bin/env python3
# rag-managed: true
"""
pre-tool-use.py — Fires before any file read/edit in Claude Code.

Behavior:
    Reads the tool input from stdin (JSON with tool_name and tool_input).
    If the tool targets a specific file, retrieves contextually relevant
    chunks for that file's neighborhood and prints them for injection.

Only activates for file-related tools (Read, Edit, Write).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    # Read hook input from stdin
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        hook_input = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Only fire for file-related tools
    file_tools = {"Read", "Edit", "Write", "read_file", "edit_file", "write_file"}
    if tool_name not in file_tools:
        return

    # Extract the target file path
    file_path = (
        tool_input.get("file_path")
        or tool_input.get("path")
        or tool_input.get("filePath")
    )
    if not file_path:
        return

    # Import RAG engine (lazy to keep non-file hooks fast)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from rag.config import load_config
        from rag.engine import retrieve, assemble_context
    except ImportError:
        return

    cfg = load_config()

    # Build a query from the file path + tool context
    rel_path = file_path
    try:
        rel_path = str(Path(file_path).relative_to(cfg.project_root))
    except (ValueError, TypeError):
        pass

    query = f"context for {rel_path}"

    # Retrieve with a smaller budget (don't overwhelm pre-tool context)
    try:
        chunks = retrieve(query, cfg)
        # Filter out chunks from the target file itself (Claude already has it)
        chunks = [c for c in chunks if c.source_file != rel_path]
        if not chunks:
            return

        budget = min(cfg.context_budget_tokens // 4, 4000)
        assembled = assemble_context(chunks[:3], budget)
        if assembled:
            print(f"<rag-context for=\"{rel_path}\">\n{assembled}\n</rag-context>")
    except Exception:
        pass


if __name__ == "__main__":
    main()
