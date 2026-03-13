#!/usr/bin/env python3
# rag-managed: true
"""
post-tool-use.py — Fires after any file write/edit in Claude Code.

Behavior:
    Reads the tool output from stdin. If a file was written or edited,
    marks it as stale in the RAG index for future reindexing.
    Does NOT reindex immediately (too slow for mid-session).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        hook_input = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    # Only fire for write/edit tools
    write_tools = {"Edit", "Write", "edit_file", "write_file", "NotebookEdit"}
    if tool_name not in write_tools:
        return

    file_path = (
        tool_input.get("file_path")
        or tool_input.get("path")
        or tool_input.get("filePath")
    )
    if not file_path:
        return

    # Record the stale file in a simple tracking file
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from rag.config import load_config

        cfg = load_config()
        stale_file = cfg.chunks_dir / "stale_queue.txt"

        rel_path = file_path
        try:
            rel_path = str(Path(file_path).relative_to(cfg.project_root))
        except (ValueError, TypeError):
            pass

        # Append to stale queue (deduped on reindex)
        with open(stale_file, "a", encoding="utf-8") as f:
            f.write(f"{time.time()}\t{rel_path}\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
