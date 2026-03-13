#!/bin/sh
# rag-managed: true
# -----------------------------------------------------------------------
# post-session-end.sh — Fires when a Claude Code session ends
#
# Behavior:
#   1. Summarizes session activity into SESSION.md
#   2. Processes the stale file queue and reindexes changed files
#   3. Appends any queued decisions to DECISIONS.md
# -----------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

RAG_DIR=".rag"

if [ ! -f "$RAG_DIR/config.yaml" ]; then
    exit 0
fi

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    exit 0
fi

# Update SESSION.md and reindex stale files
$PYTHON -c "
import sys, json, time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, '.')
from rag.config import load_config
from rag.engine import reindex_stale

cfg = load_config(Path('.'))

# --- Update SESSION.md ---
session_path = cfg.context_dir / 'SESSION.md'

# Read stale queue to determine which files were edited
stale_queue = cfg.chunks_dir / 'stale_queue.txt'
edited_files = set()
if stale_queue.exists():
    for line in stale_queue.read_text(encoding='utf-8').strip().splitlines():
        parts = line.split('\t', 1)
        if len(parts) == 2:
            edited_files.add(parts[1])

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
files_str = ', '.join(sorted(edited_files)[:10]) if edited_files else '(none)'
if len(edited_files) > 10:
    files_str += f' (+{len(edited_files) - 10} more)'

session_content = f'''<!-- rag-managed: true -->
# Session State

*Updated automatically at session end. Edit freely during a session.*

## Last Session
- Date: {now}
- Focus: (update manually or via /rag-decision)
- Files touched: {files_str}

## Active Context
- Current task: (carry forward or update)
- Open questions: (carry forward or update)
'''

session_path.write_text(session_content, encoding='utf-8')

# --- Reindex stale files ---
stats = reindex_stale(cfg)
if stats.get('reindexed', 0) > 0 or stats.get('removed', 0) > 0:
    print(f'Reindexed {stats[\"reindexed\"]} files, removed {stats[\"removed\"]}')

# Clear stale queue
if stale_queue.exists():
    stale_queue.unlink()
" 2>/dev/null || true
