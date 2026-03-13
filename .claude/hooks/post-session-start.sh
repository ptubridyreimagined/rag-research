#!/bin/sh
# rag-managed: true
# -----------------------------------------------------------------------
# post-session-start.sh — Fires when a Claude Code session begins
#
# Behavior:
#   1. Reads PROJECT.md, SESSION.md, and DECISIONS.md
#   2. Checks index staleness and warns if >20%
#   3. Retrieves top-5 chunks for the last session's focus topic
#   4. Outputs assembled context for Claude to ingest
# -----------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

RAG_DIR=".rag"
CONTEXT_DIR="$RAG_DIR/context"

# Bail gracefully if RAG isn't set up
if [ ! -f "$RAG_DIR/config.yaml" ]; then
    exit 0
fi

# Find Python
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

# Build context payload
$PYTHON -c "
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from rag.config import load_config
from rag.engine import detect_staleness, retrieve, assemble_context, get_index_stats

cfg = load_config(Path('.'))
parts = []

# Load context files
for name in ['PROJECT.md', 'SESSION.md', 'DECISIONS.md']:
    p = cfg.context_dir / name
    if p.exists():
        content = p.read_text(encoding='utf-8', errors='replace').strip()
        if content:
            parts.append(f'<{name.replace(\".md\", \"\").lower()}>\n{content}\n</{name.replace(\".md\", \"\").lower()}>')

# Staleness check
stats = get_index_stats(cfg)
if stats['staleness_pct'] > 20:
    parts.append(f'WARNING: Index is {stats[\"staleness_pct\"]}% stale ({stats[\"stale_files\"]} files). Consider running /rag-reindex.')

# Retrieve context for last session topic
session_path = cfg.context_dir / 'SESSION.md'
if session_path.exists():
    session_text = session_path.read_text(encoding='utf-8', errors='replace')
    # Extract last focus topic
    for line in session_text.splitlines():
        if line.startswith('- Focus:') or line.startswith('- Current task:'):
            topic = line.split(':', 1)[1].strip()
            if topic and topic != '(none yet)':
                chunks = retrieve(topic, cfg)
                if chunks:
                    assembled = assemble_context(chunks, min(cfg.context_budget_tokens, 6000))
                    parts.append(f'<retrieved-context topic=\"{topic}\">\n{assembled}\n</retrieved-context>')
                break

# Load session-start prompt
prompt_path = cfg.prompts_dir / 'session-start.md'
if prompt_path.exists():
    parts.insert(0, prompt_path.read_text(encoding='utf-8', errors='replace').strip())

print('\n\n'.join(parts))
" 2>/dev/null || true
