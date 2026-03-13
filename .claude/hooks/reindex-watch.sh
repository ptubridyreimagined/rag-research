#!/bin/sh
# rag-managed: true
# -----------------------------------------------------------------------
# reindex-watch.sh — Incrementally reindex changed files
#
# Intended to be called by a file watcher (e.g., fswatch, inotifywait)
# or manually after saving files.
#
# Usage:
#   bash .claude/hooks/reindex-watch.sh [file1] [file2] ...
#   fswatch -o src/ | xargs -n1 bash .claude/hooks/reindex-watch.sh
# -----------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".rag/config.yaml" ]; then
    echo "RAG not configured. Run: bash rag-init.sh"
    exit 1
fi

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON="$cmd"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "Python not found."
    exit 1
fi

if [ $# -gt 0 ]; then
    # Reindex specific files
    $PYTHON -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from rag.config import load_config
from rag.engine import index_repo

cfg = load_config(Path('.'))
paths = [Path(f) for f in sys.argv[1:] if Path(f).exists()]
if paths:
    stats = index_repo(cfg, paths=paths)
    print(f'Reindexed {stats[\"files\"]} file(s) -> {stats[\"chunks\"]} chunk(s)')
else:
    print('No valid files to reindex.')
" "$@"
else
    # Reindex all stale files
    $PYTHON rag/engine.py --reindex --root "$SCRIPT_DIR"
fi
