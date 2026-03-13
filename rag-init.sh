#!/bin/sh
# rag-managed: true
# -----------------------------------------------------------------------
# rag-init.sh — One-command RAG system installer
#
# Detects project type, installs dependencies, scaffolds .rag/ directory,
# runs the interactive setup, builds the initial index, and registers
# Claude Code hooks.
#
# Usage:  bash rag-init.sh [--defaults]
#         --defaults  Accept all setup defaults non-interactively
# -----------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors (if terminal supports them)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

info()  { printf "${BLUE}[INFO]${NC}  %s\n" "$1"; }
ok()    { printf "${GREEN}[OK]${NC}    %s\n" "$1"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; }
fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$1"; exit 1; }

DEFAULTS_FLAG=""
if [ "$1" = "--defaults" ]; then
    DEFAULTS_FLAG="--defaults"
fi

# -----------------------------------------------------------------------
# Step 1: Detect project type
# -----------------------------------------------------------------------
info "Detecting project type..."

PROJECT_TYPE="generic"
if [ -f "package.json" ]; then
    PROJECT_TYPE="node"
elif [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    PROJECT_TYPE="python"
elif [ -f "Cargo.toml" ]; then
    PROJECT_TYPE="rust"
elif [ -f "go.mod" ]; then
    PROJECT_TYPE="go"
fi

ok "Project type: $PROJECT_TYPE"

# -----------------------------------------------------------------------
# Step 2: Check Python availability
# -----------------------------------------------------------------------
info "Checking Python installation..."

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        version=$("$cmd" --version 2>&1 | head -1)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            ok "Found $version ($cmd)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    fail "Python 3.10+ is required but not found. Install it and retry."
fi

# -----------------------------------------------------------------------
# Step 3: Install Python dependencies
# -----------------------------------------------------------------------
info "Installing RAG dependencies..."

# Use pip to install (prefer --user if not in a venv)
PIP_FLAGS=""
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_PREFIX" ]; then
    PIP_FLAGS="--user"
    warn "No virtual environment detected. Installing with --user flag."
fi

$PYTHON -m pip install $PIP_FLAGS --quiet --upgrade \
    "sentence-transformers>=2.2.0" \
    "chromadb>=0.4.0" \
    "rank-bm25>=0.2.2" \
    "PyYAML>=6.0" \
    2>&1 | tail -5

# Optional: tiktoken for precise token counting
$PYTHON -m pip install $PIP_FLAGS --quiet "tiktoken>=0.5.0" 2>/dev/null || \
    warn "tiktoken not installed — using approximate token counting"

ok "Dependencies installed"

# -----------------------------------------------------------------------
# Step 4: Scaffold .rag/ directory
# -----------------------------------------------------------------------
info "Scaffolding .rag/ directory..."

mkdir -p .rag/index .rag/chunks .rag/context .rag/prompts .rag/logs

# Create retrieval.log placeholder
if [ ! -f ".rag/logs/retrieval.log" ]; then
    touch .rag/logs/retrieval.log
fi

# SESSION.md
if [ ! -f ".rag/context/SESSION.md" ]; then
    cat > .rag/context/SESSION.md << 'SESSIONEOF'
<!-- rag-managed: true -->
# Session State

*Updated automatically at session end. Edit freely during a session.*

## Last Session
- Date: (none yet)
- Focus: (none yet)
- Files touched: (none yet)

## Active Context
- Current task: (none yet)
- Open questions: (none yet)
SESSIONEOF
fi

# DECISIONS.md
if [ ! -f ".rag/context/DECISIONS.md" ]; then
    cat > .rag/context/DECISIONS.md << 'DECISIONSEOF'
<!-- rag-managed: true -->
# Architectural Decisions

*Append-only log. Each entry records a significant technical decision.*
*Format: date, decision summary, rationale.*

<!-- Entries below this line -->
DECISIONSEOF
fi

# retrieval.md prompt
cat > .rag/prompts/retrieval.md << 'RETRIEVALEOF'
<!-- rag-managed: true -->
# Retrieved Context

The following code and documentation chunks were retrieved from the project
index based on relevance to the current task. They are ordered by relevance
score (most relevant first and last, least relevant in the middle).

**Instructions:**
- Use this context to inform your responses about the codebase
- If a chunk contradicts what the user says, flag the discrepancy
- Cite sources as `file:line` when referencing retrieved content
- If the retrieved context seems insufficient, suggest running `/rag-search`
  with a more specific query

---

RETRIEVALEOF

# session-start.md prompt
cat > .rag/prompts/session-start.md << 'SESSIONSTARTEOF'
<!-- rag-managed: true -->
# Session Start Context

Load the following in order:
1. **PROJECT.md** — project overview and structure
2. **SESSION.md** — state from the last session
3. **DECISIONS.md** — architectural decisions to respect
4. **Retrieved chunks** — relevant code for the last known task

## Guidelines
- Greet the user and briefly summarize what was happening last session
- If SESSION.md shows open questions, proactively address them
- If the project index is >20% stale, suggest running `/rag-reindex`
- Prefer retrieved context over assumptions about the codebase
SESSIONSTARTEOF

ok "Directory structure created"

# -----------------------------------------------------------------------
# Step 5: Run interactive setup
# -----------------------------------------------------------------------
info "Running setup wizard..."
$PYTHON rag-setup.py --root "$SCRIPT_DIR" $DEFAULTS_FLAG

# -----------------------------------------------------------------------
# Step 6: Generate PROJECT.md
# -----------------------------------------------------------------------
info "Generating project summary..."
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from rag.config import load_config
from rag.engine import generate_project_summary
from pathlib import Path
cfg = load_config(Path('$SCRIPT_DIR'))
summary = generate_project_summary(cfg)
p = cfg.context_dir / 'PROJECT.md'
p.write_text(summary, encoding='utf-8')
print(f'  Written to: {p}')
"
ok "PROJECT.md generated"

# -----------------------------------------------------------------------
# Step 7: Build initial index
# -----------------------------------------------------------------------
info "Building initial index (this may take a minute on first run)..."
$PYTHON rag/engine.py --index --root "$SCRIPT_DIR"
ok "Index built"

# -----------------------------------------------------------------------
# Step 8: Register Claude Code hooks
# -----------------------------------------------------------------------
info "Registering Claude Code hooks..."

mkdir -p .claude/hooks .claude/commands .claude/skills

# Make hook scripts executable
chmod +x .claude/hooks/*.sh 2>/dev/null || true
chmod +x .claude/hooks/*.py 2>/dev/null || true

# Write Claude Code settings with hook registrations
SETTINGS_FILE=".claude/settings.json"
if [ -f "$SETTINGS_FILE" ]; then
    info "Found existing $SETTINGS_FILE — hooks must be registered manually."
    warn "Add hook entries from .claude/settings.json.rag-hooks to your settings."
    cat > .claude/settings.json.rag-hooks << 'HOOKSEOF'
{
  "hooks": {
    "PostSessionStart": [{"command": "bash .claude/hooks/post-session-start.sh"}],
    "PreToolUse": [{"command": "python3 .claude/hooks/pre-tool-use.py"}],
    "PostToolUse": [{"command": "python3 .claude/hooks/post-tool-use.py"}],
    "PostSessionEnd": [{"command": "bash .claude/hooks/post-session-end.sh"}]
  }
}
HOOKSEOF
else
    cat > "$SETTINGS_FILE" << 'SETTINGSEOF'
{
  "hooks": {
    "PostSessionStart": [
      {
        "command": "bash .claude/hooks/post-session-start.sh",
        "description": "RAG: Load project context and relevant chunks at session start"
      }
    ],
    "PreToolUse": [
      {
        "command": "python3 .claude/hooks/pre-tool-use.py",
        "description": "RAG: Retrieve context for file operations"
      }
    ],
    "PostToolUse": [
      {
        "command": "python3 .claude/hooks/post-tool-use.py",
        "description": "RAG: Mark edited files as stale"
      }
    ],
    "PostSessionEnd": [
      {
        "command": "bash .claude/hooks/post-session-end.sh",
        "description": "RAG: Summarize session and update session state"
      }
    ]
  }
}
SETTINGSEOF
    ok "Hooks registered in $SETTINGS_FILE"
fi

# -----------------------------------------------------------------------
# Step 9: Update .gitignore
# -----------------------------------------------------------------------
info "Updating .gitignore..."

GITIGNORE_LINES="
# --- RAG system (auto-added by rag-init.sh) ---
.rag/index/
.rag/chunks/
.rag/logs/
"

if [ -f ".gitignore" ]; then
    if ! grep -q ".rag/index/" .gitignore 2>/dev/null; then
        printf "%s" "$GITIGNORE_LINES" >> .gitignore
        ok "Appended RAG exclusions to .gitignore"
    else
        ok ".gitignore already has RAG exclusions"
    fi
else
    printf "%s" "$GITIGNORE_LINES" > .gitignore
    ok "Created .gitignore with RAG exclusions"
fi

# -----------------------------------------------------------------------
# Step 10: CI integration (if GitHub Actions detected)
# -----------------------------------------------------------------------
if [ -d ".github/workflows" ] || [ -d ".github" ]; then
    info "GitHub detected — CI workflow available at .github/workflows/rag-reindex.yml"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "================================================================"
echo ""
printf "  ${GREEN}${BOLD}RAG system installed successfully!${NC}\n"
echo ""
echo "  Installed:"
echo "    - Python dependencies (sentence-transformers, chromadb, rank-bm25)"
echo "    - .rag/ directory with config, context files, and prompts"
echo "    - Initial index built from your repository"
echo "    - Claude Code hooks registered"
echo ""
echo "  Key files to review:"
echo "    - .rag/config.yaml          — RAG configuration"
echo "    - .rag/context/PROJECT.md   — auto-generated project summary (editable)"
echo "    - .rag/context/DECISIONS.md — architectural decisions log"
echo ""
echo "  Available commands in Claude Code:"
echo "    /rag-search <query>   — Search the index"
echo "    /rag-reindex          — Reindex stale files"
echo "    /rag-status           — Show index stats"
echo "    /rag-context          — Show what context would be injected"
echo "    /rag-forget <path>    — Remove a path from the index"
echo "    /rag-decision \"...\"   — Log an architectural decision"
echo ""
echo "  To verify:"
printf "    ${BOLD}python rag/engine.py --query \"main entry point\"${NC}\n"
echo ""
echo "================================================================"
