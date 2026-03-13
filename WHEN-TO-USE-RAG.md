# When to Use RAG (and When Not To)

RAG is powerful, but it's not always the right tool. This guide helps you decide whether this system is worth the setup cost for your project, or whether a simpler approach will serve you better.

## The core question

> **Is your codebase too large for the LLM to read in a single context window?**

If the answer is no, you probably don't need RAG.

## Quick assessment checklist

Score each item. If your total is **6 or higher**, RAG is likely worth it. Below that, start with a simpler approach.

| # | Question | Yes | No |
|---|----------|:---:|:--:|
| 1 | Is your codebase larger than **50,000 tokens** (~200 files or ~40K lines of code)? | +2 | 0 |
| 2 | Do you frequently ask questions that require context from **files you haven't opened** in the current session? | +2 | 0 |
| 3 | Is the codebase changing frequently enough that a **static summary goes stale** within a day or two? | +1 | 0 |
| 4 | Do you work across **multiple sessions** on the same project and lose context between them? | +1 | 0 |
| 5 | Does the project span **multiple languages, frameworks, or domains** where no single person holds the full mental model? | +1 | 0 |
| 6 | Are there **more than 2-3 people** contributing, making it hard to know what changed and where? | +1 | 0 |
| 7 | Do you find yourself repeatedly explaining the same codebase context to the LLM at the start of each session? | +1 | 0 |
| 8 | Is the project **documentation-heavy** (API docs, specs, design docs) alongside the code? | +1 | 0 |

**Score interpretation:**

| Score | Recommendation |
|-------|---------------|
| **0-2** | Use CLAUDE.md or direct file context. RAG is overkill. |
| **3-5** | Use CLAUDE.md + structured context files. Consider RAG only if you're hitting context limits. |
| **6-8** | RAG will save you significant time. Worth the setup. |
| **9-10** | RAG is strongly recommended. You're leaving value on the table without it. |

## The alternatives, simplest to most complex

### Level 0: Just talk to the LLM

**What it is:** Open Claude Code, ask questions, let it read files as needed.

**When it's enough:**
- Small scripts or single-purpose tools (under 10 files)
- You know exactly which files are relevant and can point Claude to them
- One-off tasks where there's no session to maintain

**Limitations:** Claude has no memory of previous sessions. You re-explain context every time. It only knows about files it has explicitly read.

---

### Level 1: CLAUDE.md

**What it is:** A single markdown file at the project root that Claude reads at the start of every session. Contains project rules, architecture overview, key file locations, and coding conventions.

**When it's enough:**
- Projects with up to ~50 files where the architecture fits in 2-3 pages
- Solo developer who knows the codebase well and just needs Claude to follow conventions
- The codebase is relatively stable (not changing structure daily)

**What to put in it:**
```markdown
# CLAUDE.md
## Overview (2-3 sentences)
## Key files and what they do (10-20 bullet points)
## Coding conventions
## How to test
## Gotchas
```

**Limitations:** Static — you maintain it manually. Can't scale beyond ~2,000 tokens without cluttering Claude's context. No search capability.

---

### Level 2: CLAUDE.md + structured context files

**What it is:** CLAUDE.md plus a few purpose-built markdown files that Claude can reference:
- `docs/ARCHITECTURE.md` — system design and module relationships
- `docs/DECISIONS.md` — why things are the way they are
- `docs/SESSION.md` — what you were working on last (updated manually)

**When it's enough:**
- Projects with 50-200 files
- Small team (2-3 developers) with a shared context file
- The codebase has clear boundaries and you can describe the architecture in a few pages
- You're willing to maintain the context files by hand

**Limitations:** Manual maintenance. No automatic search — Claude only reads these files if you or a hook tells it to. Doesn't scale to large codebases where you can't predict which files are relevant.

---

### Level 3: CLAUDE.md + grep/glob (Claude's built-in tools)

**What it is:** Rely on Claude Code's native file search (Glob, Grep, Read) to find what it needs. Pair with a good CLAUDE.md that tells Claude where to look.

**When it's enough:**
- Projects up to ~500 files where the naming conventions are consistent
- When queries are specific enough that keyword search works ("find the auth middleware", "where is the database connection configured")
- You don't need proactive context injection — Claude searching on-demand is fine

**What to put in CLAUDE.md:**
```markdown
## Where things live
- Auth: src/auth/
- API routes: src/routes/
- Database: src/db/
- Tests: tests/ (mirror src/ structure)
```

**Limitations:** Slow for broad questions ("how does the system handle errors?"). Claude may not know which files to search. No semantic understanding — only finds exact keyword matches. No cross-session memory.

---

### Level 4: This RAG system

**What it is:** Automatic indexing, hybrid retrieval (keyword + semantic), reranking, context injection via hooks, session state management, and slash commands.

**When it's worth it:**
- Codebase over 200 files or 50K tokens
- Broad questions that can't be answered by searching a single directory
- Multi-session projects where context continuity matters
- You want Claude to proactively surface relevant code, not just respond to explicit searches
- Multiple contributors or fast-changing code where manual context files go stale

**What it costs:**
- ~15 minutes to install and configure
- ~80MB disk for the embedding model (downloaded once)
- ~50-200ms per retrieval query (CPU)
- Learning the slash commands and trusting the hooks

---

### Level 5: Full knowledge graph / GraphRAG

**Not included in this repo.** Mentioned for completeness.

**What it is:** Build a knowledge graph from your codebase (entities, relationships, dependencies), then traverse it during retrieval. Microsoft's GraphRAG is the leading implementation.

**When it's worth it:**
- Very large codebases (1M+ lines) with complex inter-module dependencies
- Questions requiring multi-hop reasoning ("what happens when a user signs up, end to end?")
- You have the compute budget for LLM-assisted index construction

**What it costs:**
- Many LLM API calls at index time (expensive)
- Significant setup complexity
- Still experimental for most use cases

---

## Decision flowchart

```
How big is your codebase?
│
├── < 10 files
│   └── Level 0: Just talk to Claude
│
├── 10-50 files
│   └── Level 1: CLAUDE.md
│
├── 50-200 files
│   ├── Can you describe the architecture in 2 pages?
│   │   ├── Yes → Level 2: CLAUDE.md + context files
│   │   └── No  → Level 3: CLAUDE.md + grep/glob
│   │
│   └── Do you work across multiple sessions?
│       ├── Yes → Level 4: RAG (this system)
│       └── No  → Level 2 or 3
│
├── 200-1000 files
│   └── Level 4: RAG (this system)
│
└── 1000+ files
    ├── Need multi-hop reasoning? → Level 5: GraphRAG
    └── Otherwise → Level 4: RAG (this system)
```

## Common mistakes

**Over-engineering small projects.** If your entire codebase fits in 30K tokens, just paste the relevant files into context or let Claude read them. RAG adds latency and complexity for no benefit.

**Under-engineering large projects.** If you're spending the first 5 minutes of every session re-explaining your codebase to Claude, that's a sign you need at least Level 2, probably Level 4.

**Skipping CLAUDE.md.** Even with RAG, you should still have a CLAUDE.md. RAG handles "what does the code say?" — CLAUDE.md handles "what are the rules?" (conventions, testing approach, things to never do). They serve different purposes.

**Treating RAG as a replacement for reading code.** RAG surfaces relevant chunks, but Claude should still read the full file when making edits. RAG helps Claude find the right file — it doesn't replace understanding the file.

## If you're unsure, start simple

1. Write a CLAUDE.md (30 minutes)
2. Use it for a week
3. If you find yourself frequently fighting context limits or re-explaining things, install this RAG system
4. If CLAUDE.md is enough, you saved yourself the setup

The best context management system is the simplest one that works for your project.
