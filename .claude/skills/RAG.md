<!-- rag-managed: true -->
# RAG System Skill

You have access to a project-level RAG system that indexes the codebase and
retrieves relevant context. This skill teaches you how to use it effectively.

## Available Commands

| Command | Purpose |
|---|---|
| `/rag-search <query>` | Search the index for relevant code/docs |
| `/rag-reindex` | Reindex stale files |
| `/rag-status` | Show index health stats |
| `/rag-context` | Show what context is currently loaded |
| `/rag-forget <path>` | Remove a path from the index |
| `/rag-decision "..."` | Log an architectural decision |

## When to Use /rag-search Proactively

Search the index **before** answering questions about:
- How a specific feature is implemented
- Where a function/class/pattern is used
- What dependencies or imports a module has
- Architectural patterns in the codebase

Do NOT search when:
- You already have the relevant file open (from a Read tool call)
- The user is asking about something outside the codebase
- The question is about general programming knowledge

## Interpreting Retrieval Scores

Scores come from hybrid retrieval (BM25 + dense) with cross-encoder reranking.

| Score range | Confidence | Action |
|---|---|---|
| > 0.7 | High | Trust the retrieved context |
| 0.4 - 0.7 | Medium | Use as a starting point, verify by reading the file |
| 0.25 - 0.4 | Low | Mention to the user that confidence is low |
| < 0.25 | Filtered out | Not shown — below minimum threshold |

When scores are consistently low for a topic, it likely means:
- The relevant code isn't indexed (check `/rag-status`)
- The index is stale (run `/rag-reindex`)
- The query needs to be rephrased with more specific terms

## Updating SESSION.md

SESSION.md tracks state across sessions. Update it when:
- The user shifts to a new task or focus area
- You discover important context about the current work
- The user mentions open questions or blockers

Format:
```markdown
## Last Session
- Date: (auto-updated at session end)
- Focus: Brief description of main task
- Files touched: comma-separated list

## Active Context
- Current task: What the user is working on
- Open questions: Unresolved issues to carry forward
```

## Updating DECISIONS.md

Use `/rag-decision` to log decisions when:
- The user makes an explicit architectural choice
- A significant tradeoff is discussed and resolved
- A pattern or convention is established for the project

Do NOT log:
- Routine code changes or bug fixes
- Temporary decisions that will change soon
- Decisions already documented elsewhere (README, CLAUDE.md)

## When to Trigger Manual Reindex

Run `/rag-reindex` when:
- `/rag-status` shows staleness > 20%
- You edited multiple files and need fresh context
- The user reports that retrieved context seems outdated
- Starting a new session after significant changes

Do NOT reindex:
- Mid-conversation for a single file edit (the hook handles this)
- When the user is in a time-sensitive flow

## Handling Context Contradictions

If retrieved context contradicts the user's stated intent:

1. **Flag it explicitly**: "The retrieved context from `file:lines` suggests X,
   but you mentioned wanting Y. Which should I follow?"
2. **Show the source**: Always cite the file and line range
3. **Prefer the user**: If the user confirms their intent, follow it and
   consider logging the decision with `/rag-decision`
4. **Check staleness**: The contradiction may be due to stale index data

## Context Budget

The system enforces a strict token budget (configured in `.rag/config.yaml`).
Retrieved chunks are trimmed to fit. If you need more context than what's
injected, use `/rag-search` with a specific query to get additional results.

Never try to override or work around the token budget — it exists to keep
responses fast and focused.
