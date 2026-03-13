<!-- rag-managed: true -->
# /rag-context

Show exactly what context would be injected for the current session.

Run this command:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from rag.config import load_config
from rag.engine import retrieve, assemble_context
from rag.chunker import count_tokens

cfg = load_config(Path('.'))

# Load context files
for name in ['PROJECT.md', 'SESSION.md', 'DECISIONS.md']:
    p = cfg.context_dir / name
    if p.exists():
        content = p.read_text(encoding='utf-8')
        tokens = count_tokens(content)
        print(f'{name}: {tokens} tokens')

# Show last retrieval from log
log_path = cfg.logs_dir / 'retrieval.log'
if log_path.exists():
    import json
    lines = log_path.read_text(encoding='utf-8').strip().splitlines()
    if lines:
        last = json.loads(lines[-1])
        print(f'\nLast query: {last[\"query\"]}')
        for r in last['results']:
            print(f'  {r[\"source\"]}:{r[\"lines\"]} (score: {r[\"score\"]}, {r[\"tokens\"]} tokens)')

print(f'\nContext budget: {cfg.context_budget_tokens:,} tokens')
print(f'Inject top-k: {cfg.inject_top_k}')
"
```

Display the output and explain what each component contributes to the context window.
