<!-- rag-managed: true -->
# /rag-decision

Append a timestamped architectural decision to DECISIONS.md.

Run this command:
```bash
python -c "
from datetime import datetime, timezone
from pathlib import Path

decision = '''$ARGUMENTS'''
if not decision.strip():
    print('Usage: /rag-decision \"Your decision summary here\"')
    exit(1)

decisions_path = Path('.rag/context/DECISIONS.md')
if not decisions_path.exists():
    print('DECISIONS.md not found. Run rag-init.sh first.')
    exit(1)

now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
entry = f'\n### {now}\n{decision.strip()}\n'

with open(decisions_path, 'a', encoding='utf-8') as f:
    f.write(entry)

print(f'Decision logged: {decision.strip()[:80]}')
"
```

After logging, confirm the entry was added and show the last 3 decisions from the file.
