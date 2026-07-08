from pathlib import Path
t = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl").read_text(encoding="utf-8")
for needle in ["from __future__", "REGIMES", "def esc"]:
    print(needle, t.find(needle))
# try decode entire file as if one big escaped string - find longest substring starting with \nfrom __future__
import re
m = re.search(r'(\\nfrom __future__ import annotations\\n(?:\\n|\\\\n|[\w\\"\(\)\[\]:,\.# \+\-\*/=\'\u2192\u00b7]){5000,30000})', t)
print("future match", bool(m), len(m.group(1)) if m else 0)
