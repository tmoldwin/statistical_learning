from pathlib import Path
jsonl = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl").read_text(encoding="utf-8")
idx = jsonl.rfind("normalized newlines")
start = jsonl.rfind("@'", 0, idx)
print(repr(jsonl[start:start+20]))
