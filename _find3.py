from pathlib import Path
p = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl")
t = p.read_text(encoding="utf-8")
key = "def measure_word_stream"
i = t.find(key)
print("idx", i, "file len", len(t))
if i >= 0:
    print(t[i-200:i+800])
