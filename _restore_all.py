import json
from pathlib import Path

def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_strings(v)

def normalize(s):
    if "\ndef build_figure1" in s:
        return s
    if "\\ndef build_figure1" in s:
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    return ""

root = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts")
best = ""
src = None
for p in root.rglob("*.jsonl"):
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
    except Exception:
        continue
    for line in lines:
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        for s in walk_strings(o):
            t = normalize(s)
            if t and "def main" in t and "measure_word_stream" in t and len(t) > len(best):
                best = t
                src = p
print("best", len(best), src)
