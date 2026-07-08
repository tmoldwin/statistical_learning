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

p = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl")
best = ""
for line in p.read_text(encoding="utf-8").splitlines():
    o = json.loads(line)
    for s in walk_strings(o):
        if "def build_figure1" in s and "def main" in s:
            if len(s) > len(best):
                best = s
        # escaped version in json dump sometimes as literal backslash-n
        if "\\ndef build_figure1" in s and len(s) > len(best):
            try:
                t = bytes(s, "utf-8").decode("unicode_escape")
            except Exception:
                t = s.replace("\\n", "\n").replace('\\"', '"')
            if "def main" in t and len(t) > len(best):
                best = t
print("best len", len(best))
if best:
    # normalize if still escaped
    if "\\n" in best and "\ndef " not in best:
        best = best.replace("\\n", "\n").replace('\\"', '"')
    out = Path("scripts/generate_training_diagram.py")
    out.write_text(best, encoding="utf-8", newline="\n")
    print("wrote", out, "lines", best.count(chr(10)))
