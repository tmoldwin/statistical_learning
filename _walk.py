import json, codecs
from pathlib import Path
p = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl")
pieces = []
for line in p.read_text(encoding="utf-8").splitlines():
    o = json.loads(line)
    s = json.dumps(o)
    if "measure_word_stream" in s:
        pieces.append((len(s), o))
print("pieces", len(pieces))
# walk recursively
def walk(obj, path=""):
    if isinstance(obj, str):
        if obj.count("\ndef ") > 5 and "build_figure1" in obj:
            return obj
        if "\\ndef " in obj and "build_figure1" in obj:
            return obj.encode().decode("unicode_escape")
    if isinstance(obj, dict):
        for k,v in obj.items():
            r = walk(v, path+"."+k)
            if r: return r
    if isinstance(obj, list):
        for i,v in enumerate(obj):
            r = walk(v, path+f"[{i}]")
            if r: return r
    return None
for _, o in pieces:
    r = walk(o)
    if r:
        print("found walk", len(r))
        Path("_piece.py").write_text(r[:5000], encoding="utf-8")
        break
