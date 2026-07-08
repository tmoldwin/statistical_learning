import json
from pathlib import Path
root = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts")
best = ("", 0, None)
for p in root.rglob("*.jsonl"):
    t = p.read_text(encoding="utf-8")
    if "measure_word_stream" not in t:
        continue
    # find longest contiguous python-like stretch
    idx = t.find("def measure_word_stream")
    if idx >= 0:
        start = t.rfind('"""Generate slideshow', 0, idx)
        if start < 0:
            start = t.rfind("Generate slideshow SVGs", 0, idx) - 10
        end = t.find("if __name__", idx)
        if end > 0:
            end = t.find("main()", end) + 20
            chunk = t[start:end]
            # unescape json
            try:
                chunk2 = json.loads('"' + chunk.replace('\\', '\\\\').replace('"', '\\"')[:100] + '"')
            except Exception:
                pass
            if len(chunk) > best[1]:
                best = (chunk, len(chunk), p)
print("best", best[1], best[2])
