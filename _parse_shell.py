import json, re
from pathlib import Path
p = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl")
text = p.read_text(encoding="utf-8")
# unescape json string chunks - find tool_result with stdout
for line in text.splitlines():
    o = json.loads(line)
    def find_result(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "tool_result" or "tool_result" in str(obj.get("type", "")):
                return obj.get("content") or obj.get("output")
            for v in obj.values():
                r = find_result(v)
                if r: return r
        elif isinstance(obj, list):
            for v in obj:
                r = find_result(v)
                if r: return r
        return None
    r = find_result(o)
    if r and isinstance(r, str) and "315|def build_figure1" in r:
        print("found result len", len(r))
        lines = {}
        for m in re.finditer(r"^ (\d+)\|(.*)$", r, re.M):
            lines[int(m.group(1))] = m.group(2)
        print("lines parsed", len(lines), "max", max(lines))
        break
else:
    print("no numbered output")
    # search raw
    if "315|def build_figure1" in text:
        print("raw has numbered")
