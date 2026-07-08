from pathlib import Path
p = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl")
t = p.read_text(encoding="utf-8")
for needle in ['Generate slideshow SVGs', '\\\\n\\\\\"\\\\\\\"Generate slideshow']:
    print(needle, t.find(needle))
start = t.find('Generate slideshow SVGs')
print("start", start)
# find end main
end = t.find('if __name__ == \\"__main__\\"', start)
if end < 0:
    end = t.find('if __name__', start)
print("end", end)
