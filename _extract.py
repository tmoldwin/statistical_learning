import json, re
from pathlib import Path
p = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl")
t = p.read_text(encoding="utf-8")
# split on \\n for json-escaped lines containing def 
chunks = re.findall(r'(?:^|[^\\])((?:\\n)?def [a-z_]+[^\\]{0,200})', t)
print("chunks", len(chunks))
for c in ["draw_rnn_schematic", "build_figure1", "make_stream_words"]:
    i = t.find(c)
    print(c, i)
# extract large escaped block: from """ through main
m = re.search(r'\\"\\"\\"Generate slideshow SVGs[^\\]*(?:\\\\n[^\\]*){100,600}', t)
print("doc match", bool(m), len(m.group(0)) if m else 0)
