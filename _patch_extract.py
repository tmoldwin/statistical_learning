from pathlib import Path
import re
t = Path(r"C:\Users\toviah.moldwin\.cursor\projects\c-Users-toviah-moldwin-Code-statistical-learning-1\agent-transcripts\637a992e-a3ee-4184-9031-597ef13cc821\subagents\95929ec6-d316-4831-ac9f-09c4388cfc08.jsonl").read_text(encoding="utf-8")
# find patch script between first path.read_text and path.write_text
start = t.find('path = Path(r"C:\\\\Users')
if start < 0:
    start = t.find('generate_training_diagram.py')
print("start", start)
chunk = t[start:start+40000]
# write decoded patch script
decoded = chunk.encode('utf-8').decode('unicode_escape') if '\\n' in chunk[:500] else chunk
# simpler: extract python patch file from json
m = re.search(r'path = Path\(r"[^"]+generate_training_diagram\.py"\)', t)
print("path match", m.start() if m else None)
# find subagent message with full patch - look for BUILD_FIGURE1 or chip_bottom
idx = t.find("chip_bottom = ty + 12")
print("chip idx", idx)
print(t[idx-500:idx+1500])
