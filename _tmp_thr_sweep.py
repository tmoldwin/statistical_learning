import json, time
import numpy as np
from pathlib import Path

from viz.weight_structure import compute_weight_digraph_motifs, _thresholded_digraph

# Load Dale mixed panels for DFA labels + ckpt paths via run_id
meta = json.loads(Path("experiments/comparisons/mixed_vocab_dfa_ns/data/mixed_dfa_weight_graph_metrics.json").read_text())
panels = meta["panels"]
# pick ~12 runs spanning DFA
order = sorted(range(len(panels)), key=lambda i: panels[i]["n_dfa_states"])
pick = order[:: max(1, len(order)//12)][:12]
print("picked", [(panels[i]["run_id"], panels[i]["n_dfa_states"]) for i in pick])

def load_w(run_id):
    # try rnn_dale then rnn
    for model in ("rnn_dale", "rnn"):
        p = Path(f"experiments/comparisons/mixed_vocab_dfa_ns/checkpoints/r{run_id:02d}/{model}/model_seed1.npz")
        if p.exists():
            z = np.load(p)
            # common keys
            for k in ("W_hh", "w_hh", "Whh", "recurrent", "w_rec"):
                if k in z:
                    return np.asarray(z[k], float), str(p)
            # dump keys
            keys = list(z.keys())
            # guess
            for k in keys:
                a = z[k]
                if getattr(a, "ndim", 0)==2 and a.shape[0]==a.shape[1] and a.shape[0] >= 50:
                    return np.asarray(a, float), f"{p}:{k}"
            raise KeyError(keys)
    raise FileNotFoundError(run_id)

# preload weights
ws = []
xs = []
for i in pick:
    rid = panels[i]["run_id"]
    w, src = load_w(rid)
    ws.append(w); xs.append(panels[i]["n_dfa_states"])
    print(f"r{rid:02d} DFA={panels[i]['n_dfa_states']} H={w.shape[0]} from {src}")
xs = np.array(xs, float)

# threshold modes: mean, and quantiles
modes = [("mean", None)] + [("quantile", q) for q in (0.50, 0.75, 0.90, 0.95, 0.99)]

def dens_and_motifs(w, mode, q):
    g, thr = _thresholded_digraph(w, mode=mode, q=(q or 0.75))
    n = w.shape[0]
    n_e = g.number_of_edges()
    dens = n_e / max(n*(n-1), 1)
    m = compute_weight_digraph_motifs(w, mode=mode, q=(q or 0.75))
    return dens, thr, m["motif_feedforward_rate"], m["motif_cycle_rate"], m["motif_reciprocal_frac"]

print("\n=== Dale threshold sensitivity (n=12 span) ===")
t0 = time.time()
for mode, q in modes:
    dens=[]; ff=[]; cyc=[]; recip=[]; thrs=[]
    for w in ws:
        d, thr, f, c, r = dens_and_motifs(w, mode, q if q is not None else 0.75)
        dens.append(d); ff.append(f); cyc.append(c); recip.append(r); thrs.append(thr)
    dens, ff, cyc, recip = map(np.array, (dens, ff, cyc, recip))
    def r2(y):
        y=np.asarray(y,float); m=np.isfinite(xs)&np.isfinite(y)
        if m.sum()<5: return float("nan")
        r=np.corrcoef(xs[m], y[m])[0,1]; return float(r*r)
    label = f"{mode}" if mode=="mean" else f"q={q}"
    print(f"{label:8s} dens={dens.mean():.3f} thr~{np.mean(thrs):.4f}  "
          f"R2(ff)={r2(ff):.3f} R2(cyc)={r2(cyc):.3f} R2(recip)={r2(recip):.3f}  "
          f"ff={ff.mean():.3f} cyc={cyc.mean():.3f} recip={recip.mean():.3f}")
print(f"elapsed {time.time()-t0:.1f}s")

# Same for non-Dale if ckpts exist for same run ids
print("\n=== non-Dale same runs / thresholds ===")
ws_nd=[]; xs_nd=[]; ok=True
for i in pick:
    rid = panels[i]["run_id"]
    p = Path(f"experiments/comparisons/mixed_vocab_dfa_ns/checkpoints/r{rid:02d}/rnn/model_seed1.npz")
    if not p.exists():
        print("missing", p); ok=False; break
    z=np.load(p)
    w=None
    for k in z.files:
        a=z[k]
        if getattr(a,"ndim",0)==2 and a.shape[0]==a.shape[1]:
            w=np.asarray(a,float); break
    ws_nd.append(w); xs_nd.append(panels[i]["n_dfa_states"])
xs_nd=np.array(xs_nd,float)
if ok:
  for mode, q in modes:
    dens=[]; ff=[]; cyc=[]; recip=[]; thrs=[]
    for w in ws_nd:
        d, thr, f, c, r = dens_and_motifs(w, mode, q if q is not None else 0.75)
        dens.append(d); ff.append(f); cyc.append(c); recip.append(r); thrs.append(thr)
    dens, ff, cyc, recip = map(np.array, (dens, ff, cyc, recip))
    def r2(y, xs=xs_nd):
        y=np.asarray(y,float); m=np.isfinite(xs)&np.isfinite(y)
        r=np.corrcoef(xs[m], y[m])[0,1]; return float(r*r)
    label = f"{mode}" if mode=="mean" else f"q={q}"
    print(f"{label:8s} dens={dens.mean():.3f}  "
          f"R2(ff)={r2(ff):.3f} R2(cyc)={r2(cyc):.3f} R2(recip)={r2(recip):.3f}  "
          f"ff={ff.mean():.3f} cyc={cyc.mean():.3f} recip={recip.mean():.3f}")
