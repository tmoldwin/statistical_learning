"""Regenerate README.md and sync figures into docs/figures/."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from readme_figures import README_FIGURES, ReadmeFigure, SHARED_FIGURE_NUMBERS
from experiment import shared_dir

EXP = ROOT / "experiments" / "ten_word_overlap_s"
SRC_PLOTS = EXP / "rnn" / "plots"
SRC_SHARED = shared_dir("ten_word_overlap_s")
FIGURES = ROOT / "docs" / "figures"


def export_vocabulary_pngs() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_diagram_pngs.py")],
        check=True,
        cwd=ROOT,
    )


def sync_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for old in FIGURES.iterdir():
        if old.is_file():
            old.unlink()

    for fig in README_FIGURES:
        if fig.number in SHARED_FIGURE_NUMBERS:
            src = SRC_SHARED / fig.filename()
        else:
            src = SRC_PLOTS / fig.filename()
        dest = FIGURES / fig.filename()
        if not src.is_file():
            raise FileNotFoundError(f"missing plot for README figure {fig.number}: {src}")
        shutil.copy2(src, dest)


def figure_block(fig: ReadmeFigure) -> str:
    path = f"docs/figures/{fig.filename()}"
    parts: list[str] = []
    if fig.lead.strip():
        parts.append(fig.lead.strip())
        parts.append("")
    parts.append(f"![Figure {fig.number}]({path})")
    parts.append("")
    parts.append(f"*Figure {fig.number}. {fig.caption}*")
    parts.append("")
    return "\n".join(parts)


def figure_blocks(start: int, end: int) -> str:
    by_num = {fig.number: fig for fig in README_FIGURES}
    return "\n".join(figure_block(by_num[n]) for n in range(start, end + 1))


def build_readme_body() -> str:
    parts: list[str] = []

    parts.append("""# Statistical Word Learning in a Minimal Character-Level RNN

This repository trains a **small vanilla recurrent neural network** on synthetic text streams designed to mimic infant **statistical learning** of words (Saffran, Aslin, & Newport, 1996). Words are sampled uniformly from a finite vocabulary and concatenated; in the `_s` regimes a space is inserted between words. The model receives **no word labels**, **no boundary tokens**, and **no auxiliary losses** - only next-character cross-entropy.

After training, an exhaustive visualization pipeline asks how the hidden state $\\mathbf{h}_t$ relates to two complementary structures:

1. **Position within the current word** - the in-word prefix since the last space (`h`, `ha`, `hat`, or a space token).
2. **State in the minimal vocabulary DFA** - which equivalence class of lexical continuations remains open.

**Central claim:** a generic next-character predictor develops geometry that **factorizes along both axes**. Prefix length answers "how many letters into the current word?"; DFA state answers "which vocabulary items are still possible given overlapping spellings?"

Related work: [creating_transformer (statistical learning)](https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning) uses the same synthetic regimes with a minimal transformer. This repo is the **RNN** counterpart (Karpathy char-RNN in NumPy).

---

## Quick start

**Requirements:** Python 3.10+, NumPy, Matplotlib, Seaborn, Pandas, SciPy; optional UMAP for one embedding panel.

```bash
pip install -r requirements.txt
pip install matplotlib seaborn pandas scipy umap-learn
python run_experiments.py --only ten_word_overlap_s
```

`run_experiments.py` trains, visualizes, and syncs README figures automatically for `ten_word_overlap_s`.

| Path | Contents |
|------|----------|
| `experiments/<name>/shared/` | Vocabulary trie + DFA (figures 1–2; shared by RNN and transformer) |
| `experiments/<name>/rnn/plots/` | RNN analysis figures (figures 3–19) plus auxiliary plots |
| `experiments/<name>/rnn/learning_dynamics/` | Supplementary training-time videos (not numbered figures) |
| `docs/figures/` | Copies of numbered README figures |

---

## Repository structure

```
task.py              # Corpus generator (word-sampling regimes)
min-char-rnn.py      # Vanilla RNN, BPTT, Adagrad, metrics
visualize.py         # All analysis figures
readme_figures.py    # README figure order and numbered filenames
vocab_diagrams.py    # Trie + minimal DFA construction
experiment.py        # Per-regime hyperparameters
run_experiments.py   # Batch train + visualize
docs/figures/        # README figure copies (1_vocabulary_trie.png, ...)
experiments/<name>/  # input.txt, shared/, rnn/, transformer/
```

---

## Available experiments

| Folder | Words | Notes |
|--------|-------|-------|
| `ten_word_overlap` / `_s` | 10 x 3-letter words | Primary demo; heavy `-at`/`-et` overlap |
| `ten_four_letter_overlap` / `_s` | 10 x 4-letter words | Longer words, more branching |
| `ten_four_letter_overlap_s_dale` | Same 4-letter vocab | Dale's law + ReLU, $h=50$ |
| `six_word_overlap` / `_s` | 6 words | Smaller vocab |
| `twelve_word_overlap` / `_s` | 12 words | Mixed overlap patterns |
| `sixteen_word_overlap` / `_s` | 16 words | Largest default regime |

Each regime has spaced (`_s`) and unspaced variants. Labeling uses explicit spaces when present, otherwise implicit vocabulary word boundaries (with a $\\leq 3$-character fallback).

Default training for main overlap tasks: **50k characters**, **15k steps**, **$h=32$**, **BPTT $=40$**, **50-character visualization window**.

---

## Paper: Learning words with an RNN

The walkthrough below uses **`ten_word_overlap_s`** from the most recent full train + visualize run. Figures 1–2 live in `experiments/ten_word_overlap_s/shared/`; figures 3–19 in `experiments/ten_word_overlap_s/rnn/plots/`; copies in `docs/figures/` as `N_descriptive_name.png` in narrative order.

> **Abstract.** Infants appear to segment fluent speech into words using only distributional statistics - transitional probabilities between syllables - without explicit boundaries (Saffran, Aslin, & Newport, 1996). We train a 32-unit $\\tanh$ RNN on a ten-word corpus with overlapping trigrams and visualize hidden activations, next-character predictions, PCA embeddings, correlation structure, and recurrent vector fields. Two organizing principles emerge in hidden space: timesteps cluster by **in-word prefix** (distance from the last space), and orthogonally by **minimized DFA state** after that prefix. Pairwise distances are substantially smaller within DFA state than between states, even when the current input character matches. The framework links infant statistical-learning theory to mechanistic RNN interpretability.

---

## 1. Introduction

### 1.1 Statistical learning and word segmentation

Jenny Saffran and colleagues showed that eight-month-old infants can discover word-like units from continuous artificial speech streams. Within putative words, adjacent syllables have high transitional probability (TP); across word boundaries, TP drops. No pauses, stress, or semantic cues are required - only **distributional structure**.

Our synthetic task instantiates the same logic at the character level. The generator repeatedly samples a word uniformly from a small vocabulary and appends its letters (with a space between words in `_s` corpora):

```
... hat cat met rat tea eat cat hat ...
```

The learner sees one long string. The only supervision is: predict the next character. The scientific question is whether $\\mathbf{h}_t$ - the RNN's only memory - implicitly encodes (i) **where you are inside the current word** and (ii) **which vocabulary items remain consistent** with the letters read so far.

### 1.2 Why an RNN?

A vanilla RNN compresses the past into a fixed vector and updates it causally:

$$
\\mathbf{h}_t = \\tanh\\!\\left( W_{xh} \\mathbf{x}_t + W_{hh} \\mathbf{h}_{t-1} + \\mathbf{b}_h \\right)
$$

$$
P(\\text{next} = c \\mid \\mathbf{x}_{\\leq t}) = \\mathrm{softmax}\\!\\left( W_{hy} \\mathbf{h}_t + \\mathbf{b}_y \\right)_c
$$

Here $\\mathbf{x}_t$ is the one-hot encoding of the current character. There is no attention, no stack, and no built-in word counter. If word-like structure appears in $\\mathbf{h}_t$, it is because next-character prediction on overlapping words **demands** it.

This makes the RNN a conservative model of incremental statistical learning: one pass, bounded memory, local supervision.

### 1.3 Two axes of organization (preview)

| Axis | What it encodes | How we label timesteps | Signature figures |
|------|-----------------|------------------------|-------------------|
| **Word-boundary / prefix** | Letter index within the current word | `h`, `ha`, `hat`, space | 9, 10, 16 |
| **DFA state** | Equivalence class of surviving words | Minimized automaton state $q_k$ | 12, 17, 18, 19 |

The first axis is **syntactic position** inside a word (how far from the last boundary). The second is **lexical uncertainty** given shared spellings (`cat` vs `hat` vs `mat`). A successful statistical learner needs both.

---

## 2. Task and vocabulary structure

### 2.1 The `ten_word_overlap` regime

Ten three-letter English words with controlled overlap:

| Group | Words |
|-------|-------|
| `-at` family | cat, hat, mat, rat |
| `-et` family | met, pet, net |
| `-ea` / vowel overlap | ate, eat, tea |

Character set: $\\{a,c,e,h,m,n,p,r,t\\}$ plus space. A model that only tracks the last one or two characters confuses branches that share prefixes or suffixes.

### 2.2 Trie and minimal DFA

We compile the vocabulary into classical finite-state structures **before** training and use them only for **analysis labels** (not as model inputs).""")

    parts.append(figure_blocks(1, 2))

    parts.append("""
---

## 3. Model and training

We use Andrej Karpathy's minimal character-level RNN (NumPy, from scratch).

| Hyperparameter | `ten_word_overlap_s` value |
|----------------|----------------------------|
| Hidden units $h$ | 32 |
| Activation | $\\tanh$ |
| Optimizer | Adagrad ($\\eta = 0.1$) |
| BPTT window | 40 characters |
| Training steps | 15,000 |
| Corpus size | 50,000 characters |

Training minimizes sum cross-entropy over each BPTT window. Every 100 steps we also draw stochastic samples and estimate **word error rate** on long rollouts: the fraction of space-delimited tokens not in the vocabulary. This metric asks whether free generation respects the same word units infants extract from streams.

Optional **Dale's law** mode (`--dale` in `ten_four_letter_overlap_s_dale`): fixed excitatory/inhibitory outgoing signs, ReLU activations, softer learning rate.

---

## 4. Results

All panels use the first **50 characters** of the trained corpus unless noted. Tick labels on later figures use **in-word prefix since last space**.

### 4.1 The model learns the stream

We begin by confirming that training succeeds and that generation improves.""")

    parts.append(figure_blocks(3, 4))

    parts.append("""
### 4.2 Learned parameters

Next we inspect the weights directly - the only place lexical structure can be stored.""")

    parts.append(figure_blocks(5, 6))

    parts.append("""
### 4.3 Activations and predictions over time

With parameters fixed, we turn to hidden activations and outputs along the corpus.""")

    parts.append(figure_blocks(7, 10))

    parts.append("""
### 4.4 Hidden-state geometry

Projecting $\\mathbf{h}_t \\in \\mathbb{R}^{32}$ to the plane exposes the two organizing axes visually.""")

    parts.append(figure_blocks(11, 12))

    parts.append("""
Supplementary media (not a numbered figure): [PCA learning dynamics video](experiments/ten_word_overlap_s/rnn/learning_dynamics/hidden_state_pca.mp4) — hidden states on the 50-character window during early training, projected into the **fixed final-model PCA** basis, with playback paced by cumulative displacement. Static Figure 12 shows the endpoint; the video shows how that geometry forms.""")

    parts.append(figure_blocks(13, 16))

    parts.append("""
### 4.5 Correlation structure: prefix and DFA

Finally we quantify similarity between hidden vectors - within and across DFA states.""")

    parts.append(figure_blocks(17, 19))

    parts.append("""
---

## 5. Discussion

### 5.1 Two organizations, one hidden space

The figures support a two-factor description of the learned representation:

1. **Distance from word boundary (in-word prefix).** As the network reads $h \\to ha \\to hat$, $\\mathbf{h}_t$ moves along paths that depend on how many characters have been consumed since the last space. Figures 9-10 and 16 make this explicit: prefix labels predict similarity across the corpus. This mirrors **positional uncertainty** in infant segmentation - early characters in a word are more ambiguous than later ones.

2. **DFA state (lexical hypothesis class).** After merging equivalent prefixes, each timestep has a well-defined state in the minimal vocabulary automaton. Figure 12 shows these states occupy distinct PCA regions; Figures 18-19 show correlation and distance respect DFA partitions even when the current input character is held constant. This is **competition among overlapping words** made geometric: `ca` could still become `cat`; `at` after `c` is a different state than `at` after `h`.

The axes are not redundant. Prefix length is a scalar progress variable; DFA state is a finite partition of lexical knowledge. The RNN must implement both to minimize loss on overlapping trigrams.

### 5.2 Connection to statistical learning theory

Saffran's learners track transitional probabilities and use troughs at boundaries to segment streams. Our model never sees explicit boundaries in the loss, but spaces in `_s` corpora induce bimodal statistics: high TP within words, lower TP across boundaries. The emergent prefix structure in $\\mathbf{h}_t$ is the network's solution to that problem.

The DFA axis goes further: the RNN tracks **which branch of the lexical tree** it occupies - a discrete state machine embedded in continuous hidden space.

### 5.3 Limitations and extensions

- **Small vocabulary, synthetic data** - ten words is a laboratory setting.
- **PCA is a projection** - 32 dimensions collapsed to 2 for plotting; UMAP/t-SNE panels are qualitative.
- **Vanilla RNN** - no LSTM/GRU; long-range dependencies may be harder than in modern architectures.
- **Single run** - figures are illustrative; multiple seeds would strengthen quantitative claims.
- **Unspaced corpora** - implicit word boundaries via vocabulary segmentation work for labeling; see unspaced experiment folders.

---

## 6. Reproduce

```bash
python run_experiments.py --only ten_word_overlap_s
```

This generates the corpus, trains the RNN, writes trie/DFA under `experiments/ten_word_overlap_s/shared/`, numbered RNN plots under `experiments/ten_word_overlap_s/rnn/plots/`, writes `experiments/ten_word_overlap_s/rnn/learning_dynamics/hidden_state_pca.mp4`, and runs `scripts/build_readme.py` to refresh `docs/figures/` and this README.

---

## 7. References

- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926-1928.
- Karpathy, A. [Minimal character-level RNN](https://gist.github.com/karpathy/d4dee566867f8291f086).
- Mahajne, R., et al. [creating_transformer - statistical learning](https://github.com/Raneem-mahajne/creating_transformer/tree/statistical_learning).

---

## License

`min-char-rnn.py` retains the BSD license from the Karpathy gist.
""")

    return "\n\n".join(parts)


def main() -> None:
    export_vocabulary_pngs()
    sync_figures()
    out = ROOT / "README.md"
    out.write_bytes(build_readme_body().encode("utf-8"))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    print(f"synced {len(README_FIGURES)} figures -> {FIGURES}")


if __name__ == "__main__":
    main()
