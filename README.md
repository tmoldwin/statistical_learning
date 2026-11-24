# Emergent lexical features and network structural remodeling in an RNN trained on next-token prediction for a statistical learning task

**Toviah Moldwin, Raneem Mahajne, and Idan Segev**
Edmond and Lily Safra Center for Brain Sciences, The Hebrew University of Jerusalem

Poster presented at the ELSC Annual Retreat, 2026. [PDF](poster/poster.pdf) · [HTML source](poster/index.html)

## Inspiration · Statistical learning

![Speech has no silences between words](poster/schematics/saffran/no_silences.png)

`bupadokutabakutababupadokutababupado`

**bupado** (word) ? **padoku** (non-word)

## Aims

- Is an RNN trained on next-token prediction able to learn words from an unsegmented stream?
- How is lexical information (word position, character, etc.) represented in a recurrent network that learns to produce words?
- How does the network’s representation of information change with vocabulary complexity?
- How do the network’s weights differ as a function of the complexity of the learned vocabulary (function → structure)?

## Fig 1 · Mixed vocabularies & minimal DFAs

![Fig 1](paper/figures/demo/fig_mixed_vocab_dfa_examples.jpg)

## Fig 2 · Training the RNN

![Fig 2](plots/training_diagram/02_bptt_and_rnn.svg)

## Fig 3 · Learning & generation

![Fig 3](paper/figures/demo/fig03_learning_with_samples.jpg)

## Fig 4 · Next-character probabilities

![Fig 4](paper/figures/demo/fig_next_char_probs.jpg)

## Fig 5 · Activation heatmap

![Fig 5](paper/figures/demo/fig_activation_heatmap.jpg)

## Fig 6 · Activations by prefix

![Fig 6](paper/figures/demo/fig_activation_clustered.jpg)

## Fig 7 · Hidden-state correlation

![Fig 7](paper/figures/demo/fig_state_correlation.jpg)

## Fig 8 · DFA geometry & separation

![Fig 8](paper/figures/demo/fig_dfa_pca_geometry.jpg)

## Fig 9 · Example selective units

![Fig 9](paper/figures/demo/fig_example_units.jpg)

## Fig 10 · Linear decoding

![Fig 10](paper/figures/demo/fig_decoding_with_word.jpg)

## Fig 11 · Word trajectories

![Fig 11](paper/figures/demo/fig_word_trajectories.jpg)

## Fig 12 · Activations & scaling vs DFA size

![Fig 12](paper/figures/compare/fig_mixed_scaling_overview.jpg)

## Fig 13 · Readouts vs DFA size

![Fig 13](paper/figures/compare/fig_mixed_decoding_curves.jpg)

## Fig 14 · Geometry & unit selectivity

![Fig 14](paper/figures/compare/fig_mixed_cosine_within.jpg)

## Fig 15 · Weight matrices vs DFA

![Fig 15](paper/figures/main/fig_weight_matrices_by_dfa.jpg)

## Fig 16 · Weight graph motifs vs DFA

![Fig 16](paper/figures/main/fig_weight_graph_motifs_vs_dfa.jpg)

## Key references

Saffran, Aslin & Newport (1996); Elman (1990); Aslin, Saffran & Newport (1998); Goldwater et al. (2009); Perruchet & Vinter (1998).

Contact: Toviah Moldwin

---

# Code

Unified repository for **statistical learning** experiments on synthetic word vocabularies.
The same corpora and analysis pipeline are run with:

- **RNN** — minimal character-level vanilla RNN (NumPy, Karpathy-style)
- **Transformer** — small causal character transformer (PyTorch)

Outputs are organized for side-by-side comparison:

```
experiments/<regime>/
  input.txt              # shared training corpus
  shared/                # trie + DFA (same for RNN and transformer)
  rnn/
    model.npz
    plots/
  transformer/
    model.pt
    plots/
```

## Quick start

```bash
pip install -r requirements.txt
python run_experiments.py --only ten_word_overlap_s --smoke
```

Remove `--smoke` for full training (15k steps, ~minutes per model).

## Run one model only

```bash
python run_experiments.py --only ten_word_overlap_s --models rnn
python run_experiments.py --only ten_word_overlap_s --models transformer
```

## Available regimes

Same word vocabularies as the original RNN repo (`task.py`):

| Folder | Words | Notes |
|--------|-------|-------|
| `ten_word_overlap` / `_s` | 10 × 3-letter | Primary demo |
| `ten_four_letter_overlap` / `_s` | 10 × 4-letter | Longer words |
| `six_word_overlap` / `_s` | 6 words | Smaller vocab |
| `twelve_word_overlap` / `_s` | 12 words | Mixed overlap |
| `sixteen_word_overlap` / `_s` | 16 words | Largest default |

`_s` variants insert spaces between words.

## Visualization

The RNN uses hidden-state geometry plots (`visualize.py --model-type rnn`).

The transformer uses **representation-specific** analysis: token embeddings, position
embeddings, block input, per-layer Q/K/V, attention weights, and block output (pre-`lm_head`)
are plotted separately — not as a single “hidden state.”

```bash
python visualize.py --exp ten_word_overlap_s --model-type rnn
python visualize.py --exp ten_word_overlap_s --model-type transformer
```

RNN-only plots (weight matrices, recurrent vector field) are skipped for the transformer.

## Repository layout

```
task.py              # Corpus generator (shared)
experiment.py        # Per-regime hyperparameters + directory layout
run_experiments.py   # Train + visualize both models
visualize.py         # Shared analysis (RNN plots + transformer adapter)
vocab_diagrams.py    # Trie + minimal DFA
readme_figures.py    # Numbered README figure names

rnn/
  min_char_rnn.py    # Vanilla RNN training
  rnn_dyn.py         # RNN dynamics helpers

transformer/
  model.py           # Minimal causal transformer
  train.py           # Character-level training
  adapter.py         # Extract Q/K/V, embeddings, attention for analysis
  viz_repr.py        # Representation-specific transformer plots
  data_char.py       # Char encoding / batching
```

## References

- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*.
- Karpathy minimal char-RNN: https://gist.github.com/karpathy/d4dee566867f8291f086
- Source repos merged from [rnn](https://github.com/Raneem-mahajne/rnn) and [creating_transformer](https://github.com/Raneem-mahajne/creating_transformer) (`statistical_learning` branch).
