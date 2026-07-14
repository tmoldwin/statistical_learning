# Statistical Word Segmentation as Emergent Structure in a Next-Character RNN

**Working title** · Hidden size \(h = 50\) throughout

---

## Abstract

Eight-month-old infants can segment continuous speech by tracking transitional probabilities between syllables (Saffran, Aslin, & Newport, 1996). We ask whether a vanilla Elman RNN trained only on next-character prediction develops internal representations aligned with word structure. After learning, the network generates legal vocabulary items, and its hidden states become a continuous embedding of the vocabulary’s minimal DFA.

A five-word demo (*cat*, *met*, *ate*, *tea*, *eat*) introduces learning and generated text. On a 16-word, 4-letter condition, DFA state explains \(\eta^2 \approx 0.95\) of condensed hidden variance and is linearly decodable from a few principal components (mean ± std across six seeds). Word trajectories form labeled geometric motifs. Weight structure on that same condition shows letter-columnar \(W_{xh}\) and locally clumped \(W_{hh}\). A length × vocabulary-size sweep (\(H{=}100\); word counts \(5\)–\(25\) in steps of \(5\)) then shows that hidden dimensionality, training cost, and input vs recurrent weight balance track minimized DFA size—driven mainly by the number of words, and secondarily by word length / mixing.

---

## 1. Introduction

Fluent speech arrives without reliable pauses. Infants can use transitional probabilities to find word-like units (Saffran et al., 1996; Aslin, Saffran, & Newport, 1998). Computational accounts range from Bayesian segmentation and chunking (Goldwater, Griffiths, & Johnson, 2009; Perruchet & Vinter, 1998; French, Addyman, & Mareschal, 2011) to predictive sequence models (Elman, 1990).

For a finite vocabulary streamed without separators, optimal next-character prediction depends on the state of the vocabulary’s minimal DFA—the equivalence class of in-word prefixes with identical futures. An Elman RNN has no word units and no boundary channel, yet if it solves the prediction task its hidden state \(\mathbf{h}_t\) must carry that information. We test whether the information is geometrically organized.

**Plan.** (1) Five-word demo. (2) Single 16-word condition: next-character probabilities, activations, separation, single-unit selectivity, decoding, trajectories, then weight structure. (3) Comparisons across length and vocabulary size.

---

## 2. Methods

**Demo lexicon** (`five_word_overlap_ns`): cat, met, ate, tea, eat (vowels *a*/*e*; overlapping structure, not a single shared suffix).

**Main condition** (`sixteen_word_four_letter_ns`): bake, cake, lake, rake, bank, tank, rank, sank, late, mate, rate, gate, cant, pant, rant, want.

**Comparisons.** Length × vocabulary-size sweep at \(H{=}100\) (word counts \(5\)–\(25\) step \(5\); lengths \(1\)–\(6\) plus mixed). Spectra average seeds \(1\)–\(5\); metric-vs-DFA scatters use seeds \(1\)–\(15\). Smaller seed sets \(\{1,2,3\}\) for trajectory grids; decoding aggregates seeds \(\{1,2,3,5,7,8\}\); main-condition weight metrics use all available checkpoints (\(n=16\)).

**Model.** Elman RNN, \(H = 50\), next-character cross-entropy, early stop on word-error \(\leq 3\%\).

**Analyses.** Softmax next-character probabilities; activation heatmaps; hierarchical clustering of timesteps; PCA embeddings (colored by DFA state, position, and character); feature separation (\(\eta^2\), silhouette, …); per-unit selectivity with exemplar units; linear decoding from top-\(k\) PCs or random neurons (chance-corrected; mean ± std across seeds); closed-loop word trajectories; multi-seed clustered init-vs-final weights and motif scalars.

---

## 3. Results

### 3.1 Five-word demo

![Learning curve](/paper/figures/demo/fig03_learning_curve.jpg)

**Figure 1.** Learning curve (truncated near the validity plateau).

![Vocabulary and training stream](/paper/figures/demo/fig04_corpus_stream.jpg)

**Figure 2.** Vocabulary and unsegmented training stream.

![Generation before vs after learning](/paper/figures/demo/fig04_samples.jpg)

**Figure 3.** Generation before vs after learning (green = in-vocabulary, red = out-of-vocabulary).

![DFA and PCA geometry](/paper/figures/demo/fig_dfa_pca_geometry.jpg)

**Figure 4.** Minimal DFA for *cat*, *met*, *ate*, *tea*, *eat* (top-left; nodes colored by state) beside PCA of \(\mathbf{h}\) colored by the same DFA states (top-right). Bottom: same PCA colored by position from beginning (left) and current character (right).

State colors match between the automaton and the DFA-colored PCA. The same geometry reorganized by position and character shows orthogonal feature structure without a separate legend.

Unless noted, the remainder uses the **16-word, 4-letter** condition.

### 3.2 Next-character probabilities

![Next-character probabilities](/paper/figures/main/fig_next_char_probs.jpg)

**Figure 5.** Softmax \(P(\text{next char} \mid \text{input so far})\) over time.

Probability mass concentrates late in words and spreads at ambiguous prefixes.

### 3.3 Hidden states and clustering

![Activation heatmap](/paper/figures/main/fig_activation_heatmap.jpg)

**Figure 6.** Activations over timesteps (x = input letters; units clustered).

![Activations clustered by prefix](/paper/figures/main/fig_activation_clustered.jpg)

**Figure 7.** Activations clustered by in-word prefix.

### 3.4 PCA geometry and population separation

![Feature separation summary](/paper/figures/main/fig20_feature_separation.jpg)

**Figure 8.** Feature separation summary (condensed prefixes).

DFA state dominates (\(\eta^2 \approx 0.95\)).

### 3.5 Single-unit selectivity

Per-unit selectivity uses a peak-vs-rest index on category-mean activations (flat units gated to 0). Population medians of per-unit \(\eta^2\) agree with the separation analysis (DFA 0.95, prefix 0.87, character 0.60, position 0.42). Individual units span that spectrum: some are sharply tuned to character or position, others track DFA state more diffusely.

![Unit selectivity overview](/paper/figures/main/fig_unit_selectivity.jpg)

**Figure 9.** Unit selectivity overview (population summary).

![Example selective units](/paper/figures/main/fig_example_units.jpg)

**Figure 10.** Top-2 units per feature (DFA state, character, position from beginning, position from end) on one shared corpus window. Left: activation vs input characters (color = feature category); right: mean activation by category.

### 3.6 Decoding

![Linear decoding across seeds](/paper/figures/main/fig_decoding_seed_mean.jpg)

**Figure 11.** Linear decoding, mean ± std across seeds 1, 2, 3, 5, 7, 8.

Position and DFA saturate within a few PCs; character needs more dimensions.

### 3.7 Word trajectories

![Closed-loop vs internal dynamics](/paper/figures/main/fig_word_trajectories.jpg)

**Figure 12.** Closed-loop (self-fed, color = letter position) vs internal dynamics (seed then no input, color = timestep, with vector field).

Left: autoregressive generation with prefix labels, segments colored by in-word letter position. Right: letter seed then recurrent dynamics with no further input, colored by timestep; background vector field from the no-input map.

![Closed-loop trajectories across seeds](/paper/figures/main/fig_word_trajectories_by_start.jpg)

**Figure 13.** Closed-loop trajectories across 12 training seeds (same 16-word condition).

### 3.8 Weight structure

![Weight matrices by DFA size](/paper/figures/main/fig_weight_matrices_by_dfa.jpg)

**Figure 14.** Clustered weight matrices from the \(H{=}100\) sweep (seed 1). Left column: one random init for \(W_{xh}\) (top) and \(W_{hh}\) (bottom). Remaining columns: after learning, at increasing minimized \# DFA states (titles also note \#words and length). Each panel is clustered and color-scaled on its own (± range in the panel title).

![Weight metrics](/paper/figures/main/fig_weight_metrics_all_seeds.jpg)

**Figure 15.** Weight metrics on the main 16-word condition (top; mean ± std, \(n=16\)) and pooled init/final distributions (bottom).

On the main condition, final \(W_{xh}\) becomes **letter-columnar**: after clustering, units form coherent vertical stripes (shared signed input profiles). Across all seeds, within-block cohesion rises from \(0.02 \pm 0.02\) to \(0.16 \pm 0.09\). The pooled within-block pairwise-correlation histogram shifts right accordingly. Input/recurrent Frobenius ratio rises from \(0.41 \pm 0.09\) to \(1.53 \pm 0.40\); mean input-drive fraction from \(0.49 \pm 0.01\) to \(0.64 \pm 0.09\), with the per-unit drive-fraction histogram moving toward input dominance.

Final \(W_{hh}\) becomes **locally clumped** along the cluster order: adjacent-unit \(|\mathrm{corr}|\) doubles from \(0.13 \pm 0.03\) to \(0.28 \pm 0.04\) (see sample histogram). Mean within/between \(|W_{hh}|\) stays near 1: both within- and between-block magnitude histograms inflate similarly after learning, so the structure is local neighborhood coupling rather than a clean block-diagonal partition. Across the sweep (Figure 14), easy (few-state) automata show the strongest local \(W_{hh}\) blocks and clearer \(W_{xh}\) stripes; larger DFAs yield denser, more feedforward-looking weight maps.

### 3.9 Comparisons across length and vocabulary size

We expand the grid to vocabulary sizes \(5\)–\(25\) (step \(5\)) crossed with word lengths \(1\)–\(6\) plus mixed, using a shared hidden size \(H = 100\) (mean over seeds \(1\)–\(5\) for spectra; metric scatters use seeds \(1\)–\(15\)). Rather than reading length and word count as separate categorical factors, we score each condition by its minimized vocabulary DFA size and ask how geometry, training cost, and weights track that single complexity axis.

![Sweep PC variance spectra](/paper/figures/compare/fig_sweep_pc_spectra.jpg)

**Figure 16.** Cumulative closed-loop PC variance explained across the \(H{=}100\) word-count × length sweep. Left: color = letter length (green gradient; mixed in black), marker = vocabulary size. Right: same curves colored by minimized \# DFA states (viridis).

Smaller lexicons concentrate variance in the first one or two PCs: several few-word cells reach \(\approx 100\%\) by PC 2. Increasing either word count or letter length stretches the spectrum—more PCs are needed before the cumulative curve saturates. The DFA-colored panel shows the same continuum without separating the two knobs: larger automata spread closed-loop variance across more principal directions.

![Sweep metrics vs DFA size](/paper/figures/compare/fig_sweep_metrics_scatter2d.jpg)

**Figure 17.** Core metrics vs minimized vocabulary DFA state count; one point per seed × condition (seeds 1–15). Color = \# words; marker = word length. Black curve = best of linear / sigmoid / exponential-asymptote / hyperbola (by adjusted \(R^2\)).

Figure 17 collapses the two-dimensional sweep onto DFA size. Effective dimension rises with automaton size (\(R^2 \approx 0.8\)–\(0.85\)); top-2 variance falls (\(R^2 \approx 0.6\)–\(0.75\)). Iterations to 3% word error rise steeply (\(R^2 \approx 0.96\)). Weight readouts move with the same continuum—larger DFAs yield higher input/recurrent Frobenius ratios and lower adjacent recurrent correlation / top-1 \(W_{xh}\) mass. Color (#words) tracks the main constructor of DFA complexity; marker (length) shows residual spread at a given DFA size.

---

## 4. Discussion

Next-character prediction on an unsegmented finite lexicon yields DFA-aligned hidden geometry. The five-word demo makes the task transparent. On the 16-word condition, population separation and multi-seed decoding show that automaton state is low-dimensional and stable. Trajectories form labeled geometric motifs that recur across training seeds. Weight analyses on that same condition show letter-columnar input weights and locally clumped recurrent connectivity. The \(H{=}100\) word-count × length sweep (word counts \(5\)–\(25\) step \(5\)) makes the scaling claim concrete: hidden dimensionality, training iterations, and input vs recurrent balance track minimized DFA size, which is driven primarily by vocabulary size and secondarily by word length / mixing—so the network’s geometry expands when the word automaton expands, not merely when either experimental dial is turned in isolation.

**Limits.** Toy character languages; \(H = 50\) for the main 16-word analyses (\(H{=}100\) in the length × vocabulary sweep); small seed counts for grids; no acoustic noise. The model is a hypothesis generator, not a claim that infants are Elman networks.

**Supplementary (omitted from main text).** Prefix-labeled PCA overview; within-/between-DFA distance panels; next-character decision-region / per-character readout heatmaps; activations grouped by input character; DFA-grouped correlation heatmaps; per-seed decoding panels with trajectory insets; length × word-count metric heatmaps and 3D scatters (redundant with the DFA-state projections in Figures 16–17).

---

## 5. Conclusion

Small next-character RNNs discover word structure in unsegmented streams. States cluster by prefix and DFA identity; decoding recovers that structure across seeds; weight matrices develop letter-columnar \(W_{xh}\) and locally clumped \(W_{hh}\); across length and vocabulary size, larger minimized DFAs (built mainly by more words, secondarily by longer/mixed words) yield higher-dimensional closed-loop and corpus geometry, slower word-error acquisition, and more input-heavy weight balance.

---

## References

Aslin, R. N., Saffran, J. R., & Newport, E. L. (1998). *Psychological Science, 9*(4), 321–324.

Elman, J. L. (1990). Finding structure in time. *Cognitive Science, 14*(2), 179–211.

Frank, M. C., Goldwater, S., Griffiths, T. L., & Tenenbaum, J. B. (2010). *Cognition, 117*(2), 107–125.

French, R. M., Addyman, C., & Mareschal, D. (2011). TRACX. *Psychological Review, 118*(4), 614–636.

Goldwater, S., Griffiths, T. L., & Johnson, M. (2009). *Cognition, 112*(1), 21–54.

Perruchet, P., & Vinter, A. (1998). PARSER. *Journal of Memory and Language, 39*(2), 246–263.

Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). *Science, 274*(5294), 1926–1928.
