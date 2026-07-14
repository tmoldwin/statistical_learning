# Statistical Word Segmentation as Emergent Structure in a Next-Character RNN

**Working title** · Hidden size \(h = 50\) throughout

---

## Abstract

Eight-month-old infants can segment continuous speech by tracking transitional probabilities between syllables (Saffran, Aslin, & Newport, 1996). We ask whether a vanilla Elman RNN trained only on next-character prediction develops internal representations aligned with word structure. After learning, the network generates legal vocabulary items, and its hidden states become a continuous embedding of the vocabulary’s minimal DFA.

A five-word demo (*cat*, *met*, *ate*, *tea*, *eat*) introduces learning and generated text. On a 16-word, 4-letter condition, DFA state explains \(\eta^2 \approx 0.95\) of condensed hidden variance and is linearly decodable from a few principal components (mean ± std across six seeds). Word trajectories form labeled geometric motifs. Weight structure on that same condition shows letter-columnar \(W_{xh}\) and locally clumped \(W_{hh}\). Fifty mixed-length English vocab runs (\(H{=}100\); random draws of 1–25 words from length-3/4/5/6 banks) then show that closed-loop dimensionality, training cost, and linear readouts track minimized DFA size—without holding word length or count fixed.

---

## 1. Introduction

Fluent speech arrives without reliable pauses. Infants can use transitional probabilities to find word-like units (Saffran et al., 1996; Aslin, Saffran, & Newport, 1998). Computational accounts range from Bayesian segmentation and chunking (Goldwater, Griffiths, & Johnson, 2009; Perruchet & Vinter, 1998; French, Addyman, & Mareschal, 2011) to predictive sequence models (Elman, 1990).

For a finite vocabulary streamed without separators, optimal next-character prediction depends on the state of the vocabulary’s minimal DFA—the equivalence class of in-word prefixes with identical futures. An Elman RNN has no word units and no boundary channel, yet if it solves the prediction task its hidden state \(\mathbf{h}_t\) must carry that information. We test whether the information is geometrically organized.

**Plan.** (1) Five-word demo. (2) Single 16-word condition: next-character probabilities, activations, separation, single-unit selectivity, decoding, trajectories, then weight structure. (3) Comparisons across length and vocabulary size.

---

## 2. Methods

**Demo lexicon** (`five_word_overlap_ns`): cat, met, ate, tea, eat (vowels *a*/*e*; overlapping structure, not a single shared suffix).

**Main condition** (`sixteen_word_four_letter_ns`): bake, cake, lake, rake, bank, tank, rank, sank, late, mate, rate, gate, cant, pant, rant, want.

**Comparisons.** Fifty mixed-English vocab runs at \(H{=}100\) (seed 1): sample \(n \in \{1,\ldots,25\}\) words from length-balanced banks (20 × lengths 3–6). Analyses score each run by minimized DFA size. Smaller seed sets \(\{1,2,3\}\) for trajectory grids; decoding aggregates seeds \(\{1,2,3,5,7,8\}\).

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

**Figure 14.** Clustered weight matrices from the mixed-vocab runs (\(H{=}100\), seed 1). Left column: one random init for \(W_{xh}\) / \(W_{hh}\). Remaining columns: after learning at the smallest successive minimized DFA sizes (titles also note \#words). Each matrix is color-scaled independently (± inset). Bottom row: density histograms of signed \(W_{xh}\) and \(W_{hh}\) with one shared \(x\)/\(y\) scale across columns.

On the main 16-word condition, final \(W_{xh}\) becomes **letter-columnar**: after clustering, units form coherent vertical stripes (shared signed input profiles). Across seeds, within-block cohesion rises and input drive grows relative to recurrent weights. Final \(W_{hh}\) becomes **locally clumped** along the cluster order (adjacent-unit correlation rises) without a clean block-diagonal partition. Across the mixed-vocab sweep (Figure 14), easy (few-state) automata show the strongest local \(W_{hh}\) blocks and clearer \(W_{xh}\) stripes; larger DFAs yield denser, more feedforward-looking weight maps.

### 3.9 Mixed-vocabulary runs scored by DFA size

Instead of a fixed length × word-count grid, we sample mixed English vocabs from length-balanced banks (20 words each of lengths 3–6). Each of 50 runs draws \(n \in \{1,\ldots,25\}\) words at random (\(H{=}100\), seed 1). Final analyses ignore \(n\) and score conditions by minimized vocabulary DFA size (range 4–49).

![Mixed-vocab scaling overview](/paper/figures/compare/fig_mixed_scaling_overview.jpg)

**Figure 15.** Mixed-vocab scaling with minimized DFA size. Top-left: sampled vocabulary size vs DFA state count. Top-right: closed-loop PC spectra colored by DFA size. Bottom: key metrics vs DFA (color = \# words; black = best trend by adjusted \(R^2\)): loop top-2 variance fraction, loop effective dimensionality, and iterations to 3\% word error.

![Readout curves PCA vs neurons](/paper/figures/compare/fig_mixed_decoding_curves.jpg)

**Figure 16.** Chance-corrected readouts. Top two rows: curves binned by DFA size from top-\(k\) PCA and from random subsets of \(k\) neurons. Bottom three rows: accuracy vs DFA size for each feature using the top 1 PC, top 5 PCs, or the full hidden state (color = \# words).

![Metrics vs DFA size](/paper/figures/compare/fig_mixed_metrics_vs_dfa.jpg)

**Figure 17.** Complementary corpus / weight metrics vs minimized DFA size (one point per run; color = \# words; Figure 15 overview metrics and full-hidden decode panels omitted). Black curve = best of linear / sigmoid / exponential-asymptote / hyperbola by adjusted \(R^2\).

---

## 4. Discussion

Next-character prediction on an unsegmented finite lexicon yields DFA-aligned hidden geometry. The five-word demo makes the task transparent. On the 16-word condition, population separation and multi-seed decoding show that automaton state is low-dimensional and stable. Trajectories form labeled geometric motifs that recur across training seeds. Weight analyses on that same condition show letter-columnar input weights and locally clumped recurrent connectivity. Fifty mixed-length English vocab runs (\(H{=}100\)) make the scaling claim concrete without fixing length or word count: hidden dimensionality, training iterations, and readout of position-from-end track minimized DFA size—so the network’s geometry expands when the word automaton expands.

**Limits.** Toy character languages; \(H = 50\) for the main 16-word analyses (\(H{=}100\) in the mixed-vocab runs); small seed counts for grids; no acoustic noise. The model is a hypothesis generator, not a claim that infants are Elman networks.

**Supplementary (omitted from main text).** Prefix-labeled PCA overview; within-/between-DFA distance panels; next-character decision-region / per-character readout heatmaps; activations grouped by input character; DFA-grouped correlation heatmaps; per-seed decoding panels with trajectory insets; older length × word-count metric heatmaps.

---

## 5. Conclusion

Small next-character RNNs discover word structure in unsegmented streams. States cluster by prefix and DFA identity; decoding recovers that structure across seeds; weight matrices develop letter-columnar \(W_{xh}\) and locally clumped \(W_{hh}\); across mixed-length English vocabs, larger minimized DFAs yield higher-dimensional closed-loop geometry, slower word-error acquisition, and weaker position-from-end readout.

---

## References

Aslin, R. N., Saffran, J. R., & Newport, E. L. (1998). *Psychological Science, 9*(4), 321–324.

Elman, J. L. (1990). Finding structure in time. *Cognitive Science, 14*(2), 179–211.

Frank, M. C., Goldwater, S., Griffiths, T. L., & Tenenbaum, J. B. (2010). *Cognition, 117*(2), 107–125.

French, R. M., Addyman, C., & Mareschal, D. (2011). TRACX. *Psychological Review, 118*(4), 614–636.

Goldwater, S., Griffiths, T. L., & Johnson, M. (2009). *Cognition, 112*(1), 21–54.

Perruchet, P., & Vinter, A. (1998). PARSER. *Journal of Memory and Language, 39*(2), 246–263.

Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). *Science, 274*(5294), 1926–1928.
