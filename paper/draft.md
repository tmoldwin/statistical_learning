# Statistical Word Segmentation as Emergent Structure in a Next-Character RNN

**Working title** · Hidden size \(h = 50\) throughout

---

## Abstract

Eight-month-old infants can segment continuous speech by tracking transitional probabilities between syllables (Saffran, Aslin, & Newport, 1996). We ask whether a vanilla Elman RNN trained only on next-character prediction develops internal representations aligned with word structure. After learning, the network generates legal vocabulary items, and its hidden states become a continuous embedding of the vocabulary’s minimal DFA.

A four-word demo (*cat*, *met*, *ate*, *tea*) introduces the trie, DFA, learning, and generated text. On a 16-word, 4-letter condition, DFA state explains \(\eta^2 \approx 0.95\) of condensed hidden variance and is linearly decodable from a few principal components (mean ± std across six seeds). Word trajectories form labeled geometric motifs. Cross-condition closed-loop grids then vary word length at fixed vocabulary size, and vocabulary size at fixed word length. Weight structure (clustered init vs final; motif metrics) is reported last.

---

## 1. Introduction

Fluent speech arrives without reliable pauses. Infants can use transitional probabilities to find word-like units (Saffran et al., 1996; Aslin, Saffran, & Newport, 1998). Computational accounts range from Bayesian segmentation and chunking (Goldwater, Griffiths, & Johnson, 2009; Perruchet & Vinter, 1998; French, Addyman, & Mareschal, 2011) to predictive sequence models (Elman, 1990).

For a finite vocabulary streamed without separators, optimal next-character prediction depends on the state of the vocabulary’s minimal DFA—the equivalence class of in-word prefixes with identical futures. An Elman RNN has no word units and no boundary channel, yet if it solves the prediction task its hidden state \(\mathbf{h}_t\) must carry that information. We test whether the information is geometrically organized.

**Plan.** (1) Four-word demo. (2) Single 16-word condition: next-character probabilities, activations, PCA + separation, decoding, then closed-loop trajectories. (3) Comparisons across length and vocabulary size (trajectory grids; learning; separation). (4) Weight structure.

---

## 2. Methods

**Demo lexicon** (`four_word_overlap_ns`): cat, met, ate, tea (vowels *a*/*e*; overlapping structure, not a single shared suffix).

**Main condition** (`sixteen_word_four_letter_ns`): bake, cake, lake, rake, bank, tank, rank, sank, late, mate, rate, gate, cant, pant, rant, want.

**Comparisons.** Length at 16 words: 3 / 4 / 5 letter. Vocabulary size at 4-letter words: 8 / 16. Seeds \(\{1,2,3\}\) for grids; decoding aggregates seeds \(\{1,2,3,5,7,8\}\).

**Model.** Elman RNN, \(H = 50\), next-character cross-entropy, early stop on word-error \(\leq 3\%\).

**Analyses.** Softmax next-character probabilities; activation heatmaps; hierarchical clustering of timesteps; PCA embeddings; feature separation (\(\eta^2\), silhouette, …); linear decoding from top-\(k\) PCs or random neurons (chance-corrected; mean ± std across seeds); closed-loop word trajectories; clustered init-vs-final weights and motif scalars.

Trie and DFA figures are shown **only** for the four-word demo.

![Figure 1. Prefix trie for *cat*, *met*, *ate*, *tea*.](figures/demo/fig01_trie.svg)

![Figure 2. Minimal DFA for the same four-word lexicon.](figures/demo/fig02_dfa.svg)

---

## 3. Results

### 3.1 Four-word demo

![Figure 3. Learning curve (truncated near the validity plateau).](figures/demo/fig03_learning_curve.jpg)

![Figure 4a. Vocabulary and unsegmented training stream.](figures/demo/fig04_corpus_stream.svg)

![Figure 4b. Generation before vs after learning (green = in-vocabulary, red = out-of-vocabulary).](figures/demo/fig04_samples.svg)

Unless noted, the remainder uses the **16-word, 4-letter** condition.

### 3.2 Next-character probabilities

![Figure 5. Softmax \(P(\text{next char} \mid \text{input so far})\) over time.](figures/main/fig_next_char_probs.jpg)

Probability mass concentrates late in words and spreads at ambiguous prefixes.

### 3.3 Hidden states and clustering

![Figure 6. Activations over timesteps (x = input letters; units clustered).](figures/main/fig_activation_heatmap.jpg)

![Figure 7. Activations clustered by in-word prefix.](figures/main/fig_activation_clustered.jpg)

### 3.4 PCA geometry and population separation

![Figure 8. PCA of \(\mathbf{h}\): large prefix-labeled overview (top); feature-colored points without labels (bottom).](figures/main/fig11_embedding_panels.jpg)

Prefix-labeled points form coherent regions in the plane. Coloring by character, within-word position, and DFA state shows the same geometry organized by different features.

![Figure 9. Feature separation summary (condensed prefixes).](figures/main/fig20_feature_separation.jpg)

![Figure 10. Within- vs between-DFA-state distances.](figures/main/fig19_dfa_distance.jpg)

DFA state dominates (\(\eta^2 \approx 0.95\)). Unit selectivity agrees (population median \(\eta^2\): DFA 0.97, prefix 0.84, character 0.67, position 0.40).

![Figure 11. Unit selectivity overview.](figures/main/fig_unit_selectivity.jpg)

### 3.5 Decoding

![Figure 12. Linear decoding, mean ± std across seeds 1, 2, 3, 5, 7, 8.](figures/main/fig_decoding_seed_mean.jpg)

Position and DFA saturate within a few PCs; character needs more dimensions. This closes the single-condition state analysis.

### 3.6 Word trajectories

![Figure 13. Closed-loop generation in PCA with prefix annotations.](figures/main/fig_word_trajectories.jpg)

Shared onsets share early path segments; completing a word returns toward a boundary-like region.

### 3.7 Comparisons across length and vocabulary size

![Figure 14. Closed-loop trajectories: rows = word length, columns = seed (16 words fixed).](figures/compare/fig_traj_by_length.jpg)

![Figure 15. Closed-loop trajectories: rows = vocabulary size, columns = seed (4-letter words fixed).](figures/compare/fig_traj_by_wordcount.jpg)

![Figure 16. Learning curves across 3 / 4 / 5 / mixed length (16 words).](figures/compare/fig_compare_learning.jpg)

![Figure 17. Feature separation across those length conditions.](figures/compare/fig_compare_separation.jpg)

| Condition | DFA \(\eta^2\) | Char \(\eta^2\) | Position \(\eta^2\) |
|-----------|---------------:|----------------:|--------------------:|
| 3-letter | 0.91 | 0.92 | 0.70 |
| 4-letter | 0.95 | 0.70 | 0.69 |
| 5-letter | 0.98 | 0.77 | 0.47 |
| Mixed 3–5 | 0.86 | 0.89 | 0.61 |

Fixed longer words preserve the strongest DFA geometry; mixed length elevates character \(\eta^2\).

*(PC variance spectra across word count / length are reserved for the comparison section in a later pass.)*

### 3.8 Weight structure

![Figure 18. Clustered init vs final \(W_{xh}\) and \(W_{hh}\) (2×2, no deltas).](figures/main/fig05_weights_init_final.jpg)

![Figure 19. Init vs final feedforward-balance metrics.](figures/main/fig_weight_structure_metrics.jpg)

![Figure 20. Per-unit input-drive fraction.](figures/main/fig_weight_input_drive.jpg)

![Figure 21. Motif scalars on clustered final weights.](figures/main/fig_weight_motif_summary.jpg)

Input/recurrent Frobenius ratio rises from 0.53 to 2.00; mean input-drive fraction from 0.50 to 0.71; block coupling \(\approx 0.68\).

---

## 4. Discussion

Next-character prediction on an unsegmented finite lexicon yields DFA-aligned hidden geometry. The four-word demo makes the task transparent. On the 16-word condition, population separation and multi-seed decoding show that automaton state is low-dimensional and stable. Trajectories make word structure visible as repeated paths; length and vocabulary-size grids show how that geometry changes with the lexicon. Weight analyses show learning that strengthens letter-specific input columns while building recurrent coupling.

**Limits.** Toy character languages; \(H = 50\); small seed counts for grids; no acoustic noise. The model is a hypothesis generator, not a claim that infants are Elman networks.

**Supplementary (omitted from main text).** DFA-beside-PCA composite; next-character decision-region / per-character readout heatmaps; activations grouped by input character; DFA-grouped correlation heatmaps; per-seed decoding panels with trajectory insets.

---

## 5. Conclusion

Small next-character RNNs discover word structure in unsegmented streams. States cluster by prefix and DFA identity; decoding recovers that structure across seeds; trajectories and cross-condition grids show how the solution scales with length and vocabulary size.

---

## References

Aslin, R. N., Saffran, J. R., & Newport, E. L. (1998). *Psychological Science, 9*(4), 321–324.

Elman, J. L. (1990). Finding structure in time. *Cognitive Science, 14*(2), 179–211.

Frank, M. C., Goldwater, S., Griffiths, T. L., & Tenenbaum, J. B. (2010). *Cognition, 117*(2), 107–125.

French, R. M., Addyman, C., & Mareschal, D. (2011). TRACX. *Psychological Review, 118*(4), 614–636.

Goldwater, S., Griffiths, T. L., & Johnson, M. (2009). *Cognition, 112*(1), 21–54.

Perruchet, P., & Vinter, A. (1998). PARSER. *Journal of Memory and Language, 39*(2), 246–263.

Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). *Science, 274*(5294), 1926–1928.
