"""README figure manifest: order, slugs, and numbered plot filenames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadmeFigure:
    number: int
    slug: str
    plot_basename: str
    caption: str
    lead: str = ""

    def filename(self) -> str:
        return f"{self.number}_{self.slug}.png"


README_FIGURES: list[ReadmeFigure] = [
    ReadmeFigure(
        1, "vocabulary_trie", "vocabulary_trie.png",
        "Trie over the ten-word vocabulary. Nodes are prefixes; double circles are complete words. "
        "Edges are labeled by the consumed character.",
        "Before training, we compile the word list into a **trie**: a rooted tree whose edges are "
        "characters and whose paths spell valid prefixes. Every training word appears as a root-to-terminal "
        "path. Overlap is explicit: `cat`, `hat`, `mat`, and `rat` share the suffix `at`; `met`, `pet`, and "
        "`net` share `et`. The trie is the literal lexical hypothesis tree the model must implicitly navigate "
        "when predicting the next character.",
    ),
    ReadmeFigure(
        2, "vocabulary_min_dfa", "vocabulary_min_dfa.png",
        "Minimized deterministic finite automaton (DFA) for the same vocabulary. Each state lists the "
        "vocabulary words still consistent with the prefix read since the last space.",
        "The trie is folded into a **minimal DFA** by merging states with identical future continuations. "
        "This is the canonical reference machine for our second organizing axis: at each timestep we walk "
        "the DFA on the **in-word prefix** (characters since the last space) and record the current state "
        "$q_k$. Two timesteps with the same prefix always share a DFA state; two timesteps with the same "
        "input character but different prefixes generally do not.",
    ),
    ReadmeFigure(
        3, "learning_curve", "learning_curve.png",
        "Training loss (blue, 51-iteration rolling median of per-window cross-entropy) and stochastic "
        "word-error rate (orange, right axis: percent of space-delimited tokens not in the vocabulary "
        "during long sampled rollouts).",
        "We first verify that optimization succeeds. Cross-entropy falls steadily over 15,000 iterations. "
        "In parallel we track **word error rate**: sample long strings from the model and count how often "
        "whitespace-delimited chunks are not exact vocabulary words. This metric is our analogue of "
        "\"does the generator respect the statistical word units?\" After training, invalid-word rate "
        "approaches zero: the model is not merely memorizing local trigrams; it has learned to emit legal words.",
    ),
    ReadmeFigure(
        4, "samples_before_after", "samples_before_after.png",
        "Three 50-character rows: excerpt from the training corpus (top), stochastic sample at "
        "initialization (middle), stochastic sample after training (bottom). Green/red per-character "
        "coloring marks in-vocabulary vs out-of-vocabulary segments in the generated rows.",
        "Figure 4 compares **what the model actually generates** before and after learning. The top row "
        "is ground-truth stream structure. Before training, characters are essentially unstructured noise "
        "with respect to the lexicon. After training, almost every character lies inside a valid vocabulary "
        "word. The display uses fixed 50-character windows so before/after comparisons are apples-to-apples.",
    ),
    ReadmeFigure(
        5, "weights", "weights.png",
        "Learned weight matrices after training. Left: input weights $W_{xh}$ (character columns $\\times$ "
        "hidden rows). Right: recurrent weights $W_{hh}$ (hidden $\\times$ hidden).",
        "The raw parameters reveal which letters drive which hidden units and how units recurrently mix. "
        "Input columns show letter-specific tuning; recurrent blocks show long-timescale coupling. With "
        "$h=32$ the matrices are small enough to inspect directly. There is no hand-designed word feature: "
        "any boundary or lexical structure must be implemented through these weights.",
    ),
    ReadmeFigure(
        6, "weights_eigenspectra", "weights_eigenspectra.png",
        "Eigenvalue spectra of the recurrent and cross-weight blocks, summarizing effective memory "
        "timescales and stability of the trained dynamics.",
        "The spectrum of $W_{hh}$ (and related blocks) indicates how many past characters the recurrence "
        "can integrate and whether dynamics are contractive or expansive in different directions. For word "
        "learning, we expect nontrivial structure here: the network must preserve prefix information across "
        "several timesteps without a dedicated counter.",
    ),
    ReadmeFigure(
        7, "activation_heatmap", "activation_heatmap.png",
        "Hidden activations over a 50-character analysis window. Rows are hidden units $h_0\\ldots h_{31}$; "
        "columns are timesteps (input characters shown along the bottom).",
        "This is the raw activation trace from which all geometry plots are derived. Each column is "
        "$\\mathbf{h}_t$ after reading one more character. Visual inspection already suggests structure: "
        "activations repeat with similar patterns when the model is at analogous positions inside words, "
        "even when the absolute corpus index differs.",
    ),
    ReadmeFigure(
        8, "next_char_prob_sequence", "next_char_prob_sequence_heatmap.png",
        "Softmax next-character probabilities at every timestep (columns), with the true next character "
        "highlighted. Brighter cells are higher predicted probability.",
        "Behaviorally, the model is a next-character predictor. Where the trie branches are narrow (late in "
        "a word, or after an informative prefix), probability mass concentrates on one or few characters. "
        "At ambiguous early prefixes (`c` could start `cat`; `a` is shared widely), mass spreads. Comparing "
        "to ground truth shows where the trained network is confident vs uncertain.",
    ),
    ReadmeFigure(
        9, "activation_by_input_char", "activation_by_input_char.png",
        "For each input character, all timesteps where that character was read. Columns are labeled by "
        "**in-word prefix** (e.g. `h`, `ha`, `hat` after a space). Rows are hidden units; columns are "
        "occurrences, optionally clustered by activation similarity.",
        "This panel is the first direct evidence for **prefix-axis organization**. Fix an input letter "
        "such as `a`. Every occurrence is shown, but columns are sorted/labeled by how far into the "
        "current word that `a` appeared. Timesteps with the same prefix produce similar activation "
        "profiles even when they occur at unrelated positions in the corpus. The network encodes "
        "\"where am I inside this word?\" not merely \"what letter did I just see?\"",
    ),
    ReadmeFigure(
        10, "activation_clustered_heatmap", "activation_clustered_heatmap.png",
        "Hierarchically clustered heatmap of all timesteps $\\times$ hidden units in the analysis window. "
        "Row and column dendrograms group similar timesteps; tick labels are in-word prefixes.",
        "Clustering across the full 50-timestep window reveals blocks of timesteps with near-identical "
        "hidden vectors. Many blocks align with shared prefixes or suffixes (`at`, `et`, etc.). This is "
        "unsupervised structure in $\\mathbf{h}_t$ using only prefix labels for interpretation - the "
        "clustering itself is driven purely by activation similarity.",
    ),
    ReadmeFigure(
        11, "embedding_panels_context", "embedding_panels_context.png",
        "Four 2D embeddings of the same 50 hidden states: PCA, UMAP, t-SNE, and Isomap. Points are "
        "annotated with in-word prefix labels; colors follow embedding-specific layout.",
        "Because $h=32$, we project $\\mathbf{h}_t$ to the plane for visualization. Different nonlinear "
        "methods stress different aspects (global variance, local neighborhoods, geodesics), but all show "
        "annotated prefixes grouping into coherent regions. PCA is used consistently in subsequent panels "
        "so trajectories and vector fields live in a single linear subspace.",
    ),
    ReadmeFigure(
        12, "dfa_and_embedding_pca", "dfa_and_embedding_pca.png",
        "Left: minimized DFA from Figure 2. Right: PCA of hidden states with point color = DFA state and "
        "text label = in-word prefix. Leader lines connect grouped prefix annotations.",
        "This is the **central figure**. The DFA is the discrete lexical reference; the PCA scatter is "
        "the continuous representation the RNN actually uses. Points with the same DFA color cluster "
        "together even when prefix labels differ in length. Conversely, along a single word, the trajectory "
        "visits multiple DFA states as more letters disambiguate the lexical hypothesis. The geometry "
        "**implements the automaton**: the second organizing axis is not an artifact of coloring.",
    ),
    ReadmeFigure(
        13, "next_char_regions_pca", "next_char_regions_pca.png",
        "PCA plane with 2D-reconstructed hidden states. Left: argmax next-character label in each region. "
        "Right: prediction entropy (nats). Overlaid points carry prefix annotations.",
        "These panels answer: **what would the model predict at each location in hidden space?** Low-entropy "
        "regions are predictable continuations inside words; high-entropy regions sit at trie branch points "
        "where several next characters remain viable. The overlaid real trajectory samples these regions as "
        "it moves through prefixes.",
    ),
    ReadmeFigure(
        14, "next_char_prob_panels_pca", "next_char_prob_panels_pca.png",
        "One panel per vocabulary character, showing $P(\\text{next}=c\\mid\\mathbf{h})$ over the PCA plane "
        "(from softmax on 2D-reconstructed $\\mathbf{h}$).",
        "Decomposing the output layer per character shows how each letter's logit carves a different region "
        "of hidden space. Vowels and consonants that appear in overlapping words (`a`, `t`, `e`, ...) have "
        "complex, interleaved regions - reflecting the competition among `-at`, `-et`, and `-ea` families.",
    ),
    ReadmeFigure(
        15, "vector_field_grid_pca", "vector_field_grid_pca_no_input.png",
        "Recurrent vector field in PCA coordinates with **no external input**: "
        "$\\mathbf{h}_{t+1}=\\tanh(W_{hh}\\mathbf{h}_t)$, projected to PC1-PC2. Quiver arrows show local flow.",
        "Between explicit character inputs, the hidden state still evolves under $W_{hh}$ alone. The vector "
        "field shows attractor-like structure and drift directions in the PCA plane. This is the intrinsic "
        "dynamics the network would follow if characters stopped arriving - relevant for understanding "
        "transients at word boundaries and spaces.",
    ),
    ReadmeFigure(
        16, "word_trajectories_pca", "word_trajectories_pca.png",
        "PCA trajectories from one space timestep to the next (space-to-space segments) in the 50-character "
        "window. Each word path is colored by word identity; faint background shows optional no-input flow.",
        "This is the trajectory-level view of the **first organizing axis**. Each word is a path through "
        "hidden space starting just after a space. Paths with the same prefix length tend to occupy similar "
        "\"lanes\"; completing a word returns toward a boundary region. The figure makes word segmentation "
        "visible as repeated geometric motifs without ever supervising boundaries.",
    ),
    ReadmeFigure(
        17, "state_correlation_clustered", "state_correlation_clustered_heatmap.png",
        "Pearson correlation matrix between hidden vectors at all pairs of timesteps, with hierarchical "
        "clustering on rows/columns. Tick labels: in-word prefix; label color = minimized DFA state.",
        "Correlation complements PCA: it measures linear similarity of full 32-dimensional states. Blocks "
        "of high correlation appear when both prefix and DFA state align. Tick colors show that DFA state "
        "often cuts across prefix clusters - two timesteps can share a prefix length but differ in DFA "
        "state if the letters diverge (`ca` vs `ha`).",
    ),
    ReadmeFigure(
        18, "state_correlation_by_dfa_state", "state_correlation_by_dfa_state.png",
        "Same correlation matrix, but timesteps are **grouped by DFA state** (all states shown). Diagonal "
        "blocks are within-state correlations; off-diagonal blocks are between-state.",
        "Reordering by DFA state exposes block structure directly. High values on the diagonal mean hidden "
        "states in the same automaton state are similar; lower off-diagonal values mean states representing "
        "different lexical hypotheses are separated. This is the correlation analogue of Figure 12's coloring.",
    ),
    ReadmeFigure(
        19, "dfa_state_distance_comparison", "dfa_state_distance_comparison.png",
        "Pairwise Euclidean distances between hidden vectors (subsampled pairs). Three distributions: "
        "within the same DFA state, between different DFA states, and pairs with the same input character.",
        "The quantitative summary: within-state distances are sharply smaller than between-state distances, "
        "even though same-input-character pairs can be far apart. The DFA partition captures variance in "
        "$\\mathbf{h}_t$ that raw character identity alone cannot. Error bars / overlays show means; "
        "scatter shows individual pairs.",
    ),
]

PLOT_BASENAME_TO_FIGURE: dict[str, ReadmeFigure] = {
    fig.plot_basename: fig for fig in README_FIGURES
}


def numbered_plot_path(out_dir: str | Path, plot_basename: str) -> Path:
    """Path for a README figure inside an experiment plots directory."""
    fig = PLOT_BASENAME_TO_FIGURE.get(plot_basename)
    if fig is None:
        return Path(out_dir) / plot_basename
    return Path(out_dir) / fig.filename()


def remove_legacy_readme_plot_names(plots_dir: str | Path) -> None:
    """Drop unnumbered README plot files once numbered copies exist."""
    root = Path(plots_dir)
    for fig in README_FIGURES:
        legacy = root / fig.plot_basename
        numbered = root / fig.filename()
        if numbered.is_file() and legacy.is_file() and legacy != numbered:
            legacy.unlink()
