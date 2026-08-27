def _overview_learning_curves_from_checkpoint(
    task: str,
    *,
    model_type: str,
    seed: int,
) -> dict[str, np.ndarray] | None:
    """Return CE + word-error series for one mixed-dfa run checkpoint."""
    ckpt = checkpoint_path(task, model_type, seed=seed)
    if not ckpt.is_file():
        return None
    data = np.load(ckpt, allow_pickle=True)
    metric_iters = np.asarray(data["metric_iterations"], dtype=float).ravel()
    word_err = np.asarray(data["metric_word_error_frac"], dtype=float).ravel()
    if metric_iters.size < 2 or word_err.size != metric_iters.size:
        return None

    if "metric_val_ce" in data.files:
        ce_iters = metric_iters
        ce = np.asarray(data["metric_val_ce"], dtype=float).ravel()
        if ce.size != ce_iters.size:
            return None
    elif "loss_smooth" in data.files and "loss_iterations" in data.files:
        ce_iters = np.asarray(data["loss_iterations"], dtype=float).ravel()
        ce = np.asarray(data["loss_smooth"], dtype=float).ravel()
        seq_len = float(np.asarray(data["sequence_length"]).reshape(-1)[0]) if "sequence_length" in data.files else 0.0
        if seq_len > 0:
            ce = ce / seq_len
        if ce_iters.size < 2 or ce.size != ce_iters.size:
            return None
    else:
        return None

    return {
        "ce_iters": ce_iters,
        "ce": ce,
        "we_iters": metric_iters,
        "word_err": word_err,
    }


def plot_mixed_dfa_scaling_overview(
    payload: dict[str, Any] | None = None,
    *,
    outfile: str = "scaling_overview.png",
    recompute: bool = False,
    model_type: str | None = None,
    seed: int | None = None,
) -> Path:
    """Paper overview: training cost, PC spectra, CE + word-error curves."""
    from viz.compare.pow2_sweep_metric_board import _fit_trend

    decode_payload = payload or _load_panels()
    decode_panels = [
        p for p in decode_payload["panels"]
        if "error" not in p and p.get("spectrum_pct")
    ]
    mt = model_type or str(decode_payload.get("model_type") or "rnn_dale")
    metric_path = collect_mixed_dfa_metric_board(recompute=recompute, model_type=mt)
    metric_panels = [
        p for p in json.loads(metric_path.read_text(encoding="utf-8"))["panels"]
        if "error" not in p
    ]

    # Story: learning (CE â†’ word-error) then summaries (iters â†’ spectra).
    # Colorbar columns sit with the panels that introduce each scale.
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_ce = fig.add_subplot(gs[0, 0])
    gs_we = gs[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.12)
    ax_we = fig.add_subplot(gs_we[0, 0])
    cax_dfa = fig.add_subplot(gs_we[0, 1])
    gs_it = gs[1, 0].subgridspec(1, 2, width_ratios=[1.0, 0.055], wspace=0.12)
    ax_it = fig.add_subplot(gs_it[0, 0])
    cax_w = fig.add_subplot(gs_it[0, 1])
    ax_sp = fig.add_subplot(gs[1, 1])

    max_pcs = int(decode_payload.get("max_k", _DEFAULT_MAX_PCS))
    ks = np.arange(1, max_pcs + 1, dtype=float)
    dfa_vals = [float(p["n_dfa_states"]) for p in decode_panels]
    vmin = min(dfa_vals) if dfa_vals else 0.0
    vmax = max(dfa_vals) if dfa_vals else 1.0
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    for panel in decode_panels:
        y = np.asarray(panel["spectrum_pct"], dtype=float)
        n = min(len(y), max_pcs)
        ax_sp.plot(
            ks[:n], y[:n],
            color=cmap(norm(float(panel["n_dfa_states"]))),
            lw=1.0, alpha=0.75,
        )
    ax_sp.set_xlabel("PC index", fontsize=8)
    ax_sp.set_ylabel("% variance", fontsize=8)
    ax_sp.set_title("closed-loop PC spectra", fontsize=9, pad=4)
    ax_sp.set_xlim(1, max_pcs)
    ax_sp.grid(True, alpha=0.25)
    ax_sp.tick_params(labelsize=7)

    path_key, title, log_y = _OVERVIEW_METRICS[0]
    words_cmap = plt.get_cmap("YlOrRd")
    words_norm = plt.Normalize(vmin=1.0, vmax=25.0)
    mx: list[float] = []
    my: list[float] = []
    mn: list[float] = []
    for p in metric_panels:
        y = _dig(p, path_key)
        if y is None:
            continue
        mx.append(float(p["n_dfa_states"]))
        my.append(y)
        mn.append(float(p["n_words"]))
    words_cbar_spec: tuple[Any, Any] | None
    if len(mx) >= 3:
        x = np.asarray(mx, dtype=float)
        y = np.asarray(my, dtype=float)
        n_words = np.asarray(mn, dtype=float)
        use_log = bool(log_y and np.all(y > 0))
        y_plot = np.log10(np.clip(y, 1e-12, None)) if use_log else y
        ax_it.scatter(
            x, y_plot,
            c=n_words, cmap=words_cmap, norm=words_norm,
            s=18, alpha=0.75, linewidths=0.25, edgecolors="white", zorder=2,
        )
        x_fit, y_fit, r2, _model = _fit_trend(x, y_plot)
        panel_title = title
        if x_fit is not None and y_fit is not None and np.isfinite(r2):
            ax_it.plot(x_fit, y_fit, color="#111111", lw=1.15, zorder=3)
            panel_title = f"{title}\n$R^2$={r2:.2f}"
        ax_it.set_title(panel_title, fontsize=8, pad=4)
        ax_it.set_xlabel("DFA states", fontsize=8)
        if use_log:
            ax_it.set_ylabel("log10 iters", fontsize=8)
        ax_it.grid(True, alpha=0.25)
        ax_it.tick_params(labelsize=7)
        words_cbar_spec = (words_cmap, words_norm)
    else:
        ax_it.set_axis_off()
        words_cbar_spec = None

    # Learning curves: one curve per run, colored by DFA size (same as spectra).
    run_seed = int(seed if seed is not None else (decode_payload.get("seeds") or [1])[0])
    n_curves = 0
    for panel in sorted(decode_panels, key=lambda p: float(p["n_dfa_states"])):
        curves = _overview_learning_curves_from_checkpoint(
            str(panel["task"]),
            model_type=mt,
            seed=int(panel.get("seed", run_seed)),
        )
        if curves is None:
            continue
        color = cmap(norm(float(panel["n_dfa_states"])))
        ax_ce.plot(curves["ce_iters"], curves["ce"], color=color, lw=1.0, alpha=0.75)
        ax_we.plot(curves["we_iters"], curves["word_err"], color=color, lw=1.0, alpha=0.75)
        n_curves += 1
    ax_ce.set_xlabel("iteration", fontsize=8)
    ax_ce.set_ylabel("val CE / char", fontsize=8)
    ax_ce.set_title("cross-entropy learning", fontsize=9, pad=4)
    ax_ce.grid(True, alpha=0.25)
    ax_ce.tick_params(labelsize=7)
    ax_we.axhline(0.03, color="0.45", ls="--", lw=0.9, zorder=1)
    ax_we.set_xlabel("iteration", fontsize=8)
    ax_we.set_ylabel("word error frac", fontsize=8)
    ax_we.set_title("word-error learning", fontsize=9, pad=4)
    ax_we.set_ylim(0.0, 1.05)
    ax_we.grid(True, alpha=0.25)
    ax_we.tick_params(labelsize=7)
    if n_curves == 0:
        ax_ce.set_axis_off()
        ax_we.set_axis_off()

    finalize_grid_figure(
        fig,
        suptitle="Mixed-vocab scaling with DFA size",
        bottom=0.08,
        left=0.08,
        right=0.94,
        top=0.90,
        wspace=0.28,
        hspace=0.32,
    )

    if words_cbar_spec is not None:
        w_cmap, w_norm = words_cbar_spec
        cbar_w = fig.colorbar(
            plt.cm.ScalarMappable(cmap=w_cmap, norm=w_norm),
            cax=cax_w,
        )
        cbar_w.set_label("# words", fontsize=7)
        cbar_w.ax.tick_params(labelsize=6)
    else:
        cax_w.set_axis_off()

    if decode_panels:
        cbar_dfa = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            cax=cax_dfa,
        )
        cbar_dfa.set_label("DFA states", fontsize=7)
        cbar_dfa.ax.tick_params(labelsize=6)
    else:
        cax_dfa.set_axis_off()

    out = sweep_figures_dir(COMPARISON_NAME) / outfile
    save_figure(fig, out)
    plt.close(fig)
    return out

