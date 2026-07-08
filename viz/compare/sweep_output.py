"""Sweep comparison artifact paths: JSON in data/, figures by kind."""

from __future__ import annotations

from pathlib import Path

from experiment import comparison_dir


def sweep_data_dir(comparison_name: str) -> Path:
    path = comparison_dir(comparison_name, "data")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sweep_figures_dir(comparison_name: str) -> Path:
    """Trajectory / geometry / spectrum figures."""
    path = comparison_dir(comparison_name, "trajectories")
    path.mkdir(parents=True, exist_ok=True)
    return path


def sweep_decoding_dir(comparison_name: str) -> Path:
    """Decoding curve figures (separate from trajectories)."""
    path = comparison_dir(comparison_name, "decoding")
    path.mkdir(parents=True, exist_ok=True)
    return path
