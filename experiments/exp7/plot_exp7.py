#!/usr/bin/env python3
"""Standalone matplotlib plots for EXP-7 (no seaborn/subplots)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSES = ["Foreground", "LV", "MYO", "RV"]


def _row(summary, radius, cls, domain="All"):
    return summary[(summary.radius == radius) & (summary['class'] == cls) & (summary.domain == domain)].iloc[0]


def make_all_figures(summary, pair, rank, boundary, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    row = [_row(summary, 2, c) for c in CLASSES]
    x = np.arange(len(CLASSES)); width = .34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width/2, [r.same_agreement for r in row], width, label="SamePosition")
    ax.bar(x + width/2, [r.local_agreement for r in row], width, label="LocalNN r=2")
    ax.set_xticks(x, CLASSES); ax.set_ylabel("Semantic agreement"); ax.set_ylim(0, 1); ax.set_title("Same-position versus LocalNN")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "fig_same_vs_local_agreement.png", dpi=300); plt.close(fig)

    classes = ["LV", "MYO", "RV"]
    vals = [_row(summary, 2, c).delta_local for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(classes, vals, color=["#59a14f", "#f28e2b", "#e15759"]); ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("LocalNN r=2 delta"); ax.set_title("Local correspondence gain by class"); fig.tight_layout()
    fig.savefig(out_dir / "fig_local_delta_by_class.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cls in CLASSES:
        vals = [_row(summary, r, cls).delta_local for r in [1, 2, 3]]
        ax.plot([1, 2, 3], vals, marker="o", label=cls)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xticks([1, 2, 3]); ax.set_xlabel("LocalNN radius")
    ax.set_ylabel("Local agreement delta"); ax.set_title("Radius sensitivity"); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "fig_radius_sensitivity.png", dpi=300); plt.close(fig)

    labels = ["SamePosition", "LocalNN r=2", "GlobalNN", "LocalLabelOracle"]
    vals = [row[0].same_agreement, row[0].local_agreement, row[0].global_nn_agreement, row[0].oracle_local_agreement]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(labels, vals, color=["#bab0ab", "#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylim(0, 1); ax.set_ylabel("Foreground semantic agreement"); ax.set_title("Correspondence ceilings"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(out_dir / "fig_local_vs_global_nn.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(pair.mean_displacement_r2.dropna(), bins=35, color="#4c78a8", alpha=.85)
    ax.set_xlabel("LocalNN r=2 displacement (feature-grid pixels)"); ax.set_ylabel("Pair count"); ax.set_title("Local displacement distribution")
    fig.tight_layout(); fig.savefig(out_dir / "fig_displacement_distribution.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(rank['rank'], rank.delta_local_fg, marker="o", label="Δ Local")
    ax.plot(rank['rank'], rank.mean_displacement_r2, marker="s", label="Displacement")
    ax.axhline(0, color="black", linewidth=.8); ax.set_xlabel("Memory rank"); ax.set_title("Rank versus local gain"); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "fig_rank_vs_local_gain.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.bar(["Boundary", "Interior"], [boundary.boundary_delta.iloc[0], boundary.interior_delta.iloc[0]], color=["#e45756", "#59a14f"])
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("LocalNN r=2 delta"); ax.set_title("Boundary versus interior"); fig.tight_layout()
    fig.savefig(out_dir / "fig_boundary_vs_interior_gain.png", dpi=300); plt.close(fig)

    valid = pair[['delta_similarity_r2_fg', 'local_r2_delta_fg']].dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.scatter(valid.delta_similarity_r2_fg, valid.local_r2_delta_fg, s=3, alpha=.2)
    ax.axhline(0, color="black", linewidth=.8); ax.axvline(0, color="black", linewidth=.8)
    ax.set_xlabel("Δ feature similarity"); ax.set_ylabel("Δ semantic agreement"); ax.set_title("Feature similarity gain versus semantic gain")
    fig.tight_layout(); fig.savefig(out_dir / "fig_similarity_gain_vs_semantic_gain.png", dpi=300); plt.close(fig)
