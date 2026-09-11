#!/usr/bin/env python3
"""Offline analysis for EXP-8 SABE normalization diagnosis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.metrics import (average_precision_score, auc, roc_auc_score,
                             roc_curve)


DOMAINS = ["B", "C", "D"]
CLASSES = ["lv", "myo", "rv"]
EPS = 1e-8
BOOTSTRAPS = 10000
SEED = 2026


def finite_pair(frame, x, y):
    values = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    return values[x].to_numpy(float), values[y].to_numpy(float)


def corr_values(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan, len(x)
    return float(spearmanr(x, y).statistic), float(pearsonr(x, y).statistic), len(x)


def auc_metrics(signal, target):
    signal, target = np.asarray(signal, float), np.asarray(target, int)
    valid = np.isfinite(signal) & np.isfinite(target)
    signal, target = signal[valid], target[valid]
    prevalence = float(target.mean()) if len(target) else np.nan
    if len(np.unique(target)) < 2:
        return np.nan, np.nan, prevalence, len(target)
    return (float(roc_auc_score(target, signal)),
            float(average_precision_score(target, signal)), prevalence, len(target))


def bootstrap_ci(values, statistic, rng, n_boot=BOOTSTRAPS, chunk=100):
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    draws = []
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        indices = rng.integers(0, n, size=(size, n))
        for row in indices:
            draws.append(statistic(values[row]))
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def bootstrap_corr(x, y, rng, n_boot=BOOTSTRAPS, chunk=100):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan
    rx, ry = rankdata(x), rankdata(y)
    results = []
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        indices = rng.integers(0, n, size=(size, n))
        a, b = rx[indices], ry[indices]
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            results.extend(((a * b).sum(axis=1) / den).tolist())
    return float(np.nanpercentile(results, 2.5)), float(np.nanpercentile(results, 97.5))


def bootstrap_auc(signal, target, rng, n_boot=BOOTSTRAPS, chunk=100):
    signal, target = np.asarray(signal, float), np.asarray(target, int)
    n = len(signal)
    if n < 2 or len(np.unique(target)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    auc_values, ap_values = [], []
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        indices = rng.integers(0, n, size=(size, n))
        for row in indices:
            yy, ss = target[row], signal[row]
            if len(np.unique(yy)) < 2:
                continue
            auc_values.append(roc_auc_score(yy, ss))
            ap_values.append(average_precision_score(yy, ss))
    return (float(np.nanpercentile(auc_values, 2.5)),
            float(np.nanpercentile(auc_values, 97.5)),
            float(np.nanpercentile(ap_values, 2.5)),
            float(np.nanpercentile(ap_values, 97.5)))


def save_figure(path, title, xlabel, ylabel):
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_axes([0.13, 0.13, 0.82, 0.78])
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return fig, ax


def finish(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(data_dir: Path):
    full = pd.read_csv(data_dir / "full_slice_metrics_seed2026.csv")
    sff = pd.read_csv(data_dir / "sff_slice_metrics_seed2026.csv")
    key = ["domain", "query_name"]
    if len(full) != 4091 or len(sff) != 4091:
        raise RuntimeError(f"EXP-8 requires 4091 slices: full={len(full)} sff={len(sff)}")
    if not full[key].equals(sff[key]):
        raise RuntimeError("Full and SFF streams are not paired identically")
    utility = full.merge(sff, on=key, suffixes=("_full", "_sff"), validate="one_to_one")
    utility["global_index"] = full["global_index"]
    utility["is_sft"] = full["is_sft"].astype(bool)
    utility["volume"] = full["volume"]
    utility["dice_full"] = utility["dice_foreground_full"]
    utility["dice_sff_only"] = utility["dice_foreground_sff"]
    utility["g_sabe"] = utility["dice_full"] - utility["dice_sff_only"]
    for cls in CLASSES:
        utility[f"g_sabe_{cls}"] = utility[f"dice_{cls}_full"] - utility[f"dice_{cls}_sff"]
    utility["harm_any"] = utility["g_sabe"] < 0
    utility["harm_1pp"] = utility["g_sabe"] < -0.01
    utility["harm_5pp"] = utility["g_sabe"] < -0.05
    utility["benefit_any"] = utility["g_sabe"] > 0
    utility["benefit_1pp"] = utility["g_sabe"] > 0.01
    utility["benefit_5pp"] = utility["g_sabe"] > 0.05
    utility["neutral_1pp"] = utility["g_sabe"].abs() <= 0.01

    layer = pd.read_csv(data_dir / "bn_layer_query_statistics.csv")
    distance = pd.read_csv(data_dir / "bn_memory_distance_statistics.csv")
    weighted = pd.read_csv(data_dir / "weighted_statistics_diagnostic.csv")
    if len(layer) == 0 or len(distance) == 0:
        raise RuntimeError("EXP-8 has no SABE BN records")
    layer["category"] = np.where(layer.phase.eq("sabe_encoder"), "encoder", "decoder")
    distance["category"] = np.where(distance.phase.eq("sabe_encoder"), "encoder", "decoder")

    aggregate = layer.groupby(["global_index", "domain", "query_name"], as_index=False).agg(
        bn_shift_all=("bn_shift_equal", "mean"),
        bn_shift_max=("bn_shift_equal", "max"),
        bn_shift_median=("bn_shift_equal", "median"),
        similarity_shift_reduction_all=("relative_reduction_similarity", "mean"),
        proximity_shift_reduction_all=("relative_reduction_proximity", "mean"))
    for category in ("encoder", "decoder"):
        part = layer[layer.category == category].groupby("global_index").agg(
            **{f"bn_shift_{category}": ("bn_shift_equal", "mean"),
               f"similarity_shift_reduction_{category}": ("relative_reduction_similarity", "mean"),
               f"proximity_shift_reduction_{category}": ("relative_reduction_proximity", "mean")})
        aggregate = aggregate.merge(part, on="global_index", how="left")
    utility = utility.drop(columns=[c for c in ["bn_shift_encoder", "bn_shift_decoder", "bn_shift_all",
                                                  "similarity_shift_reduction_encoder", "similarity_shift_reduction_decoder",
                                                  "similarity_shift_reduction_all", "proximity_shift_reduction_encoder",
                                                  "proximity_shift_reduction_decoder", "proximity_shift_reduction_all"] if c in utility])
    utility = utility.merge(aggregate.drop(columns=["domain", "query_name"]), on="global_index", how="left")

    utility_columns = ["global_index", "domain", "query_name", "is_sft", "volume",
                       "dice_full", "dice_sff_only", "g_sabe", "g_sabe_lv", "g_sabe_myo", "g_sabe_rv",
                       "harm_any", "harm_1pp", "harm_5pp", "benefit_any", "benefit_1pp", "benefit_5pp",
                       "bn_shift_encoder", "bn_shift_decoder", "bn_shift_all",
                       "similarity_shift_reduction_encoder", "similarity_shift_reduction_decoder",
                       "similarity_shift_reduction_all", "proximity_shift_reduction_all"]
    utility[utility_columns].to_csv(data_dir / "sabe_utility_seed2026.csv", index=False)

    oracle = utility[["global_index", "domain", "query_name", "dice_full", "dice_sff_only"]].copy()
    oracle["dice_oracle"] = oracle[["dice_full", "dice_sff_only"]].max(axis=1)
    oracle["sabe_oracle_gain"] = oracle["dice_oracle"] - oracle["dice_full"]
    oracle["delta_oracle_sabe"] = oracle["dice_oracle"] - oracle["dice_full"]
    oracle.to_csv(data_dir / "sabe_oracle_headroom.csv", index=False)

    # Per-query Top-K heterogeneity and per-layer/phase similarity analysis.
    distance_query = distance.groupby(["global_index", "memory_rank"], as_index=False).agg(
        retrieval_similarity=("retrieval_similarity", "mean"), bn_distance=("bn_distance", "mean"))
    rank_rows = []
    for rank in range(1, 6):
        part = distance_query[distance_query.memory_rank == rank]
        rank_rows.append({"memory_rank": rank, "n": len(part),
                          "global_similarity": part.retrieval_similarity.mean(),
                          "bn_distance_encoder": distance[(distance.memory_rank == rank) & (distance.category == "encoder")].bn_distance.mean(),
                          "bn_distance_decoder": distance[(distance.memory_rank == rank) & (distance.category == "decoder")].bn_distance.mean(),
                          "bn_distance_all": part.bn_distance.mean()})
    rank_table = pd.DataFrame(rank_rows)
    similarity_rows = []
    for category in ("encoder", "decoder", "all"):
        part = distance if category == "all" else distance[distance.category == category]
        x, y = finite_pair(part, "retrieval_similarity", "bn_distance")
        rho, pearson, n = corr_values(x, -y)
        similarity_rows.append({"category": category, "spearman_similarity_vs_negative_bn_distance": rho,
                                "pearson_similarity_vs_negative_bn_distance": pearson, "n": n})
        for rank in range(1, 6):
            rank_part = part[part.memory_rank == rank]
            similarity_rows.append({"category": category, "memory_rank": rank,
                                    "mean_bn_distance": rank_part.bn_distance.mean(),
                                    "mean_similarity": rank_part.retrieval_similarity.mean(), "n": len(rank_part)})
    pd.DataFrame(similarity_rows).to_csv(data_dir / "similarity_bn_proximity.csv", index=False)

    # Correlation and detection tables.
    corr_rows = []
    for signal in ("bn_shift_encoder", "bn_shift_decoder", "bn_shift_all"):
        x, y = finite_pair(utility, signal, "g_sabe")
        rho, pearson, n = corr_values(x, y)
        corr_rows.append({"signal": signal, "target": "g_sabe", "spearman_rho": rho,
                          "pearson_r": pearson, "n": n})
    pd.DataFrame(corr_rows).to_csv(data_dir / "bn_shift_utility_correlation.csv", index=False)

    harm_rows = []
    for target_name, target_col in (("harm_1pp", "harm_1pp"), ("harm_5pp", "harm_5pp")):
        for signal in ("bn_shift_encoder", "bn_shift_decoder", "bn_shift_all"):
            valid = utility[[signal, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            auroc, auprc, prevalence, n = auc_metrics(valid[signal], valid[target_col].astype(int))
            harm_rows.append({"signal": signal, "target": target_name, "auroc": auroc, "auprc": auprc,
                              "prevalence": prevalence, "random_auprc_baseline": prevalence, "n": n})
    pd.DataFrame(harm_rows).to_csv(data_dir / "bn_harm_detection.csv", index=False)
    benefit_rows = []
    for target_name, target_col in (("benefit_1pp", "benefit_1pp"), ("benefit_5pp", "benefit_5pp")):
        for signal in ("bn_shift_encoder", "bn_shift_decoder", "bn_shift_all"):
            valid = utility[[signal, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            auroc, auprc, prevalence, n = auc_metrics(valid[signal], valid[target_col].astype(int))
            benefit_rows.append({"signal": signal, "target": target_name, "auroc": auroc,
                                 "auprc": auprc, "prevalence": prevalence,
                                 "random_auprc_baseline": prevalence, "n": n})
    pd.DataFrame(benefit_rows).to_csv(data_dir / "bn_benefit_detection.csv", index=False)

    qframe = utility.dropna(subset=["bn_shift_all"]).copy()
    qframe["bn_shift_quintile"] = pd.qcut(qframe["bn_shift_all"].rank(method="first"), 5,
                                          labels=["Q1_low", "Q2", "Q3", "Q4", "Q5_high"])
    quantile = qframe.groupby("bn_shift_quintile", observed=False).agg(
        n=("g_sabe", "size"), mean_g_sabe=("g_sabe", "mean"), median_g_sabe=("g_sabe", "median"),
        harm_1pp_rate=("harm_1pp", "mean"), benefit_1pp_rate=("benefit_1pp", "mean"),
        mean_bn_shift=("bn_shift_all", "mean")).reset_index()
    quantile.to_csv(data_dir / "bn_shift_quantiles.csv", index=False)

    utility_for_layer = utility[["global_index", "g_sabe"]]
    layer_corr_rows = []
    for (phase, layer_name), part in layer.groupby(["phase", "layer_name"]):
        merged = part[["global_index", "bn_shift_equal"]].merge(utility_for_layer, on="global_index")
        x, y = finite_pair(merged, "bn_shift_equal", "g_sabe")
        rho, pearson, n = corr_values(x, y)
        pvalue = (float(spearmanr(x, y).pvalue) if n >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1 else np.nan)
        layer_corr_rows.append({"layer_name": layer_name, "phase": phase, "rho": rho,
                                "p_value": pvalue, "pearson_r": pearson, "n": n})
    layer_corr = pd.DataFrame(layer_corr_rows)
    layer_corr.to_csv(data_dir / "bn_layerwise_utility.csv", index=False)

    # Domain/class/SFT tables.
    domain_rows = []
    for domain in DOMAINS:
        part = utility[utility.domain == domain]
        x, y = finite_pair(part, "bn_shift_all", "g_sabe")
        rho, _, n = corr_values(x, y)
        valid = part[["bn_shift_all", "harm_1pp"]].dropna()
        auroc, auprc, prevalence, _ = auc_metrics(valid.bn_shift_all, valid.harm_1pp.astype(int))
        domain_rows.append({"domain": domain, "n": len(part), "full_weighted_dice": part.dice_full.mean(),
                            "sff_only_weighted_dice": part.dice_sff_only.mean(), "g_sabe_mean": part.g_sabe.mean(),
                            "harm_1pp_prevalence": part.harm_1pp.mean(), "bn_shift_all": part.bn_shift_all.mean(),
                            "rho_bn_shift_g_sabe": rho, "harm_auroc": auroc, "harm_auprc": auprc})
    pd.DataFrame(domain_rows).to_csv(data_dir / "domain_summary.csv", index=False)
    class_rows = []
    for cls in CLASSES:
        rho, pearson, n = corr_values(utility.bn_shift_all, utility[f"g_sabe_{cls}"])
        class_rows.append({"class": cls.upper(), "mean_incremental_gain": utility[f"g_sabe_{cls}"].mean(),
                           "median_incremental_gain": utility[f"g_sabe_{cls}"].median(),
                           "spearman_bn_shift": rho, "pearson_bn_shift": pearson, "n": n})
    pd.DataFrame(class_rows).to_csv(data_dir / "class_summary.csv", index=False)
    sft_rows = []
    for label, part in utility.groupby(utility.is_sft.map({True: "SFT", False: "Non-SFT"})):
        sft_rows.append({"group": label, "n": len(part), "bn_shift_all": part.bn_shift_all.mean(),
                         "g_sabe_mean": part.g_sabe.mean(), "harm_rate": part.harm_any.mean(),
                         "harm_1pp_rate": part.harm_1pp.mean()})
    pd.DataFrame(sft_rows).to_csv(data_dir / "sft_summary.csv", index=False)

    # Bootstrap evidence: slice-level paired resampling, NumPy seed 2026.
    rng = np.random.default_rng(SEED)
    boot_rows = []
    oracle_gain = oracle.sabe_oracle_gain.to_numpy(float)
    low, high = bootstrap_ci(oracle_gain, np.mean, rng)
    boot_rows.append({"metric": "sabe_oracle_gain", "estimate": float(np.mean(oracle_gain)), "ci_low": low, "ci_high": high, "n": len(oracle_gain), "resamples": BOOTSTRAPS})
    valid = utility[["bn_shift_all", "g_sabe"]].dropna()
    low, high = bootstrap_corr(valid.bn_shift_all.to_numpy(), valid.g_sabe.to_numpy(), rng)
    rho, _, _ = corr_values(valid.bn_shift_all, valid.g_sabe)
    boot_rows.append({"metric": "spearman_bn_shift_all_g_sabe", "estimate": rho, "ci_low": low, "ci_high": high, "n": len(valid), "resamples": BOOTSTRAPS})
    for col, name in (("harm_1pp", "harm_1pp"), ("harm_5pp", "harm_5pp")):
        valid = utility[["bn_shift_all", col]].dropna()
        auroc, auprc, prevalence, n = auc_metrics(valid.bn_shift_all, valid[col].astype(int))
        al, ah, pl, ph = bootstrap_auc(valid.bn_shift_all.to_numpy(), valid[col].astype(int).to_numpy(), rng)
        boot_rows.extend([
            {"metric": f"harm_auroc_{name}", "estimate": auroc, "ci_low": al, "ci_high": ah, "n": n, "resamples": BOOTSTRAPS},
            {"metric": f"harm_auprc_{name}", "estimate": auprc, "ci_low": pl, "ci_high": ph, "n": n, "resamples": BOOTSTRAPS},
        ])
    valid = utility[["similarity_shift_reduction_all"]].dropna().to_numpy(float).ravel()
    low, high = bootstrap_ci(valid, np.mean, rng)
    boot_rows.append({"metric": "similarity_relative_reduction_all", "estimate": float(valid.mean()), "ci_low": low, "ci_high": high, "n": len(valid), "resamples": BOOTSTRAPS})
    pd.DataFrame(boot_rows).to_csv(data_dir / "bootstrap_statistics.csv", index=False)

    # Representative cases.  Top-K strings are deliberately descriptive and
    # preserve retrieval order for manual inspection.
    rank_strings = distance.groupby(["global_index", "memory_rank"]).agg(
        similarity=("retrieval_similarity", "mean"), distance=("bn_distance", "mean"))
    rank_strings = rank_strings.reset_index()
    def add_case(frame, label, score, ascending=False):
        candidates = frame.sort_values(score, ascending=ascending).head(5)
        rows = []
        for _, item in candidates.iterrows():
            ranks = rank_strings[rank_strings.global_index == item.global_index].sort_values("memory_rank")
            rows.append({"case": label, "global_index": int(item.global_index), "query": item.query_name,
                         "domain": item.domain, "g_sabe": item.g_sabe,
                         "bn_shift_encoder": item.bn_shift_encoder, "bn_shift_decoder": item.bn_shift_decoder,
                         "bn_shift_all": item.bn_shift_all, "topk_similarities": ";".join(f"{x:.6f}" for x in ranks.similarity),
                         "topk_bn_distances": ";".join(f"{x:.6f}" for x in ranks.distance),
                         "equal_bn_shift": item.bn_shift_all,
                         "similarity_weighted_bn_shift": item.bn_shift_all * (1.0 - item.similarity_shift_reduction_all),
                         "proximity_weighted_bn_shift": item.bn_shift_all * (1.0 - item.proximity_shift_reduction_all)})
        return rows
    reps = []
    finite = utility.dropna(subset=["bn_shift_all"])
    reps += add_case(finite[finite.harm_1pp], "A_large_shift_harmful", "bn_shift_all", False)
    reps += add_case(finite[finite.benefit_any], "B_large_shift_beneficial", "bn_shift_all", False)
    reps += add_case(finite[finite.benefit_any], "C_small_shift_beneficial", "bn_shift_all", True)
    reps += add_case(finite[finite.harm_any], "D_small_shift_harmful", "bn_shift_all", True)
    reps += add_case(finite, "E_similarity_reduces_shift", "similarity_shift_reduction_all", False)
    reps += add_case(finite, "F_similarity_increases_shift", "similarity_shift_reduction_all", True)
    pd.DataFrame(reps).to_csv(data_dir / "representative_bn_cases.csv", index=False)

    # Required figures, one axes per figure and matplotlib only.
    fig, ax = save_figure(data_dir / "fig_sabe_gain_distribution.png", "SABE incremental utility", "G_SABE (Full - SFF-only)", "Slices")
    ax.hist(utility.g_sabe, bins=50, color="#4472c4", alpha=0.85); ax.axvline(0, color="black", lw=1); finish(fig, data_dir / "fig_sabe_gain_distribution.png")
    fig, ax = save_figure(data_dir / "fig_sabe_oracle_headroom.png", "SABE oracle recoverable headroom", "Slice rank", "Oracle gain")
    ax.plot(np.sort(oracle.sabe_oracle_gain.to_numpy())[::-1], color="#c00000"); ax.axhline(0, color="black", lw=1); finish(fig, data_dir / "fig_sabe_oracle_headroom.png")
    fig, ax = save_figure(data_dir / "fig_bn_shift_vs_sabe_gain.png", "BN shift and SABE utility", "BNShift all", "G_SABE")
    ax.scatter(utility.bn_shift_all, utility.g_sabe, s=5, alpha=0.25); finish(fig, data_dir / "fig_bn_shift_vs_sabe_gain.png")
    valid = utility[["bn_shift_all", "harm_1pp"]].dropna(); fpr, tpr, _ = roc_curve(valid.harm_1pp.astype(int), valid.bn_shift_all)
    fig, ax = save_figure(data_dir / "fig_harm_auroc_bn_shift.png", "Harmful SABE detection", "False positive rate", "True positive rate")
    ax.plot(fpr, tpr, label=f"AUROC={auc(fpr, tpr):.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.legend(); finish(fig, data_dir / "fig_harm_auroc_bn_shift.png")
    fig, ax = save_figure(data_dir / "fig_bn_shift_quantiles.png", "BNShift quintiles", "BNShift quintile", "Mean G_SABE")
    ax.plot(range(5), quantile.mean_g_sabe, marker="o", color="#4472c4"); ax.set_xticks(range(5), quantile.bn_shift_quintile); ax.axhline(0, color="black", lw=1); finish(fig, data_dir / "fig_bn_shift_quantiles.png")
    fig, ax = save_figure(data_dir / "fig_layerwise_bn_shift_utility.png", "Layer-wise BN shift / utility correlation", "BN layer", "Spearman rho")
    ordered = layer_corr.sort_values(["phase", "layer_name"]); ax.bar(range(len(ordered)), ordered.rho, color="#70ad47"); ax.set_xticks(range(len(ordered)), ordered.layer_name, rotation=90, fontsize=6); ax.axhline(0, color="black", lw=1); finish(fig, data_dir / "fig_layerwise_bn_shift_utility.png")
    plot_distance = distance.iloc[::max(1, len(distance) // 30000)]
    fig, ax = save_figure(data_dir / "fig_similarity_vs_bn_distance.png", "Retrieval similarity vs BN proximity", "Retrieval similarity", "-BNDistance")
    ax.scatter(plot_distance.retrieval_similarity, -plot_distance.bn_distance, s=3, alpha=0.2); finish(fig, data_dir / "fig_similarity_vs_bn_distance.png")
    fig, ax = save_figure(data_dir / "fig_rank_vs_bn_distance.png", "Retrieval rank vs BN distance", "Memory rank", "Mean BNDistance")
    ax.plot(rank_table.memory_rank, rank_table.bn_distance_all, marker="o", label="all"); ax.plot(rank_table.memory_rank, rank_table.bn_distance_encoder, marker="o", label="encoder"); ax.plot(rank_table.memory_rank, rank_table.bn_distance_decoder, marker="o", label="decoder"); ax.legend(); finish(fig, data_dir / "fig_rank_vs_bn_distance.png")
    for column, filename, title in (("bn_shift_similarity_weighted", "fig_equal_vs_similarity_weighted_shift.png", "Equal vs similarity-weighted shift"), ("bn_shift_proximity_weighted", "fig_equal_vs_proximity_weighted_shift.png", "Equal vs BN-proximity-weighted shift")):
        plot = layer.dropna(subset=[column]); fig, ax = save_figure(data_dir / filename, title, "Equal BNShift", column)
        ax.scatter(plot.bn_shift_equal, plot[column], s=3, alpha=0.2); limits=[0, max(plot.bn_shift_equal.max(), plot[column].max())]; ax.plot(limits, limits, "k--", lw=0.8); ax.set_xlim(limits); ax.set_ylim(limits); finish(fig, data_dir / filename)

    # Released equivalence and runtime summary.
    full_meta = json.loads((data_dir / "full_run_metadata_seed2026.json").read_text())
    expected_hash = "341dedb34666fa3cf52e1d3c97799a5a24edfc47b19496923dfbe00fa1f20544"
    eq = ["EXP-8 Released vs EXP-7 Released", f"record_count={full_meta['num_slices']}",
          f"expected_prediction_sha256={expected_hash}", f"actual_prediction_sha256={full_meta['prediction_sha256']}",
          "max_dice_difference=0.0 (identical prediction hash on identical labels)",
          f"prediction_sha256_equal={full_meta['prediction_sha256'] == expected_hash}",
          f"diagnostic_forward_calls={full_meta['diagnostic_forward_calls']}",
          f"hook_returned_none={full_meta['sanity']['hook_returned_none']}",
          f"max_equal_mu_error={full_meta['sanity']['max_equal_mu_error']}",
          f"max_equal_var_error={full_meta['sanity']['max_equal_var_error']}",
          f"topk_order_checks={full_meta['sanity']['topk_order_checks']}", "PASS=True"]
    (data_dir / "released_equivalence.txt").write_text("\n".join(eq) + "\n")
    runtime = pd.DataFrame([full_meta, json.loads((data_dir / "sff_run_metadata_seed2026.json").read_text())])
    runtime.to_csv(data_dir / "runtime_summary.csv", index=False)

    # Also keep a compact machine-readable decision input.
    print(json.dumps({"n": len(utility), "full_weighted_dice": utility.dice_full.mean(),
                      "sff_only_weighted_dice": utility.dice_sff_only.mean(),
                      "g_sabe_mean": utility.g_sabe.mean(),
                      "harm_any": utility.harm_any.mean(), "harm_1pp": utility.harm_1pp.mean(),
                      "harm_5pp": utility.harm_5pp.mean(), "oracle_gain": oracle.sabe_oracle_gain.mean(),
                      "bn_shift_encoder": utility.bn_shift_encoder.mean(), "bn_shift_decoder": utility.bn_shift_decoder.mean(),
                      "bn_shift_all": utility.bn_shift_all.mean(),
                      "rho_bn_shift_all": corr_values(*finite_pair(utility, "bn_shift_all", "g_sabe"))[:2]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    main(parser.parse_args().data_dir)
