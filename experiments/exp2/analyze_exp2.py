#!/usr/bin/env python3
"""Analyze EXP-2 class reliability and semantic retrieval diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp2"
SEEDS = [2024, 2025, 2026]
DOMAINS = ["B", "C", "D"]
CLASSES = [("lv", "LV"), ("myo", "MYO"), ("rv", "RV")]
LABELS = {"lv": "LV", "myo": "MYO", "rv": "RV"}
COLORS = {"lv": "#1f77b4", "myo": "#ff7f0e", "rv": "#2ca02c"}


def finite(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def avg(series):
    values = finite(series)
    return float(values.mean()) if len(values) else np.nan


def spread(series):
    values = finite(series)
    return float(values.std(ddof=0)) if len(values) else np.nan


def med(series):
    values = finite(series)
    return float(values.median()) if len(values) else np.nan


def bootstrap(values, seed=2026, samples=10000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def holm(pvalues):
    valid = [i for i, value in enumerate(pvalues) if np.isfinite(value)]
    result = [np.nan] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(valid, key=lambda i: pvalues[i])):
        adjusted = min(1.0, max(previous, pvalues[index] * (len(valid) - rank)))
        result[index] = adjusted
        previous = adjusted
    return result


def load_frames():
    frames = {}
    for seed in SEEDS:
        path = OUT / f"per_slice_diagnostics_seed{seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 4091:
            raise RuntimeError(f"Unexpected row count in {path}: {len(frame)}")
        frames[seed] = frame
    return frames


def sft_reliability(frames):
    rows = []
    for seed, frame in frames.items():
        for domain in DOMAINS:
            part = frame[(frame.domain == domain) & (frame.is_sft == True)]
            rows.append({
                "seed": seed, "domain": domain, "num_sft": len(part),
                "overall_mean": avg(part.anchor_dice_overall_present),
                "overall_median": med(part.anchor_dice_overall_present),
                "lv_mean": avg(part.anchor_dice_lv), "myo_mean": avg(part.anchor_dice_myo),
                "rv_mean": avg(part.anchor_dice_rv),
                "weakest_mean": avg(part.class_dice_min), "weakest_median": med(part.class_dice_min),
                "class_range_mean": avg(part.class_dice_range), "class_range_median": med(part.class_dice_range),
                "class_std_mean": avg(part.class_dice_std),
                "weak_gap_mean": avg(part.weak_class_gap), "weak_gap_median": med(part.weak_class_gap),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "sft_class_reliability_summary.csv", index=False)
    return result


def reliability_mismatch(frames):
    rows = []
    for seed, frame in frames.items():
        for domain in DOMAINS:
            part = frame[(frame.domain == domain) & (frame.is_sft == True)]
            for overall_threshold in (.75, .80, .85):
                for weak_threshold in (.50, .60, .70):
                    high = part.anchor_dice_overall_present >= overall_threshold
                    weak = pd.Series(False, index=part.index)
                    for suffix, _ in CLASSES:
                        weak = weak | (
                            part[f"gt_present_{suffix}"].astype(bool)
                            & part[f"anchor_dice_{suffix}"].lt(weak_threshold)
                        )
                    count = int(high.sum())
                    mismatch = int((high & weak).sum())
                    rows.append({
                        "seed": seed, "domain": domain,
                        "overall_threshold": overall_threshold, "weak_threshold": weak_threshold,
                        "num_high_overall_sft": count,
                        "num_high_overall_but_weak_class": mismatch,
                        "ratio": mismatch / count if count else np.nan,
                    })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "sft_reliability_mismatch.csv", index=False)
    return result


def correlation_rows(frames):
    rows = []
    for seed, frame in frames.items():
        frame = frame.copy()
        frame["reliability_score"] = -pd.to_numeric(frame.ccd, errors="coerce")
        subsets = [("All", frame), ("SFT", frame[frame.is_sft == True])]
        for subset_name, subset in subsets:
            for domain in DOMAINS + ["All"]:
                part = subset if domain == "All" else subset[subset.domain == domain]
                targets = [("overall", "anchor_dice_overall_present", None)]
                targets.extend((suffix, f"anchor_dice_{suffix}", f"gt_present_{suffix}") for suffix, _ in CLASSES)
                for target_name, target_field, presence_field in targets:
                    values = pd.DataFrame({"x": part.reliability_score, "y": part[target_field]})
                    if presence_field:
                        values = values[part[presence_field] == True]
                    values = values.dropna()
                    if len(values) >= 3 and values.x.nunique() > 1 and values.y.nunique() > 1:
                        pearson = stats.pearsonr(values.x, values.y)
                        spearman = stats.spearmanr(values.x, values.y)
                        rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                     "predictor": "reliability_score=-CCD", "target": target_name,
                                     "pearson_r": pearson.statistic, "spearman_rho": spearman.statistic,
                                     "p_value": pearson.pvalue, "n": len(values)})
                    else:
                        rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                     "predictor": "reliability_score=-CCD", "target": target_name,
                                     "pearson_r": np.nan, "spearman_rho": np.nan,
                                     "p_value": np.nan, "n": len(values)})
                for suffix, _ in CLASSES:
                    values = pd.DataFrame({"x": part.anchor_dice_overall_present,
                                           "y": part[f"anchor_dice_{suffix}"],
                                           "present": part[f"gt_present_{suffix}"]})
                    values = values[values.present == True].dropna()
                    if len(values) >= 3 and values.x.nunique() > 1 and values.y.nunique() > 1:
                        pearson = stats.pearsonr(values.x, values.y)
                        spearman = stats.spearmanr(values.x, values.y)
                        rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                     "predictor": "anchor_overall_present", "target": suffix,
                                     "pearson_r": pearson.statistic, "spearman_rho": spearman.statistic,
                                     "p_value": pearson.pvalue, "n": len(values)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "class_reliability_correlations.csv", index=False)
    return result


def overlap_tables(frames):
    relation_fields = {
        "global_vs_lv": "global_lv_overlap", "global_vs_myo": "global_myo_overlap",
        "global_vs_rv": "global_rv_overlap", "lv_vs_myo": "lv_myo_overlap",
        "lv_vs_rv": "lv_rv_overlap", "myo_vs_rv": "myo_rv_overlap",
    }
    rows = []
    for seed, frame in frames.items():
        frame = frame[frame.pool_size_before >= 5]
        for subset_name, subset in [("All", frame), ("SFT", frame[frame.is_sft == True])]:
            for domain in DOMAINS + ["All"]:
                part = subset if domain == "All" else subset[subset.domain == domain]
                for relation, field in relation_fields.items():
                    values = finite(part[field])
                    rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                 "relation": relation, "n": len(values), "mean": avg(values),
                                 "std": spread(values), "median": med(values),
                                 "p25": float(values.quantile(.25)) if len(values) else np.nan,
                                 "p75": float(values.quantile(.75)) if len(values) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "topk_overlap_summary.csv", index=False)
    return result


def quality_long(frames):
    rows = []
    for seed, frame in frames.items():
        for _, item in frame.iterrows():
            for suffix, label in CLASSES:
                rows.append({
                    "seed": seed, "global_index": item.global_index, "domain": item.domain,
                    "is_sft": item.is_sft, "class": suffix,
                    "gt_present": item[f"gt_present_{suffix}"],
                    "global_presence": item[f"global_neighbor_presence_{suffix}"],
                    "class_presence": item[f"class_neighbor_presence_{suffix}"],
                    "presence_gain": item[f"presence_gain_{suffix}"],
                    "Q_global": item[f"Q_global_{suffix}"], "Q_class": item[f"Q_class_{suffix}"],
                    "quality_gain": item[f"deltaQ_{suffix}"],
                    "global_mass": item[f"global_memory_mass_{suffix}"],
                    "class_mass": item[f"class_memory_mass_{suffix}"],
                    "overlap": item[f"global_{suffix}_overlap"],
                    "query_anchor_class_dice": item[f"anchor_dice_{suffix}"],
                    "query_anchor_overall_present": item.anchor_dice_overall_present,
                })
    return pd.DataFrame(rows)


def neighbor_quality_outputs(long):
    per_slice = long.copy()
    per_slice["weakness"] = per_slice.query_anchor_overall_present - per_slice.query_anchor_class_dice
    per_slice.to_csv(OUT / "neighbor_quality_per_slice.csv", index=False)
    summary_rows, stat_rows = [], []
    for seed in sorted(long.seed.unique()):
        for subset_name, subset in [("All", long[long.seed == seed]),
                                    ("SFT", long[(long.seed == seed) & (long.is_sft == True)])]:
            for domain in DOMAINS + ["All"]:
                domain_part = subset if domain == "All" else subset[subset.domain == domain]
                for class_name, _ in CLASSES:
                    part = domain_part[domain_part['class'] == class_name]
                    both = part.dropna(subset=["Q_global", "Q_class"])
                    gains = finite(both.quality_gain).to_numpy()
                    ci_low, ci_high = bootstrap(gains)
                    summary_rows.append({
                        "seed": seed, "subset": subset_name, "domain": domain, "class": class_name,
                        "n_queries": len(part), "n_quality_pairs": len(gains),
                        "global_neighbor_presence_rate": avg(part.global_presence),
                        "class_neighbor_presence_rate": avg(part.class_presence),
                        "presence_gain": avg(part.presence_gain),
                        "global_neighbor_anchor_dice_mean": avg(part.Q_global),
                        "class_neighbor_anchor_dice_mean": avg(part.Q_class),
                        "quality_gain": avg(part.quality_gain), "quality_gain_median": med(part.quality_gain),
                        "fraction_quality_gain_positive": float((gains > 0).mean()) if len(gains) else np.nan,
                        "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
                    })
                    if len(gains):
                        try:
                            test = stats.wilcoxon(gains, alternative="two-sided", zero_method="wilcox")
                            pvalue = float(test.pvalue)
                        except ValueError:
                            pvalue = np.nan
                        ci_low, ci_high = bootstrap(gains)
                        stat_rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                          "class": class_name, "n": len(gains), "mean_delta": float(gains.mean()),
                                          "median_delta": float(np.median(gains)),
                                          "bootstrap_ci95_low": ci_low, "bootstrap_ci95_high": ci_high,
                                          "wilcoxon_raw_p": pvalue})
    summary = pd.DataFrame(summary_rows)
    statistics = pd.DataFrame(stat_rows)
    if len(statistics):
        statistics["wilcoxon_holm_p"] = holm(statistics.wilcoxon_raw_p.tolist())
    summary.to_csv(OUT / "neighbor_quality_summary.csv", index=False)
    statistics.to_csv(OUT / "retrieval_quality_statistics.csv", index=False)
    return summary, statistics


def gain_correlations(long):
    rows = []
    for seed in sorted(long.seed.unique()):
        for subset_name, subset in [("All", long[long.seed == seed]),
                                    ("SFT", long[(long.seed == seed) & (long.is_sft == True)])]:
            for domain in DOMAINS + ["All"]:
                domain_part = subset if domain == "All" else subset[subset.domain == domain]
                for class_name, _ in CLASSES:
                    part = domain_part[(domain_part['class'] == class_name) & (domain_part.gt_present == True)]
                    part = part.dropna(subset=["quality_gain", "query_anchor_class_dice", "query_anchor_overall_present"])
                    weakness = part.query_anchor_overall_present - part.query_anchor_class_dice
                    for predictor, values in [("weakness", weakness),
                                               ("query_anchor_class_dice", part.query_anchor_class_dice)]:
                        valid = pd.DataFrame({"x": values, "y": part.quality_gain}).dropna()
                        if len(valid) >= 3 and valid.x.nunique() > 1 and valid.y.nunique() > 1:
                            pearson = stats.pearsonr(valid.x, valid.y)
                            spearman = stats.spearmanr(valid.x, valid.y)
                            rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                         "class": class_name, "predictor": predictor,
                                         "pearson_r": pearson.statistic, "spearman_rho": spearman.statistic,
                                         "p_value": pearson.pvalue, "n": len(valid)})
                        else:
                            rows.append({"seed": seed, "subset": subset_name, "domain": domain,
                                         "class": class_name, "predictor": predictor,
                                         "pearson_r": np.nan, "spearman_rho": np.nan,
                                         "p_value": np.nan, "n": len(valid)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "retrieval_gain_correlation.csv", index=False)
    return result


def multiseed_summary(long, sft_summary, overlap_summary, quality_summary):
    rows = []
    for seed in SEEDS:
        for domain in DOMAINS:
            for class_name, _ in CLASSES:
                sft = sft_summary[(sft_summary.seed == seed) & (sft_summary.domain == domain)]
                overlap = overlap_summary[(overlap_summary.seed == seed) & (overlap_summary.domain == domain) &
                                          (overlap_summary.subset == "All") & (overlap_summary.relation == f"global_vs_{class_name}")]
                quality = quality_summary[(quality_summary.seed == seed) & (quality_summary.domain == domain) &
                                          (quality_summary.subset == "All") & (quality_summary['class'] == class_name)]
                sft_class = sft[f"{class_name}_mean"].iloc[0] if len(sft) else np.nan
                rows.append({"seed": seed, "domain": domain, "class": class_name,
                             "sft_class_dice_mean": sft_class,
                             "global_class_overlap": overlap['mean'].iloc[0] if len(overlap) else np.nan,
                             "quality_gain": quality.quality_gain.iloc[0] if len(quality) else np.nan,
                             "presence_gain": quality.presence_gain.iloc[0] if len(quality) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "multiseed_summary.csv", index=False)
    return result


def representative_cases(frames):
    frame = frames[2026].copy()
    frame["weakest_class"] = frame[["anchor_dice_lv", "anchor_dice_myo", "anchor_dice_rv"]].idxmin(axis=1)
    candidates = []
    case_a = frame[(frame.is_sft == True) & (frame.anchor_dice_overall_present >= .80) & (frame.class_dice_min < .60)].copy()
    case_a["case_type"] = "A_high_overall_weak_class"
    candidates.append(case_a.sort_values(["anchor_dice_overall_present", "class_dice_min"], ascending=[False, True]).head(6))
    for suffix, _ in CLASSES:
        case_b = frame[(frame.pool_size_before >= 5) & (frame[f"global_{suffix}_overlap"] < .40) &
                       (frame[f"deltaQ_{suffix}"] > 0)].copy()
        case_b["case_type"] = f"B_low_overlap_high_gain_{suffix}"
        case_b["case_class"] = suffix
        candidates.append(case_b.sort_values(f"deltaQ_{suffix}", ascending=False).head(3))
        case_c = frame[(frame.pool_size_before >= 5) & (frame[f"global_{suffix}_overlap"] >= .80) &
                       (frame[f"deltaQ_{suffix}"].abs() < .01)].copy()
        case_c["case_type"] = f"C_high_overlap_near_zero_gain_{suffix}"
        case_c["case_class"] = suffix
        candidates.append(case_c.sort_values(f"global_{suffix}_overlap", ascending=False).head(2))
    selected = pd.concat(candidates, ignore_index=True).drop_duplicates("slice_name").head(20)
    rows = []
    for _, item in selected.iterrows():
        suffix = item.get("case_class")
        if not isinstance(suffix, str):
            suffix = item.weakest_class.replace("anchor_dice_", "")
        rows.append({
            "case_type": item.case_type, "domain": item.domain, "slice_name": item.slice_name,
            "is_sft": item.is_sft, "overall_anchor_dice": item.anchor_dice_overall_present,
            "lv_dice": item.anchor_dice_lv, "myo_dice": item.anchor_dice_myo, "rv_dice": item.anchor_dice_rv,
            "weakest_class": suffix.upper(), "case_class": suffix,
            "global_topk": item.global_topk_names, "class_topk": item[f"{suffix}_topk_names"],
            "overlap": item[f"global_{suffix}_overlap"], "Q_global": item[f"Q_global_{suffix}"],
            "Q_class": item[f"Q_class_{suffix}"], "deltaQ": item[f"deltaQ_{suffix}"],
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "representative_cases.csv", index=False)
    return result


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def figures(frames, correlations, overlap_summary, quality_summary, sft_summary, long):
    frame = frames[2026]
    accepted = frame[frame.is_sft == True]
    values, labels, colors = [], [], []
    for suffix, label in CLASSES:
        values.append(accepted[f"anchor_dice_{suffix}"].dropna().to_numpy() * 100)
        labels.append(label); colors.append(COLORS[suffix])
    fig, ax = plt.subplots(figsize=(8, 5)); box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors): patch.set_facecolor(color); patch.set_alpha(.65)
    ax.set_ylabel("anchor Dice (%)"); ax.set_title("Accepted SFT class reliability")
    savefig(OUT / "fig_sft_class_dice_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5)); range_values = [frame[frame.domain == d].class_dice_range.dropna().to_numpy() * 100 for d in DOMAINS]
    box = ax.boxplot(range_values, labels=DOMAINS, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ["#4c78a8", "#f58518", "#54a24b"]): patch.set_facecolor(color); patch.set_alpha(.65)
    ax.set_ylabel("class Dice range (%)"); ax.set_title("Class-level reliability disparity")
    savefig(OUT / "fig_sft_class_gap_distribution.png")

    corr = correlations[(correlations.seed == 2026) & (correlations.subset == "All") & (correlations.domain == "All") & (correlations.predictor == "reliability_score=-CCD")]
    corr = corr.set_index("target").reindex(["overall", "lv", "myo", "rv"])
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["Overall", "LV", "MYO", "RV"], corr.spearman_rho, color=["#555555"] + [COLORS[x] for x in ["lv", "myo", "rv"]])
    ax.axhline(0, color="black", linewidth=.7); ax.set_ylabel("Spearman rho"); ax.set_title("CCD reliability vs anchor class Dice")
    savefig(OUT / "fig_ccd_vs_class_dice.png")

    ov = overlap_summary[(overlap_summary.seed == 2026) & (overlap_summary.subset == "All") & (overlap_summary.domain == "All")]
    names = ["global_vs_lv", "global_vs_myo", "global_vs_rv"]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["Global-LV", "Global-MYO", "Global-RV"], [ov[ov.relation == n]["mean"].iloc[0] for n in names], color=[COLORS[x] for x in ["lv", "myo", "rv"]])
    ax.set_ylim(0, 1.05); ax.set_ylabel("overlap fraction"); ax.set_title("Global vs class-wise Top-K overlap")
    savefig(OUT / "fig_global_class_topk_overlap.png")

    names = ["lv_vs_myo", "lv_vs_rv", "myo_vs_rv"]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["LV-MYO", "LV-RV", "MYO-RV"], [ov[ov.relation == n]["mean"].iloc[0] for n in names], color="#777777")
    ax.set_ylim(0, 1.05); ax.set_ylabel("overlap fraction"); ax.set_title("Inter-class Top-K overlap")
    savefig(OUT / "fig_interclass_topk_overlap.png")

    q = quality_summary[(quality_summary.seed == 2026) & (quality_summary.subset == "All") & (quality_summary.domain == "All")]
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["LV", "MYO", "RV"], [q[q['class'] == s].presence_gain.iloc[0] for s, _ in CLASSES], color=[COLORS[s] for s, _ in CLASSES])
    ax.axhline(0, color="black", linewidth=.7); ax.set_ylabel("presence gain"); ax.set_title("Class-wise neighbor GT presence gain")
    savefig(OUT / "fig_neighbor_presence_gain.png")

    q = quality_summary[(quality_summary.seed == 2026) & (quality_summary.subset == "All") & (quality_summary.domain.isin(DOMAINS))]
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(DOMAINS)); width = .25
    for offset, (suffix, label) in enumerate(CLASSES):
        ax.bar(x + (offset - 1) * width, [q[(q.domain == d) & (q['class'] == suffix)].quality_gain.iloc[0] for d in DOMAINS], width, label=label, color=COLORS[suffix])
    ax.axhline(0, color="black", linewidth=.7); ax.set_xticks(x, DOMAINS); ax.set_ylabel("ΔQ = Q_class − Q_global"); ax.set_title("Neighbor class quality gain"); ax.legend()
    savefig(OUT / "fig_neighbor_quality_gain.png")

    for suffix, label in CLASSES:
        part = long[(long.seed == 2026) & (long['class'] == suffix) & (long.gt_present == True) & (long.global_index >= 0)]
        fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(part.query_anchor_class_dice * 100, part.quality_gain, s=4, alpha=.25, color=COLORS[suffix])
        ax.axhline(0, color="black", linewidth=.7); ax.set_xlabel(f"query {label} anchor Dice (%)"); ax.set_ylabel("ΔQ"); ax.set_title(f"Retrieval gain vs query quality: {label}")
        savefig(OUT / f"fig_retrieval_gain_vs_quality_{suffix}.png")


def fmt(value, scale=100, digits=2):
    return "NA" if pd.isna(value) else f"{value * scale:.{digits}f}"


def build_report(frames, sft_summary, mismatch, correlations, overlap_summary, quality_summary,
                 quality_stats, gain_corr, multiseed, representatives, equivalence):
    primary = sft_summary[sft_summary.seed == 2026]
    main_quality = quality_summary[(quality_summary.seed == 2026) & (quality_summary.subset == "All")]
    main_overlap = overlap_summary[(overlap_summary.seed == 2026) & (overlap_summary.subset == "All")]
    sft_range = primary.class_range_mean.mean()
    high_mismatch = mismatch[(mismatch.seed == 2026) & (mismatch.overall_threshold == .80) & (mismatch.weak_threshold == .60)]
    high_mismatch_total = int(high_mismatch.num_high_overall_but_weak_class.sum())
    high_overall_total = int(high_mismatch.num_high_overall_sft.sum())
    high_mismatch_ratio = high_mismatch_total / high_overall_total if high_overall_total else np.nan
    correlations_main = correlations[(correlations.seed == 2026) & (correlations.subset == "All") & (correlations.domain == "All") & (correlations.predictor == "reliability_score=-CCD")]
    main_overlap_all = main_overlap[main_overlap.domain == "All"]
    global_class_overlap = [main_overlap_all[main_overlap_all.relation == f"global_vs_{s}"]["mean"].iloc[0] for s, _ in CLASSES]
    inter_overlap = [main_overlap_all[main_overlap_all.relation == relation]["mean"].iloc[0] for relation in ["lv_vs_myo", "lv_vs_rv", "myo_vs_rv"]]
    gains = [main_quality[(main_quality.domain == "All") & (main_quality['class'] == s)].quality_gain.iloc[0] for s, _ in CLASSES]
    gain_ci = [main_quality[(main_quality.domain == "All") & (main_quality['class'] == s)][["bootstrap_95_ci_low", "bootstrap_95_ci_high"]].iloc[0].tolist() for s, _ in CLASSES]
    stable_positive = []
    for suffix, _ in CLASSES:
        vals = quality_summary[(quality_summary.subset == "All") & (quality_summary.domain == "All") & (quality_summary['class'] == suffix)].quality_gain
        stable_positive.append(bool((vals > 0).all()))
    disparity = bool(sft_range >= .10)
    overlap_mismatch = bool(np.mean(global_class_overlap) < .90)
    gain_support = sum(g > 0 and ci[0] > 0 for g, ci in zip(gains, gain_ci)) >= 2 and sum(stable_positive) >= 2
    if disparity and overlap_mismatch and gain_support:
        case = "A"
    elif disparity and not gain_support:
        case = "B"
    elif overlap_mismatch and not disparity:
        case = "C"
    else:
        case = "D"
    lines = [
        "# EXP-2 Class-Level Reliability and Semantic Retrieval Diagnosis", "",
        "## 1. Objective", "",
        "Diagnostics only. No adaptation behavior, CCD, SFT admission, Top-K, memory update, SABE, SFF, or prediction rule was changed.", "",
        "## 2. Released Equivalence", "",
        f"{'PASS' if equivalence else 'FAIL'}: seed=2026 Full Released SicTTA output-derived Dice matches EXP-1 Released within 1e-8; see `released_equivalence.txt`.", "",
        "## 3. Image-Level vs Class-Level Reliability", "",
        f"For accepted SFT at seed=2026, mean class Dice range is {fmt(sft_range)} percentage points across domains.",
        "The accepted-SFT class reliability table is:", primary.to_markdown(index=False), "",
        "At overall anchor Dice >= 0.80 and weak-class threshold < 0.60:", high_mismatch.to_markdown(index=False), "",
        "## 4. CCD vs Class-wise Quality", "", correlations_main[["target", "spearman_rho", "pearson_r", "n"]].to_markdown(index=False), "",
        "## 5. Global vs Semantic Retrieval", "", main_overlap[main_overlap.relation.str.startswith("global_vs")].to_markdown(index=False), "",
        "## 6. Inter-Class Retrieval Difference", "", main_overlap[main_overlap.relation.isin(["lv_vs_myo", "lv_vs_rv", "myo_vs_rv"])].to_markdown(index=False), "",
        "## 7. Neighbor Anatomical Presence", "", main_quality[main_quality.domain == "All"][["class", "global_neighbor_presence_rate", "class_neighbor_presence_rate", "presence_gain"]].to_markdown(index=False), "",
        "## 8. Neighbor Quality", "", main_quality[main_quality.domain.isin(DOMAINS)][["domain", "class", "global_neighbor_anchor_dice_mean", "class_neighbor_anchor_dice_mean", "quality_gain", "bootstrap_95_ci_low", "bootstrap_95_ci_high"]].to_markdown(index=False), "",
        "## 9. Hard-Class Benefit", "", gain_corr[(gain_corr.seed == 2026) & (gain_corr.subset == "All") & (gain_corr.domain == "All")][["class", "predictor", "spearman_rho", "pearson_r", "n"]].to_markdown(index=False), "",
        "## 10. Multi-Seed Robustness", "", multiseed.groupby("class").agg({"sft_class_dice_mean": ["mean", "std"], "global_class_overlap": ["mean", "std"], "quality_gain": ["mean", "std"], "presence_gain": ["mean", "std"]}).to_markdown(), "",
        "## 11. Representative Cases", "", representatives.to_markdown(index=False), "",
        "## 12. Findings", "",
        f"Observed: accepted SFT mean class Dice range is {fmt(sft_range)} percentage points; global-vs-class overlap for LV/MYO/RV is " + ", ".join(fmt(value, 100, 2) for value in global_class_overlap) + "%; neighbor ΔQ is " + ", ".join(fmt(value) for value in gains) + " Dice points.",
        "Statistical results must be read with confidence intervals, paired tests, and seed direction; GT was used only offline.",
        "Interpretation is diagnostic and does not establish that a class-wise prototype method will improve adaptation.", "",
        "## 13. Decision for EXP-3", "",
        f"Predefined decision: **Case {case}**.",
    ]
    if case == "A":
        lines.append("Reliability disparity, nontrivial retrieval mismatch, positive quality gain for at least two classes, and multi-seed direction support entering EXP-3 Semantic Class Prototype Memory / Class-wise Retrieval.")
    elif case == "B":
        lines.append("Class-level reliability mismatch exists, but class-wise retrieval does not provide stable higher-quality neighbors; do not directly implement prototype memory. Consider reliability gating or anatomy-aware filtering in a future controlled study.")
    elif case == "C":
        lines.append("Retrieval differs structurally, but class reliability mismatch is weak; evidence is insufficient to claim a current SicTTA failure.")
    else:
        lines.append("Global and class-wise retrieval are sufficiently similar and/or quality gains are negligible; the prototype-memory direction is not supported.")
    lines += ["", "## Final Answers", "",
              f"1. Accepted-SFT LV/MYO/RV anchor Dice: {fmt(primary.lv_mean.mean())}% / {fmt(primary.myo_mean.mean())}% / {fmt(primary.rv_mean.mean())}%.",
              f"2. Mean accepted-SFT class Dice range: **{fmt(sft_range)} percentage points**.",
              f"3. Overall >=0.80 and any present class <0.60: **{fmt(high_mismatch_ratio, 100, 2) if len(high_mismatch) else 'NA'}%** ({high_mismatch_total}/{high_overall_total}).",
              "4. CCD reliability Spearman: " + ", ".join(f"{target}={correlations_main[correlations_main.target == target].spearman_rho.iloc[0]:.4f}" for target in ["overall", "lv", "myo", "rv"]) + ".",
              "5. Global-vs-Class overlap LV/MYO/RV: " + ", ".join(f"{value:.4f}" for value in global_class_overlap) + ".",
              "6. Inter-class overlap LV-MYO/LV-RV/MYO-RV: " + ", ".join(f"{value:.4f}" for value in inter_overlap) + ".",
              "7. Global neighbor GT-class presence LV/MYO/RV: " + ", ".join(fmt(main_quality[main_quality['class'] == s].global_neighbor_presence_rate.iloc[0], 100, 2) + "%" for s, _ in CLASSES) + ".",
              "8. Class neighbor GT-class presence LV/MYO/RV: " + ", ".join(fmt(main_quality[main_quality['class'] == s].class_neighbor_presence_rate.iloc[0], 100, 2) + "%" for s, _ in CLASSES) + ".",
              "9. Neighbor quality ΔQ LV/MYO/RV: " + ", ".join(fmt(value) + " pp" for value in gains) + ".",
              "10. ΔQ 95% CI excluding 0: " + ", ".join(label for (s, label), ci in zip(CLASSES, gain_ci) if ci[0] > 0 or ci[1] < 0) + ("." if any(ci[0] > 0 or ci[1] < 0 for ci in gain_ci) else "none."),
              "11. Hard-class query retrieval gain: " + ("supported by the weakness/deltaQ correlation" if any(gain_corr[(gain_corr.seed == 2026) & (gain_corr.subset == "All") & (gain_corr.domain == "All") & (gain_corr.predictor == "weakness")].spearman_rho > 0) else "not consistently supported") + ".",
              "12. Multi-seed stable: " + ("yes" if all(stable_positive) else "no; inspect mean ± std in multiseed_summary.csv") + ".",
              f"13. Final case: **Case {case}**.",
              f"14. Sufficient evidence for EXP-3 Semantic Class Prototype Memory: **{'yes, as a controlled next diagnostic' if case == 'A' else 'no'}**.", ""]
    (OUT / "exp2_report.md").write_text("\n".join(lines))
    return case


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    sft = sft_reliability(frames)
    mismatch = reliability_mismatch(frames)
    correlations = correlation_rows(frames)
    overlap_summary = overlap_tables(frames)
    long = quality_long(frames)
    quality_summary, quality_stats = neighbor_quality_outputs(long)
    gain_corr = gain_correlations(long)
    multiseed = multiseed_summary(long, sft, overlap_summary, quality_summary)
    representatives = representative_cases(frames)
    equivalence = released_equivalence(frames)
    figures(frames, correlations, overlap_summary, quality_summary, sft, long)
    case = build_report(frames, sft, mismatch, correlations, overlap_summary, quality_summary,
                        quality_stats, gain_corr, multiseed, representatives, equivalence)
    print(f"seeds={len(frames)} rows={sum(len(x) for x in frames.values())} equivalence={equivalence} case={case}")


def released_equivalence(frames):
    new = frames[2026]
    old = pd.read_csv(ROOT / "results/exp1/raw/released_seed2026.csv")
    fields = [("adapted_dice_average", "adapted_dice_average"),
              ("anchor_dice_lv", "anchor_dice_lv"), ("anchor_dice_myo", "anchor_dice_myo"),
              ("anchor_dice_rv", "anchor_dice_rv")]
    deltas = [abs(float(a) - float(b)) for left, right in fields for a, b in zip(new[left], old[right])]
    names_equal = len(new) == len(old) and all(new.slice_name == old.slice_name)
    exact = names_equal and max(deltas, default=0) <= 1e-8
    lines = ["EXP-2 Released seed=2026 vs EXP-1 Released equivalence", "",
             f"record_count_exp2={len(new)}", f"record_count_exp1={len(old)}",
             f"slice_names_and_order_equal={names_equal}", f"max_dice_difference={max(deltas, default=0):.12f}",
             "tolerance=1e-8", f"PASS={exact}", ""]
    for domain in DOMAINS:
        current = new[new.domain == domain].adapted_dice_average.mean() * 100
        reference = old[old.domain == domain].adapted_dice_average.mean() * 100
        lines.append(f"{domain},EXP-2={current:.12f},EXP-1={reference:.12f},delta={abs(current-reference):.12f}")
    (OUT / "released_equivalence.txt").write_text("\n".join(lines) + "\n")
    return exact


if __name__ == "__main__":
    main()
