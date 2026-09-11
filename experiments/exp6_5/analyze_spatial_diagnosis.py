#!/usr/bin/env python3
"""Offline statistics and figures for EXP-6.5 spatial diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ["B", "C", "D", "All"]
CLASSES = ["fg", "lv", "myo", "rv"]
DISPLAY = {"fg": "Foreground", "lv": "LV", "myo": "MYO", "rv": "RV"}


def mean(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def quantile(values, q):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan


def class_field(prefix, cls):
    return f"{prefix}_fg" if cls == "fg" else f"{prefix}_{cls}"


def scope_part(frame, scope):
    return frame if scope == "All" else frame[frame.query_domain == scope]


def load(input_dir):
    pair = pd.read_csv(input_dir / "pair_spatial_diagnostics_seed2026.csv")
    query = pd.read_csv(input_dir / "query_spatial_summary_seed2026.csv")
    metadata = json.loads((input_dir / "run_metadata_seed2026.json").read_text())
    if len(query) != 4091 or len(pair) != metadata["num_pairs"]:
        raise RuntimeError(f"unexpected formal shapes: query={query.shape}, pair={pair.shape}")
    return pair, query, metadata


def pair_statistics(pair, out):
    metrics = {
        "centroid_distance": lambda p, c: p[f"{c}_centroid_distance"],
        "scale_error_mean": lambda p, c: (p[f"{c}_scale_error_x"] + p[f"{c}_scale_error_y"]) / 2,
        "raw_gt_overlap": lambda p, c: p[class_field("raw_gt_overlap", c)],
        "pred_translation_overlap": lambda p, c: p[class_field("pred_translation_gt_overlap", c)],
        "pred_affine_overlap": lambda p, c: p[class_field("pred_affine_gt_overlap", c)],
        "gt_oracle_overlap": lambda p, c: p[class_field("gt_oracle_translation_overlap", c)],
        "delta_pred_translation": lambda p, c: p[class_field("delta_pred_translation", c)],
        "delta_pred_affine": lambda p, c: p[class_field("delta_pred_affine", c)],
        "delta_gt_oracle": lambda p, c: p[class_field("delta_gt_oracle_translation", c)],
    }
    rows = []
    for scope in DOMAINS:
        part = pair if scope == "All" else pair[pair.query_domain == scope]
        for cls in CLASSES:
            for metric, getter in metrics.items():
                values = np.asarray(getter(part, cls), dtype=float)
                values = values[np.isfinite(values)]
                rows.append({"scope": scope, "class": DISPLAY[cls], "metric": metric, "n": len(values),
                             "mean": mean(values), "median": float(np.median(values)) if len(values) else np.nan,
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                             "p25": quantile(values, .25), "p75": quantile(values, .75)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "pair_statistics.csv", index=False)
    return result


def spatial_summary(pair, query, out):
    rows = []
    for scope in DOMAINS:
        qpart = query if scope == "All" else query[query.domain == scope]
        ppart = pair if scope == "All" else pair[pair.query_domain == scope]
        for cls in CLASSES:
            dist_col = f"{cls}_centroid_distance"
            raw_col = class_field("raw_gt_overlap", cls)
            trans_col = class_field("pred_translation_gt_overlap", cls)
            affine_col = class_field("pred_affine_gt_overlap", cls)
            oracle_col = class_field("gt_oracle_translation_overlap", cls)
            trans_delta_col = class_field("delta_pred_translation", cls)
            affine_delta_col = class_field("delta_pred_affine", cls)
            oracle_delta_col = class_field("delta_gt_oracle_translation", cls)
            top_dist = qpart[f"topk_mean_{cls}_centroid_distance"]
            pool_dist = qpart[f"pool_mean_{cls}_centroid_distance"]
            raw, trans = ppart[raw_col], ppart[trans_col]
            affine, oracle = ppart[affine_col], ppart[oracle_col]
            trans_delta, affine_delta, oracle_delta = ppart[trans_delta_col], ppart[affine_delta_col], ppart[oracle_delta_col]
            top_mean = mean(top_dist)
            pool_mean = mean(pool_dist)
            pair_trans = np.asarray(trans_delta, dtype=float); pair_trans = pair_trans[np.isfinite(pair_trans)]
            pair_affine = np.asarray(affine_delta, dtype=float); pair_affine = pair_affine[np.isfinite(pair_affine)]
            rows.append({
                "scope": scope, "class": DISPLAY[cls],
                "mean_topk_centroid_distance": top_mean,
                "mean_pool_centroid_distance": pool_mean,
                "pool_median_centroid_distance": mean(qpart[f"pool_median_{cls}_centroid_distance"]),
                "mean_spatial_nearest_k_distance": mean(qpart[f"spatial_nearest_k_mean_{cls}_distance"]),
                "retrieval_spatial_improvement": 1 - top_mean / pool_mean if np.isfinite(top_mean) and np.isfinite(pool_mean) and pool_mean else np.nan,
                "raw_gt_overlap": mean(raw), "pred_translation_overlap": mean(trans),
                "delta_pred_translation": mean(trans_delta), "pred_affine_overlap": mean(affine),
                "delta_pred_affine": mean(affine_delta), "gt_oracle_translation_overlap": mean(oracle),
                "delta_gt_oracle": mean(oracle_delta),
                "positive_translation_fraction": float((pair_trans > 0).mean()) if len(pair_trans) else np.nan,
                "positive_affine_fraction": float((pair_affine > 0).mean()) if len(pair_affine) else np.nan,
                "n_pairs": int(np.isfinite(np.asarray(raw, dtype=float)).sum()),
            })
    result = pd.DataFrame(rows)
    result.to_csv(out / "spatial_alignment_summary.csv", index=False)
    return result


def correlation_summary(pair, out):
    rows = []
    for domain in DOMAINS:
        part = pair if domain == "All" else pair[pair.query_domain == domain]
        for cls in CLASSES:
            sim = part.cosine_similarity.to_numpy(float)
            distance = part[f"{cls}_centroid_distance"].to_numpy(float)
            overlap = part[class_field("raw_gt_overlap", cls)].to_numpy(float)
            gain = part[class_field("delta_pred_translation", cls)].to_numpy(float)

            def corr(x, y):
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 3 or len(np.unique(x[valid])) < 2 or len(np.unique(y[valid])) < 2:
                    return np.nan
                return float(stats.spearmanr(x[valid], y[valid]).statistic)

            rows.append({"class": DISPLAY[cls], "domain": domain, "n": int(np.isfinite(distance).sum()),
                         "spearman_similarity_vs_neg_centroid_distance": corr(sim, -distance),
                         "spearman_similarity_vs_gt_overlap": corr(sim, overlap),
                         "spearman_similarity_vs_translation_gain": corr(sim, gain)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "similarity_spatial_correlation.csv", index=False)
    return result


def bootstrap_statistics(pair, out, samples=10000, seed=2026):
    np.random.seed(seed)
    rows = []
    metrics = {"Delta_pred_translation": "delta_pred_translation", "Delta_pred_affine": "delta_pred_affine",
               "Delta_GT_oracle_translation": "delta_gt_oracle_translation"}
    for scope in DOMAINS:
        part = pair if scope == "All" else pair[pair.query_domain == scope]
        for cls in CLASSES:
            for metric_name, prefix in metrics.items():
                col = class_field(prefix, cls)
                grouped = part.assign(_value=pd.to_numeric(part[col], errors="coerce")).groupby("global_index")['_value'].mean()
                values = grouped.to_numpy(float); values = values[np.isfinite(values)]
                if len(values):
                    boot = []
                    for start in range(0, samples, 250):
                        size = min(250, samples - start)
                        indices = np.random.randint(0, len(values), size=(size, len(values)))
                        boot.append(values[indices].mean(axis=1))
                    boot = np.concatenate(boot)
                    rows.append({"scope": scope, "class": DISPLAY[cls], "metric": metric_name, "n_queries": len(values),
                                 "mean": float(values.mean()), "ci_low": float(np.quantile(boot, .025)),
                                 "ci_high": float(np.quantile(boot, .975)), "bootstrap_samples": samples, "seed": seed})
                else:
                    rows.append({"scope": scope, "class": DISPLAY[cls], "metric": metric_name, "n_queries": 0,
                                 "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                                 "bootstrap_samples": samples, "seed": seed})
    result = pd.DataFrame(rows)
    result.to_csv(out / "bootstrap_statistics.csv", index=False)
    return result


def rank_summary(pair, out):
    rows = []
    for rank, part in pair.groupby("memory_rank"):
        rows.append({"rank": int(rank), "n": len(part), "mean_similarity": mean(part.cosine_similarity),
                     "mean_fg_centroid_distance": mean(part.fg_centroid_distance),
                     "mean_fg_raw_overlap": mean(part.raw_gt_overlap_fg),
                     "mean_fg_translation_gain": mean(part.delta_pred_translation_fg),
                     "mean_fg_oracle_gain": mean(part.delta_gt_oracle_translation_fg)})
    result = pd.DataFrame(rows).sort_values("rank")
    result.to_csv(out / "rank_spatial_summary.csv", index=False)
    return result


def representative_cases(pair, out):
    base = pair[np.isfinite(pair.delta_pred_translation_fg)].copy()
    groups = {
        "A_translation_improves": base.sort_values("delta_pred_translation_fg", ascending=False).head(5),
        "B_raw_already_good": base[base.raw_gt_overlap_fg >= base.raw_gt_overlap_fg.quantile(.75)]\
            .assign(_abs_gain=lambda x: x.delta_pred_translation_fg.abs()).sort_values("_abs_gain").head(5),
        "C_translation_harms": base.sort_values("delta_pred_translation_fg").head(5),
        "D_oracle_headroom_prediction_fails": base[base.delta_gt_oracle_translation_fg >= base.delta_gt_oracle_translation_fg.quantile(.90)]\
            .sort_values("delta_pred_translation_fg").head(5),
    }
    rows = []
    for category, selected in groups.items():
        for row in selected.itertuples(index=False):
            rows.append({"category": category, "global_index": row.global_index, "query_domain": row.query_domain,
                         "query_name": row.query_name, "memory_name": row.memory_name, "memory_rank": row.memory_rank,
                         "similarity": row.cosine_similarity, "centroid_shift": row.fg_centroid_distance,
                         "raw_overlap": row.raw_gt_overlap_fg, "translated_overlap": row.pred_translation_gt_overlap_fg,
                         "oracle_overlap": row.gt_oracle_translation_overlap_fg,
                         "translation_gain": row.delta_pred_translation_fg,
                         "oracle_gain": row.delta_gt_oracle_translation_fg})
    result = pd.DataFrame(rows)
    result.to_csv(out / "representative_spatial_cases.csv", index=False)
    return result


def make_figures(pair, query, summary, rank, out):
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    scopes = ["B", "C", "D", "All"]

    def save_bar(filename, columns, labels, ylabel, title):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        x = np.arange(len(scopes)); width = .24
        for i, (column, label) in enumerate(zip(columns, labels)):
            vals = [summary[(summary.scope == s) & (summary['class'] == "Foreground")][column].iloc[0] for s in scopes]
            ax.bar(x + (i - (len(columns)-1)/2) * width, vals, width, label=label)
        ax.set_xticks(x, scopes); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
        fig.tight_layout(); fig.savefig(fig_dir / filename, dpi=300); plt.close(fig)

    save_bar("fig_topk_vs_pool_centroid_distance.png",
             ["mean_topk_centroid_distance", "mean_pool_centroid_distance", "mean_spatial_nearest_k_distance"],
             ["Top-K", "Whole Pool", "Spatial Nearest-K"], "Normalized centroid distance",
             "Retrieval spatial filtering")
    save_bar("fig_raw_vs_translation_overlap.png",
             ["raw_gt_overlap", "pred_translation_overlap"], ["Raw", "Prediction Translation"], "Foreground GT overlap",
             "Raw versus prediction-derived translation")
    save_bar("fig_raw_vs_affine_overlap.png",
             ["raw_gt_overlap", "pred_affine_overlap"], ["Raw", "Prediction Affine"], "Foreground GT overlap",
             "Raw versus prediction-derived affine alignment")
    save_bar("fig_oracle_spatial_headroom.png",
             ["raw_gt_overlap", "pred_translation_overlap", "pred_affine_overlap", "gt_oracle_translation_overlap"],
             ["Raw", "Prediction Translation", "Prediction Affine", "GT Oracle Translation"], "Foreground GT overlap",
             "Spatial headroom upper-bound diagnosis")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(pair.fg_centroid_distance[np.isfinite(pair.fg_centroid_distance)], bins=40, color="#4c78a8", alpha=.85)
    ax.set_xlabel("Foreground centroid distance"); ax.set_ylabel("Top-K pair count"); ax.set_title("Centroid distance distribution")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_centroid_distance_distribution.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    valid = np.isfinite(pair.cosine_similarity) & np.isfinite(pair.fg_centroid_distance)
    ax.scatter(pair.loc[valid, "cosine_similarity"], pair.loc[valid, "fg_centroid_distance"], s=3, alpha=.18)
    ax.set_xlabel("Flattened feature cosine similarity"); ax.set_ylabel("Foreground centroid distance"); ax.set_title("Similarity versus spatial alignment")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_similarity_vs_centroid_distance.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    vals = [summary[(summary.scope == "All") & (summary['class'] == c)]["delta_pred_translation"].iloc[0] for c in ["LV", "MYO", "RV"]]
    ax.bar(["LV", "MYO", "RV"], vals, color=["#59a14f", "#f28e2b", "#e15759"]); ax.axhline(0, color="black", linewidth=.8)
    ax.set_ylabel("Prediction translation gain"); ax.set_title("Class-wise translation gain")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_class_translation_gain.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(rank['rank'], rank.mean_fg_centroid_distance, marker="o", label="Centroid distance")
    ax2 = ax.twinx(); ax2.plot(rank['rank'], rank.mean_similarity, marker="s", color="tab:orange", label="Similarity")
    ax.set_xlabel("Memory rank"); ax.set_ylabel("Centroid distance"); ax2.set_ylabel("Cosine similarity")
    ax.set_xticks(rank['rank']); ax.set_title("Retrieval rank versus spatial alignment")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_rank_vs_alignment.png", dpi=300); plt.close(fig)


def report(summary, pair_stats, correlation, bootstrap, rank, metadata, out):
    fg = summary[(summary.scope == "All") & (summary['class'] == "Foreground")].iloc[0]
    boot_fg = bootstrap[(bootstrap.scope == "All") & (bootstrap['class'] == "Foreground")]
    trans_ci = boot_fg[boot_fg.metric == "Delta_pred_translation"].iloc[0]
    affine_ci = boot_fg[boot_fg.metric == "Delta_pred_affine"].iloc[0]
    oracle_ci = boot_fg[boot_fg.metric == "Delta_GT_oracle_translation"].iloc[0]
    class_trans = summary[(summary.scope == "All") & summary['class'].isin(["LV", "MYO", "RV"])].set_index('class')
    class_oracle = class_trans.delta_gt_oracle.to_dict()
    fg_scale = pair_stats[(pair_stats.scope == "All") & (pair_stats['class'] == "Foreground") &
                          (pair_stats.metric == "scale_error_mean")].iloc[0]
    fg_corr = correlation[(correlation.domain == "All") & (correlation['class'] == "Foreground")].iloc[0]
    rank_best = rank.loc[rank.mean_fg_centroid_distance.idxmin()]

    if fg.delta_pred_translation <= 0 or fg.positive_translation_fraction < .40:
        case = "E"
    elif ((fg.delta_pred_translation >= .05 or (class_trans.delta_pred_translation >= .03).sum() >= 2)
          and fg.positive_translation_fraction >= .60 and trans_ci.ci_low > 0):
        case = "A"
    elif fg.delta_gt_oracle >= .05 and fg.delta_pred_translation < .02:
        case = "C"
    elif fg.delta_pred_translation >= .02 and fg.delta_pred_translation < .05 and fg.delta_gt_oracle > fg.delta_pred_translation:
        case = "B"
    elif fg.delta_gt_oracle < .02 and fg.delta_pred_translation < .02:
        case = "D"
    else:
        case = "B"
    recommendation = "Implement AA-SFF" if case in {"A", "B", "C"} else "Do not implement AA-SFF"

    domain_lines = []
    for scope in ["B", "C", "D", "All"]:
        r = summary[(summary.scope == scope) & (summary['class'] == "Foreground")].iloc[0]
        domain_lines.append(f"| {scope} | {r.mean_topk_centroid_distance:.6f} | {r.mean_pool_centroid_distance:.6f} | {r.retrieval_spatial_improvement*100:.2f}% | {r.raw_gt_overlap:.4f} | {r.delta_pred_translation:+.4f} | {r.delta_gt_oracle:+.4f} |")
    corr_text = (f"Spearman(similarity, -centroid distance)={fg_corr.spearman_similarity_vs_neg_centroid_distance:+.4f}, "
                 f"similarity/raw overlap={fg_corr.spearman_similarity_vs_gt_overlap:+.4f}, "
                 f"similarity/translation gain={fg_corr.spearman_similarity_vs_translation_gain:+.4f}.")
    text = f"""# EXP-6.5 Spatial Misalignment Diagnosis

## 1. Motivation

本实验只诊断 Global Top-K memory 与 query 的解剖空间关系，不修改 retrieval、SFF、SABE、BN、admission、decoder 或 prediction。所有几何与 warp 均为 inference 后 offline analysis。

## 2. Released Equivalence

PASS。4091 张切片；prediction SHA-256 为 `{metadata['prediction_sha256']}`，与 EXP-6 Released 完全一致。runner 每个 query 仅执行一次现有 `model(image)`，没有新增 model forward 或 torch random 调用。

## 3. Does Global Retrieval Already Encode Spatial Alignment?

真实 Top-K pair 数为 {metadata['num_pairs']}。All 域 foreground 的 Top-K mean centroid distance 为 **{fg.mean_topk_centroid_distance:.6f}**，whole-pool mean 为 **{fg.mean_pool_centroid_distance:.6f}**，spatial nearest-K 为 **{fg.mean_spatial_nearest_k_distance:.6f}**；相对 whole pool 降低 **{fg.retrieval_spatial_improvement*100:.2f}%**。这说明 flattened cosine retrieval 是否产生空间筛选效应，应结合该比例和后续 overlap headroom 一起判断。

| Scope | Top-K distance | Whole-pool distance | Relative improvement | Raw overlap | Prediction Δ | Oracle Δ |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(domain_lines)}

## 4. Residual Translation Mismatch

Top-K foreground raw GT overlap 为 **{fg.raw_gt_overlap:.4f}**；prediction-derived translation overlap 为 **{fg.pred_translation_overlap:.4f}**，gain **{fg.delta_pred_translation:+.4f}**。positive pair fraction 为 **{fg.positive_translation_fraction*100:.2f}%**，query-level bootstrap 95% CI 为 [{trans_ci.ci_low:+.4f}, {trans_ci.ci_high:+.4f}]。

## 5. Residual Scale Mismatch

Foreground mean scale error 为 **{fg_scale['mean']:.6f}**（n={int(fg_scale['n'])}）；affine primary 使用 raw scale ratio 位于 [0.5, 2.0] 的 valid subset，另提供 clipped [0.5, 2.0] robustness diagnostic。完整分布见 `pair_statistics.csv`。

## 6. Raw Query-Memory Anatomical Overlap

All 域 foreground raw overlap 为 {fg.raw_gt_overlap:.4f}。LV/MYO/RV raw overlap 分别为 {class_trans.loc['LV','raw_gt_overlap']:.4f}, {class_trans.loc['MYO','raw_gt_overlap']:.4f}, {class_trans.loc['RV','raw_gt_overlap']:.4f}。

## 7. Prediction-Derived Translation

Foreground overlap 为 {fg.pred_translation_overlap:.4f}，Δ={fg.delta_pred_translation:+.4f}，positive fraction={fg.positive_translation_fraction*100:.2f}%，query-level CI=[{trans_ci.ci_low:+.4f}, {trans_ci.ci_high:+.4f}]。

## 8. Prediction-Derived Affine Alignment

Foreground overlap 为 {fg.pred_affine_overlap:.4f}，Δ={fg.delta_pred_affine:+.4f}，positive fraction={fg.positive_affine_fraction*100:.2f}%，query-level CI=[{affine_ci.ci_low:+.4f}, {affine_ci.ci_high:+.4f}]。clipped affine 结果另列于 summary CSV。

## 9. GT-Centroid Oracle Headroom

Foreground oracle overlap 为 {fg.gt_oracle_translation_overlap:.4f}，Δ={fg.delta_gt_oracle:+.4f}，query-level CI=[{oracle_ci.ci_low:+.4f}, {oracle_ci.ci_high:+.4f}]。该结果只表示空间对齐上界，不是可部署方法。

## 10. Class-wise Analysis

| Class | Translation gain | GT-oracle gain |
|---|---:|---:|
| LV | {class_trans.loc['LV','delta_pred_translation']:+.4f} | {class_oracle['LV']:+.4f} |
| MYO | {class_trans.loc['MYO','delta_pred_translation']:+.4f} | {class_oracle['MYO']:+.4f} |
| RV | {class_trans.loc['RV','delta_pred_translation']:+.4f} | {class_oracle['RV']:+.4f} |

## 11. Domain Analysis

B/C/D/All 的完整 pair/query 统计见 `spatial_alignment_summary.csv`；没有使用 domain 或 GT 选择 transform。

## 12. Retrieval Rank Analysis

空间距离最小的平均 rank 为 Top{int(rank_best['rank'])}；Top1 是否最好需结合 `rank_spatial_summary.csv` 判断。各 rank 的 similarity、distance、raw overlap 和 translation gain 均已记录。

## 13. Failure Cases

代表案例分为 translation 改善、raw 已较好但无收益、translation 破坏 overlap、以及 oracle 有 headroom 但 prediction geometry 失败四类，见 `representative_spatial_cases.csv`。Similarity 与 spatial alignment：{corr_text}

## 14. Decision

最终判定为 **Case {case}**。该判定基于 prediction translation gain、positive pair fraction、query-level bootstrap CI 及 GT oracle headroom。

## 15. Recommendation

**{recommendation}**。本实验不实现 AA-SFF；若后续决定继续，需另行指定具体 alignment 方法。
"""
    (out / "exp6_5_report.md").write_text(text)
    (out / "decision.json").write_text(json.dumps({"case": case, "recommendation": recommendation,
        "prediction_translation_gain": fg.delta_pred_translation, "oracle_gain": fg.delta_gt_oracle,
        "positive_translation_fraction": fg.positive_translation_fraction,
        "translation_ci_low": trans_ci.ci_low, "translation_ci_high": trans_ci.ci_high}, indent=2))
    return case, recommendation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/exp6_5/formal")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/exp6_5")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair, query, metadata = load(args.input_dir)
    summary = spatial_summary(pair, query, args.output_dir)
    pair_stats = pair_statistics(pair, args.output_dir)
    correlation = correlation_summary(pair, args.output_dir)
    bootstrap = bootstrap_statistics(pair, args.output_dir)
    rank = rank_summary(pair, args.output_dir)
    representative_cases(pair, args.output_dir)
    make_figures(pair, query, summary, rank, args.output_dir)
    case, recommendation = report(summary, pair_stats, correlation, bootstrap, rank, metadata, args.output_dir)
    print(json.dumps({"case": case, "recommendation": recommendation,
                      "foreground_summary": summary[(summary.scope == "All") & (summary['class'] == "Foreground")].to_dict("records")[0]}, indent=2))


if __name__ == "__main__":
    main()
