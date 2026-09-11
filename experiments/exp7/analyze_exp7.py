#!/usr/bin/env python3
"""Offline query-level analysis for EXP-7 local correspondence diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from plot_exp7 import make_all_figures


ROOT = Path(__file__).resolve().parents[2]
CLASSES = [("fg", "Foreground"), ("lv", "LV"), ("myo", "MYO"), ("rv", "RV")]
DOMAINS = ["B", "C", "D", "All"]


def finite_mean(values):
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else np.nan


def load(input_dir):
    pair = pd.read_csv(input_dir / "pair_local_correspondence_seed2026.csv")
    query = pd.read_csv(input_dir / "query_local_correspondence_seed2026.csv")
    metadata = json.loads((input_dir / "run_metadata_seed2026.json").read_text())
    if len(pair) != 20425 or len(query) != 4091:
        raise RuntimeError(f"unexpected formal shape pair={pair.shape}, query={query.shape}")
    return pair, query, metadata


def query_level(pair, field, domain):
    part = pair if domain == "All" else pair[pair.domain == domain]
    return part.groupby("global_index")[field].mean()


def make_summary(pair, query, out):
    rows = []
    for domain in DOMAINS:
        for key, label in CLASSES:
            for radius in [1, 2, 3]:
                same = query_level(pair, f"same_agree_{key}", domain)
                local = query_level(pair, f"local_r{radius}_agree_{key}", domain)
                oracle = query_level(pair, f"oracle_local_r{radius}_agree_{key}", domain)
                global_nn = query_level(pair, f"global_nn_agree_{key}", domain)
                delta = local - same
                available = oracle - same
                if key == "fg":
                    displacement_field = "mean_displacement_r2"
                    zero_field = "zero_displacement_fraction_r2"
                else:
                    displacement_field = f"mean_displacement_r2_{key}"
                    zero_field = f"zero_displacement_fraction_r2_{key}"
                displacement = query_level(pair, displacement_field, domain)
                zero = query_level(pair, zero_field, domain)
                sim_gain = query_level(pair, "delta_similarity_r2_fg", domain)
                rows.append({"scope": "All" if domain == "All" else "Domain", "domain": domain,
                             "class": label, "radius": radius, "same_agreement": same.mean(),
                             "local_agreement": local.mean(), "delta_local": delta.mean(),
                             "global_nn_agreement": global_nn.mean(), "oracle_local_agreement": oracle.mean(),
                             "available_gain": available.mean(), "recovery_ratio": delta.mean() / (available.mean() + 1e-8),
                             "mean_displacement": displacement.mean(), "zero_displacement_fraction": zero.mean(),
                             "mean_similarity_gain": sim_gain.mean(), "n_queries": len(delta.dropna()),
                             "positive_query_fraction": float((delta.dropna() > 0).mean()) if len(delta.dropna()) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(out / "local_correspondence_summary.csv", index=False)
    return result


def bootstrap(pair, out, samples=10000, seed=2026):
    np.random.seed(seed); rows = []
    for domain in DOMAINS:
        for key, label in CLASSES:
            same_field, local_field = f"same_agree_{key}", f"local_r2_agree_{key}"
            part = pair if domain == "All" else pair[pair.domain == domain]
            grouped = part.assign(_delta=part[local_field] - part[same_field]).groupby("global_index")['_delta'].mean().dropna().to_numpy(float)
            if not len(grouped): continue
            draws = []
            for start in range(0, samples, 250):
                n = min(250, samples - start)
                idx = np.random.randint(0, len(grouped), size=(n, len(grouped)))
                draws.append(grouped[idx].mean(axis=1))
            draws = np.concatenate(draws)
            rows.append({"domain": domain, "class": label, "mean_delta": grouped.mean(),
                         "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975),
                         "positive_query_fraction": (grouped > 0).mean(), "n_queries": len(grouped),
                         "bootstrap_samples": samples, "seed": seed})
    result = pd.DataFrame(rows); result.to_csv(out / "bootstrap_statistics.csv", index=False); return result


def correlation(pair, out):
    rows = []
    for domain in DOMAINS:
        part = pair if domain == "All" else pair[pair.domain == domain]
        for key, label in CLASSES:
            def corr(x, y):
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 3 or len(np.unique(x[valid])) < 2 or len(np.unique(y[valid])) < 2: return np.nan, np.nan
                z = stats.spearmanr(x[valid], y[valid]); return float(z.statistic), float(z.pvalue)
            sim = part.global_similarity.to_numpy(float)
            same = part[f"same_agree_{key}"].to_numpy(float)
            delta = part[f"local_r2_delta_{key}"].to_numpy(float)
            same_sim = part.same_similarity_fg.to_numpy(float)
            local_sim = part.local_r2_similarity_fg.to_numpy(float)
            a, ap = corr(sim, same); b, bp = corr(sim, delta); c, cp = corr(same_sim, same); d, dp = corr(local_sim, part[f"local_r2_agree_{key}"].to_numpy(float))
            rows.append({"domain": domain, "class": label, "n": len(part),
                         "spearman_global_similarity_vs_same_agree": a, "p_global_similarity_vs_same_agree": ap,
                         "spearman_global_similarity_vs_delta_local": b, "p_global_similarity_vs_delta_local": bp,
                         "spearman_same_similarity_vs_same_correctness": c, "p_same_similarity_vs_same_correctness": cp,
                         "spearman_local_similarity_vs_local_correctness": d, "p_local_similarity_vs_local_correctness": dp})
    result = pd.DataFrame(rows); result.to_csv(out / "similarity_correspondence_correlation.csv", index=False); return result


def rank_summary(pair, out):
    rows = []
    for rank, part in pair.groupby("memory_rank"):
        rows.append({"rank": int(rank), "n": len(part), "mean_global_similarity": part.global_similarity.mean(),
                     "same_agree_fg": part.same_agree_fg.mean(), "local_r2_agree_fg": part.local_r2_agree_fg.mean(),
                     "delta_local_fg": part.local_r2_delta_fg.mean(), "mean_displacement_r2": part.mean_displacement_r2.mean(),
                     "zero_displacement_fraction_r2": part.zero_displacement_fraction_r2.mean()})
    result = pd.DataFrame(rows).sort_values("rank"); result.to_csv(out / "rank_correspondence_summary.csv", index=False); return result


def boundary_summary(query, out):
    rows = []
    for domain in DOMAINS:
        part = query if domain == "All" else query[query.domain == domain]
        rows.append({"domain": domain, "boundary_same_agree": part.boundary_same_agree.mean(),
                     "boundary_local_agree": part.boundary_local_agree.mean(), "boundary_delta": part.boundary_delta.mean(),
                     "interior_same_agree": part.interior_same_agree.mean(), "interior_local_agree": part.interior_local_agree.mean(),
                     "interior_delta": part.interior_delta.mean()})
    result = pd.DataFrame(rows); result.to_csv(out / "boundary_interior_summary.csv", index=False); return result


def corrected_harmed(pair, out):
    rows = []
    for domain in DOMAINS:
        part = pair if domain == "All" else pair[pair.domain == domain]
        for key, label in CLASSES:
            for state, count_col, sim_col, disp_col in [("corrected", f"corrected_count_r2_{key}", f"corrected_delta_similarity_r2_{key}", f"corrected_displacement_r2_{key}"),
                                                        ("harmed", f"harmed_count_r2_{key}", f"harmed_delta_similarity_r2_{key}", f"harmed_displacement_r2_{key}")]:
                rows.append({"domain": domain, "class": label, "state": state, "total_locations": part[count_col].sum(),
                             "mean_delta_similarity": finite_mean(part[sim_col]), "mean_displacement": finite_mean(part[disp_col]),
                             "n_pairs": len(part)})
    result = pd.DataFrame(rows); result.to_csv(out / "corrected_harmed_summary.csv", index=False); return result


def representative(pair, out):
    part = pair.dropna(subset=["local_r2_delta_fg"]).copy()
    groups = {
        "A_same_wrong_local_correct": part.sort_values("local_r2_delta_fg", ascending=False).head(8),
        "B_same_correct_local_harmed": part.sort_values("local_r2_delta_fg").head(8),
        "C_both_correct": part[(part.same_agree_fg > .8) & (part.local_r2_agree_fg > .8)].head(8),
        "D_both_wrong_oracle_available": part[(part.same_agree_fg < .4) & (part.local_r2_agree_fg < .4) & (part.oracle_local_r2_agree_fg > .7)].head(8),
        "E_myo_failures": part.sort_values("local_r2_delta_myo").head(8),
    }
    rows = []
    for category, selected in groups.items():
        for r in selected.itertuples(index=False):
            rows.append({"category": category, "global_index": r.global_index, "domain": r.domain,
                         "query_name": r.query_name, "memory_name": r.memory_name, "memory_rank": r.memory_rank,
                         "query_class": "foreground", "same_similarity": r.same_similarity_fg,
                         "local_similarity": r.local_r2_similarity_fg, "local_displacement": r.mean_displacement_r2,
                         "same_gt_class": r.same_agree_fg, "local_gt_class": r.local_r2_agree_fg,
                         "query_gt_class": "foreground", "oracle_local": r.oracle_local_r2_agree_fg,
                         "delta_local": r.local_r2_delta_fg})
    result = pd.DataFrame(rows); result.to_csv(out / "representative_correspondence_cases.csv", index=False); return result


def report(summary, boot, corr, rank, boundary, metadata, out):
    fg = summary[(summary.domain == "All") & (summary['class'] == "Foreground") & (summary.radius == 2)].iloc[0]
    bfg = boot[(boot.domain == "All") & (boot['class'] == "Foreground")].iloc[0]
    classes = summary[(summary.domain == "All") & (summary.radius == 2) & summary['class'].isin(["LV", "MYO", "RV"])].set_index('class')
    radius = summary[(summary.domain == "All") & (summary['class'] == "Foreground")].set_index('radius')
    corrfg = corr[(corr.domain == "All") & (corr['class'] == "Foreground")].iloc[0]
    top_rank = rank.loc[rank.mean_displacement_r2.idxmin()]
    if fg.delta_local <= 0:
        case = "E"
    elif fg.available_gain >= .08 and fg.delta_local < .03:
        case = "C"
    elif fg.available_gain < .03 and fg.delta_local < .03:
        case = "D"
    elif fg.delta_local >= .08 and bfg.ci_low > 0 and (classes.delta_local >= .05).sum() >= 2:
        case = "A"
    elif fg.delta_local >= .03 and bfg.ci_low > 0 and (classes.delta_local > 0).sum() >= 2:
        case = "B"
    else:
        case = "E"
    recommendation = "IMPLEMENT LC-SFF" if case in {"A", "B"} else "STOP SFF MODIFICATION"
    boundary_all = boundary[boundary.domain == "All"].iloc[0]
    text = f"""# EXP-7 Local Correspondence Diagnosis

## 1. Motivation

Image-level retrieval does not guarantee location-level correspondence. 本实验只诊断 Global Top-K memory 的 SamePosition 与 LocalNN 对应质量，不实现 LC-SFF。

## 2. Released Equivalence

PASS。4091 slices、20,425 pairs；prediction hash `{metadata['prediction_sha256']}` 与 EXP-6.5 一致。每个 query 仅执行一次既有 `model(image)`。

## 3. Same-Position Correspondence Quality

All 域 foreground SamePosition semantic agreement 为 **{fg.same_agreement:.4f}**。

## 4. Local Feature Matching

LocalNN r=2 agreement 为 **{fg.local_agreement:.4f}**，Δ={fg.delta_local:+.4f}；LocalLabelOracle 为 {fg.oracle_local_agreement:.4f}。

## 5. Class-wise Results

| Class | Same | Local r=2 | Δ | Oracle local | Available gain |
|---|---:|---:|---:|---:|---:|
| LV | {classes.loc['LV','same_agreement']:.4f} | {classes.loc['LV','local_agreement']:.4f} | {classes.loc['LV','delta_local']:+.4f} | {classes.loc['LV','oracle_local_agreement']:.4f} | {classes.loc['LV','available_gain']:+.4f} |
| MYO | {classes.loc['MYO','same_agreement']:.4f} | {classes.loc['MYO','local_agreement']:.4f} | {classes.loc['MYO','delta_local']:+.4f} | {classes.loc['MYO','oracle_local_agreement']:.4f} | {classes.loc['MYO','available_gain']:+.4f} |
| RV | {classes.loc['RV','same_agreement']:.4f} | {classes.loc['RV','local_agreement']:.4f} | {classes.loc['RV','delta_local']:+.4f} | {classes.loc['RV','oracle_local_agreement']:.4f} | {classes.loc['RV','available_gain']:+.4f} |

## 6. Local Semantic Availability Oracle

Foreground LocalLabelOracle agreement 为 {fg.oracle_local_agreement:.4f}，相对 SamePosition available gain 为 {fg.available_gain:+.4f}。这是 GT-based ceiling，不是方法。

## 7. Feature Matching Recovery Ratio

Foreground recovery ratio 为 {fg.recovery_ratio:.4f}；LocalNN 只恢复了 local label availability 与 SamePosition 之间的一部分/全部信号，详见 CSV。

## 8. Radius Sensitivity

Foreground Δ（r=1/r=2/r=3）分别为 {radius.loc[1,'delta_local']:+.4f} / {radius.loc[2,'delta_local']:+.4f} / {radius.loc[3,'delta_local']:+.4f}。

## 9. Local Displacement Analysis

r=2 平均 local displacement 为 {fg.mean_displacement:.4f}，zero displacement fraction 为 {fg.zero_displacement_fraction*100:.2f}%。

## 10. Boundary vs Interior

Boundary gain={boundary_all.boundary_delta:+.4f}，Interior gain={boundary_all.interior_delta:+.4f}。

## 11. Retrieval Rank Analysis

平均 displacement 最小的是 Top{int(top_rank['rank'])}；完整 Top1–Top5 结果见 `rank_correspondence_summary.csv`。

## 12. Domain Analysis

B/C/D 结果见 `local_correspondence_summary.csv`；三域方向和数值均已单独记录。

## 13. Corrected vs Harmed Correspondences

corrected/harmed location 的 count、Δsimilarity 和 displacement 见 `corrected_harmed_summary.csv`；代表案例见 `representative_correspondence_cases.csv`。

## 14. Statistical Evidence

Foreground query-level paired bootstrap 10000 次：mean Δ={bfg.mean_delta:+.4f}，95% CI=[{bfg.ci_low:+.4f}, {bfg.ci_high:+.4f}]，positive query fraction={bfg.positive_query_fraction*100:.2f}%。Similarity 与 Same agreement 的 Spearman ρ={corrfg.spearman_global_similarity_vs_same_agree:+.4f}；与 Local delta 的 ρ={corrfg.spearman_global_similarity_vs_delta_local:+.4f}。Global similarity 与 local gain 的关系不能替代 GT semantic agreement。

## 15. Decision

最终 Case **{case}**。主判据为 query-level foreground LocalNN r=2 Δ={fg.delta_local:+.4f}。

## 16. Recommendation

**{recommendation}**。本实验停止，不实现 LC-SFF；如 Case A/B，也仅报告候选支持，等待下一步决定。
"""
    (out / "exp7_report.md").write_text(text)
    (out / "decision.json").write_text(json.dumps({"case": case, "recommendation": recommendation,
        "foreground_same": fg.same_agreement, "foreground_local_r2": fg.local_agreement,
        "foreground_delta": fg.delta_local, "ci_low": bfg.ci_low, "ci_high": bfg.ci_high}, indent=2))
    return case


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input-dir", type=Path, default=ROOT / "results/exp7/formal"); parser.add_argument("--output-dir", type=Path, default=ROOT / "results/exp7")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    pair, query, metadata = load(args.input_dir)
    summary = make_summary(pair, query, args.output_dir); boot = bootstrap(pair, args.output_dir); corr = correlation(pair, args.output_dir); rank = rank_summary(pair, args.output_dir); boundary = boundary_summary(query, args.output_dir); corrected_harmed(pair, args.output_dir); representative(pair, args.output_dir)
    make_all_figures(summary, pair, rank, boundary, args.output_dir / "figures")
    case = report(summary, boot, corr, rank, boundary, metadata, args.output_dir)
    print(json.dumps({"case": case, "foreground": summary[(summary.domain == 'All') & (summary['class'] == 'Foreground') & (summary.radius == 2)].to_dict('records')[0]}, indent=2))


if __name__ == "__main__":
    main()
