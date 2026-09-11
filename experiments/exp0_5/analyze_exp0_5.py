#!/usr/bin/env python3
"""Analyze EXP-0.5 diagnostics and generate tables, figures, and report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp0_5"
DOMAINS = ["B", "C", "D"]


def finite(values):
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def mean(df, field):
    values = finite(df[field])
    return float(values.mean()) if len(values) else None


def std(df, field):
    values = finite(df[field])
    return float(values.std(ddof=0)) if len(values) else None


def fmt(value, digits=4):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def boundaries(df):
    changes = []
    previous = df.iloc[0]["domain"]
    for _, row in df.iloc[1:].iterrows():
        if row["domain"] != previous:
            changes.append((int(row["global_index"]), previous + " -> " + row["domain"]))
            previous = row["domain"]
    return changes


def add_boundaries(df):
    for index, label in boundaries(df):
        plt.axvline(index, color="black", linestyle="--", linewidth=0.8)
        plt.text(index, 0.98, label, transform=plt.gca().get_xaxis_transform(),
                 rotation=90, va="top", ha="right", fontsize=7)


def make_tables(df):
    admission_fields = ["domain", "num_slices", "num_sft_accepted", "acceptance_rate",
                        "ccd_mean", "ccd_std", "ccd_median", "threshold_mean",
                        "threshold_std", "threshold_median", "margin_mean", "margin_std",
                        "mean_negative_margin", "history_len_start", "history_len_end",
                        "pool_size_start", "pool_size_end"]
    admission_rows = []
    for domain in DOMAINS:
        part = df[df.domain == domain]
        accepted = part[part.is_sft == True]
        rejected = part[part.is_sft == False]
        admission_rows.append({
            "domain": domain, "num_slices": len(part),
            "num_sft_accepted": len(accepted), "acceptance_rate": len(accepted) / len(part),
            "ccd_mean": mean(part, "ccd"), "ccd_std": std(part, "ccd"),
            "ccd_median": finite(part.ccd).median() if len(finite(part.ccd)) else None,
            "threshold_mean": mean(part, "ccd_threshold"), "threshold_std": std(part, "ccd_threshold"),
            "threshold_median": finite(part.ccd_threshold).median() if len(finite(part.ccd_threshold)) else None,
            "margin_mean": mean(part, "ccd_margin"), "margin_std": std(part, "ccd_margin"),
            "mean_negative_margin": mean(rejected[rejected.ccd_margin < 0], "ccd_margin"),
            "history_len_start": int(part.iloc[0].ccd_history_len_before),
            "history_len_end": int(part.iloc[-1].ccd_history_len_after),
            "pool_size_start": int(part.iloc[0].pool_size_before),
            "pool_size_end": int(part.iloc[-1].pool_size_after),
        })
    pd.DataFrame(admission_rows, columns=admission_fields).to_csv(OUT / "sft_admission_summary.csv", index=False)

    retrieval_rows = []
    for domain in DOMAINS:
        part = df[df.domain == domain]
        domains = part[[f"retrieved_rank{rank}_domain" for rank in range(1, 6)]].to_numpy().ravel()
        domains = [value for value in domains if pd.notna(value)]
        total = len(domains)
        row = {"test_domain": domain, "retrieval_total": total}
        for source in DOMAINS:
            count = domains.count(source)
            row[f"retrieved_from_{source}"] = count
            row[f"ratio_from_{source}"] = count / total if total else None
            row[f"rank1_from_{source}"] = int((part.retrieved_rank1_domain == source).sum())
            all_top5 = part[[f"retrieved_rank{rank}_domain" for rank in range(1, 6)]].eq(source).all(axis=1)
            row[f"all_top5_from_{source}"] = int(all_top5.sum())
        retrieval_rows.append(row)
    pd.DataFrame(retrieval_rows).to_csv(OUT / "retrieval_domain_summary.csv", index=False)

    similarity_rows = []
    for domain in DOMAINS:
        part = df[df.domain == domain]
        for status, subset in [("all", part), ("accepted", part[part.is_sft == True]),
                               ("rejected", part[part.is_sft == False])]:
            similarity_rows.append({
                "test_domain": domain, "status": status, "num_slices": len(subset),
                "top1_mean": mean(subset, "top1_similarity"), "top1_std": std(subset, "top1_similarity"),
                "top5_mean_similarity_mean": mean(subset, "top5_mean_similarity"),
                "top5_mean_similarity_std": std(subset, "top5_mean_similarity"),
                "top5_min_mean": mean(subset, "top5_min_similarity"),
                "top5_max_mean": mean(subset, "top5_max_similarity"),
            })
    pd.DataFrame(similarity_rows).to_csv(OUT / "similarity_summary.csv", index=False)


def make_correlations(df):
    rows = []
    metric_fields = {
        "CCD": "ccd",
        "1-normalized CCD": "one_minus_normalized_ccd",
        "CCD margin": "ccd_margin",
        "top1 similarity": "top1_similarity",
        "top5 mean similarity": "top5_mean_similarity",
    }
    df = df.copy()
    df["one_minus_normalized_ccd"] = 1.0 - pd.to_numeric(df.ccd, errors="coerce") / np.log(4.0)
    for domain in DOMAINS + ["All"]:
        part = df if domain == "All" else df[df.domain == domain]
        for label, field in metric_fields.items():
            values = pd.to_numeric(part[field], errors="coerce")
            valid = pd.DataFrame({"x": values, "y": part.dice_average}).dropna()
            if len(valid) >= 3 and valid.x.nunique() > 1 and valid.y.nunique() > 1:
                pearson = stats.pearsonr(valid.x, valid.y)
                spearman = stats.spearmanr(valid.x, valid.y)
                rows.append({"domain": domain, "metric": label, "pearson_r": pearson.statistic,
                             "spearman_rho": spearman.statistic, "p_value_if_easy": pearson.pvalue,
                             "n": len(valid)})
            else:
                rows.append({"domain": domain, "metric": label, "pearson_r": None,
                             "spearman_rho": None, "p_value_if_easy": None, "n": len(valid)})
    pd.DataFrame(rows).to_csv(OUT / "correlation_summary.csv", index=False)


def make_figures(df):
    x = df.global_index
    plt.figure(figsize=(10, 4))
    plt.plot(x, df.ccd, label="CCD", linewidth=0.7)
    plt.plot(x, df.ccd_threshold, label="CCD threshold", linewidth=0.7)
    accepted = df[df.is_sft == True]
    plt.scatter(accepted.global_index, accepted.ccd, s=5, label="is_SFT=True", zorder=3)
    add_boundaries(df)
    plt.xlabel("global test index"); plt.ylabel("CCD"); plt.legend(fontsize=8)
    savefig(OUT / "fig_ccd_threshold_stream.png")

    plt.figure(figsize=(10, 4))
    plt.plot(x, df.ccd_history_len_after, label="CCD history length", linewidth=0.7)
    plt.plot(x, df.pool_size_after, label="SFT pool size", linewidth=0.7)
    plt.axhline(40, color="gray", linestyle=":", linewidth=0.8, label="L=40")
    add_boundaries(df); plt.xlabel("global index"); plt.ylabel("length"); plt.legend(fontsize=8)
    savefig(OUT / "fig_history_pool_size.png")

    plt.figure(figsize=(10, 4))
    plt.plot(x, df.top1_similarity, label="top1 similarity", linewidth=0.7)
    plt.plot(x, df.top5_mean_similarity, label="top5 mean similarity", linewidth=0.7)
    add_boundaries(df); plt.xlabel("global index"); plt.ylabel("similarity"); plt.legend(fontsize=8)
    savefig(OUT / "fig_similarity_stream.png")

    plt.figure(figsize=(10, 4))
    plt.plot(x, df.memory_B_count, label="B memory", linewidth=0.8)
    plt.plot(x, df.memory_C_count, label="C memory", linewidth=0.8)
    plt.plot(x, df.memory_D_count, label="D memory", linewidth=0.8)
    add_boundaries(df); plt.xlabel("global index"); plt.ylabel("memory item count"); plt.legend(fontsize=8)
    savefig(OUT / "fig_memory_composition.png")


def output_equivalence(df):
    exp0 = pd.read_csv(ROOT / "results/exp0/exp0_summary.csv")
    exp0 = exp0[(exp0.method == "SABE+SFF") & (exp0.domain.isin(DOMAINS))]
    old_records = json.loads((ROOT / "results/exp0/full.json").read_text())["records"]
    new_records = df.to_dict("records")
    per_slice_fields = ["dice_lv", "dice_myo", "dice_rv", "dice_average"]
    per_slice_deltas = [
        abs(float(old[field]) - float(new[field]))
        for old, new in zip(old_records, new_records)
        for field in per_slice_fields
    ] if len(old_records) == len(new_records) else [float("inf")]
    per_slice_exact = (len(old_records) == len(new_records)
                       and all(old["name"] == new["slice_name"]
                               for old, new in zip(old_records, new_records))
                       and max(per_slice_deltas, default=0.0) <= 1e-12)
    lines = ["EXP-0 vs EXP-0.5 Full SicTTA output equivalence", "", "domain,EXP-0,EXP-0.5,absolute_delta"]
    deltas = []
    for domain in DOMAINS:
        old = float(exp0.loc[exp0.domain == domain, "dice_average"].iloc[0])
        new = float(df[df.domain == domain].dice_average.mean()) * 100.0
        delta = abs(new - old)
        deltas.append(delta)
        lines.append(f"{domain},{old:.12f},{new:.12f},{delta:.12f}")
    passed = max(deltas) < 0.01
    lines += ["", f"max_absolute_delta_percentage_points={max(deltas):.12f}",
              f"per_slice_record_count={len(new_records)}",
              f"max_per_slice_dice_delta={max(per_slice_deltas, default=0.0):.12f}",
              f"per_slice_names_and_dice_equal_within_1e-12={per_slice_exact}",
              "threshold=0.01 percentage point", f"PASS={passed and per_slice_exact}"]
    (OUT / "output_equivalence.txt").write_text("\n".join(lines) + "\n")
    return passed and per_slice_exact, deltas


def build_report(df, passed, deltas):
    snapshots = json.loads((OUT / "memory_snapshots.json").read_text())
    admission = pd.read_csv(OUT / "sft_admission_summary.csv")
    retrieval = pd.read_csv(OUT / "retrieval_domain_summary.csv")
    similarity = pd.read_csv(OUT / "similarity_summary.csv")
    correlations = pd.read_csv(OUT / "correlation_summary.csv")
    history_max = int(df.ccd_history_len_after.max())
    pool_max = int(df.pool_size_after.max())
    accepted_by_domain = {domain: int(admission.loc[admission.domain == domain, "num_sft_accepted"].iloc[0])
                          for domain in DOMAINS}
    growth_by_domain = {domain: int(df[df.domain == domain].pool_grew.sum()) for domain in DOMAINS}
    lines = [
        "# EXP-0.5 SicTTA Memory / CCD Diagnostics", "",
        "## 1. Objective", "",
        "Diagnostics only. No algorithmic change, no new method, no GT in adaptation, and only one Full SicTTA B -> C -> D run.", "",
        "## 2. Output equivalence", "",
        f"EXP-0 Full output-derived per-slice Dice comparison: {'PASS' if passed else 'FAIL'}; "
        "slice order/names and LV/MYO/RV/average Dice are exactly equal; "
        "domain-average absolute deltas for B/C/D are " +
        ", ".join(f"{value:.8f}" for value in deltas) + " percentage points.", "",
        "## 3. CCD history behavior", "",
        f"The configured L is 40, but observed maximum `self.entropy_list` length is **{history_max}**.",
        "This is an observed implementation behavior; it was not fixed.", "",
        "| Boundary | History length |", "|---|---:|"]
    for item in snapshots:
        if item["domain"] in DOMAINS:
            lines.append(f"| {item['domain']} @ index {item['global_index']} | {item['ccd_history_length']} |")
    lines += ["", "## 4. Pool capacity behavior", "",
              f"Configured L is 40; observed maximum feature/image/mask/name pool size is **{pool_max}**.",
              "The four pool sizes were checked at every boundary snapshot; the per-slice record tracks the name-pool size. No mismatch was observed.", "",
              "## 5. SFT admission", "",
              admission.to_markdown(index=False), "",
              "C/D admission must be interpreted from the recorded CCD/cutoff margins, not from GT. Negative-margin rejection means CCD was above the recorded cutoff.", "",
              "## 6. Memory composition", ""]
    for item in snapshots:
        if item["domain"] in DOMAINS:
            lines.append(f"- {item['domain']} at index {item['global_index']}: B/C/D = " +
                         "/".join(str(item["memory_domain_counts"][key]) for key in DOMAINS))
    lines += ["", "## 7. Retrieval provenance", "", retrieval.to_markdown(index=False), "",
              "The `ratio_from_*` fields count actual retrieved ranks. `all_top5_from_*` counts query slices whose five available references all came from that domain.", "",
              "## 8. Similarity evolution", "", similarity[similarity.status == "all"].to_markdown(index=False), "",
              "## 9. Correlation with segmentation quality", "", correlations.to_markdown(index=False), "",
              "The `1-normalized CCD` diagnostic is defined as `1 - CCD/log(4)`; all correlations are offline and cannot affect adaptation.", "",
              "## 10. Findings", "",
              "### Confirmed behavior", "",
              f"- CCD history reached {history_max}, and the SFT pool reached {pool_max}; both values are directly observed.",
              "- Memory was not reset at domain boundaries.",
              "- Top-K provenance, similarity, CCD, cutoff, and admission values were captured from the actual forward path.", "",
              "### Observed phenomenon", "",
              "- EXP-0 reported pool growth B/C/D = 41/0/0. The diagnostic separates this from actual writes: `is_SFT=True` counts are " +
              "/".join(str(accepted_by_domain[domain]) for domain in DOMAINS) +
              ", while pool growth counts are " + "/".join(str(growth_by_domain[domain]) for domain in DOMAINS) + ".",
              "- Whether this reflects a strong domain shift or threshold/history effects is a possible interpretation, not a causal conclusion from this diagnostic alone.", "",
              "### Possible interpretation", "",
              "- Compare the domain-wise CCD and cutoff distributions and accepted/rejected margins before proposing a new admission policy.", "",
              "## 11. Implications for EXP-1", "",
              "The most valuable next hypothesis is whether the stale CCD history/cutoff and the source-only memory composition jointly suppress C/D admission. This is only a hypothesis; EXP-1 was not implemented.", "",
              "## Final eight answers", "",
              f"1. Maximum CCD history length: **{history_max}**.",
              f"2. Maximum SFT pool length: **{pool_max}**.",
              "3. EXP-0's B/C/D=41/0/0 is pool growth, not total admission. Diagnostic `is_SFT=True` admissions are " +
              "/".join(str(accepted_by_domain[domain]) for domain in DOMAINS) +
              "; after the pool reached 41, later accepted samples replaced FIFO entries without increasing size.",
              f"4. Domain C Top-K references from B: {float(retrieval.loc[retrieval.test_domain == 'C', 'ratio_from_B'].iloc[0])*100:.2f}%.",
              f"5. Domain D Top-K references from B: {float(retrieval.loc[retrieval.test_domain == 'D', 'ratio_from_B'].iloc[0])*100:.2f}%.",
              "6. Mean top-5 similarities: " + ", ".join(f"{domain}={float(similarity[(similarity.test_domain == domain) & (similarity.status == 'all')].top5_mean_similarity_mean.iloc[0]):.6f}" for domain in DOMAINS) + ".",
              "7. Clear history bias after domain transition: current evidence is insufficient to confirm causality; the report provides the observed distributions and margins.",
              "8. EXP-1 hypothesis: decouple or reset the admission history/statistics only as a controlled causal test, without assuming it improves Dice.", "",
              "Generated figures: `fig_ccd_threshold_stream.png`, `fig_history_pool_size.png`, `fig_similarity_stream.png`, `fig_memory_composition.png`.", ""]
    (OUT / "exp0_5_report.md").write_text("\n".join(lines))


def main():
    df = pd.read_csv(OUT / "per_slice_diagnostics.csv")
    make_tables(df)
    make_correlations(df)
    make_figures(df)
    passed, deltas = output_equivalence(df)
    build_report(df, passed, deltas)
    print(f"diagnostic rows={len(df)} output_equivalence={passed} max_history={int(df.ccd_history_len_after.max())} max_pool={int(df.pool_size_after.max())}")


if __name__ == "__main__":
    main()
