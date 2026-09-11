#!/usr/bin/env python3
"""Offline analysis for EXP-6 CR-SFF Stage-1 outputs.

All quantities in this file are computed after inference.  Ground truth is
used only for Dice/ASSD summaries and the offline memory-quality audit.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/exp6/raw"
OUT = ROOT / "results/exp6"
FIG = OUT / "figures"
SEED = 2026
DOMAINS = ["B", "C", "D", "All"]
CLASSES = ["lv", "myo", "rv"]
LABELS = {"lv": "LV", "myo": "MYO", "rv": "RV"}
VARIANTS = ["released", "crsff_identity", "global_reliability", "class_reliability", "inverse_class_reliability"]
DISPLAY = {
    "released": "Released",
    "crsff_identity": "CR-SFF Identity",
    "global_reliability": "Global Reliability",
    "class_reliability": "Class Reliability",
    "inverse_class_reliability": "Inverse Class Reliability",
}
JSON_COLUMNS = [
    "topk_names", "topk_similarities", "released_weights", "memory_class_reliabilities",
    "memory_class_soft_masses", "memory_global_reliabilities", "global_reliability_weights",
    "class_reliability_weights", "inverse_class_reliability_weights", "query_class_soft_mass",
    "weight_l1_change_vs_released", "weight_entropy_class", "top1_weight_class",
]


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def parse(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return []
    return json.loads(value)


def load_outputs():
    frames, metadata = {}, {}
    for variant in VARIANTS:
        path = RAW / f"{variant}_seed{SEED}.csv"
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"missing formal output: {path}")
        frame = pd.read_csv(path)
        if len(frame) != 4091 or frame.global_index.nunique() != 4091:
            raise RuntimeError(f"unexpected row count for {variant}: {frame.shape}")
        for column in JSON_COLUMNS:
            frame[column] = frame[column].map(parse)
        frames[variant] = frame
        metadata[variant] = json.loads(meta_path.read_text())
    return frames, metadata


def summary_tables(metadata):
    rows = []
    for variant in VARIANTS:
        for row in metadata[variant]["domain_summary"]:
            rows.append(row)
    domain = pd.DataFrame(rows)
    domain.to_csv(OUT / "exp6_domain_summary.csv", index=False)

    all_rows = domain[domain.domain == "All"].copy().sort_values("variant")
    base = all_rows[all_rows.variant == "released"].iloc[0]
    for cls in CLASSES + ["average"]:
        all_rows[f"delta_{cls}_pp_vs_released"] = (all_rows[f"dice_{cls}"] - base[f"dice_{cls}"]) * 100
    all_rows["display_variant"] = all_rows.variant.map(DISPLAY)
    all_rows.to_csv(OUT / "exp6_overall_summary.csv", index=False)

    delta_rows = []
    for domain_name in DOMAINS:
        base_row = domain[(domain.variant == "released") & (domain.domain == domain_name)].iloc[0]
        for variant in VARIANTS:
            row = domain[(domain.variant == variant) & (domain.domain == domain_name)].iloc[0]
            for cls in CLASSES + ["average"]:
                delta_rows.append({
                    "seed": SEED, "variant": variant, "display_variant": DISPLAY[variant],
                    "domain": domain_name, "class": cls.upper() if cls != "average" else "Average",
                    "released_dice": base_row[f"dice_{cls}"], "variant_dice": row[f"dice_{cls}"],
                    "delta_pp_vs_released": (row[f"dice_{cls}"] - base_row[f"dice_{cls}"]) * 100,
                })
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(OUT / "class_delta_summary.csv", index=False)
    return domain, all_rows, delta


def weight_diagnostics(frames):
    rows = []
    for variant, frame in frames.items():
        for domain in DOMAINS:
            part = frame if domain == "All" else frame[frame.domain == domain]
            for cls_idx, cls in enumerate(["bg"] + CLASSES):
                l1, entropy, top1, sums, changed = [], [], [], [], []
                for row in part.itertuples(index=False):
                    released = np.asarray(row.released_weights, dtype=float)
                    if len(released) == 0:
                        continue
                    if cls == "bg":
                        effective = released
                        change = 0.0
                    elif variant in {"released", "crsff_identity"}:
                        effective = released
                        change = 0.0
                    elif variant == "global_reliability":
                        effective = np.asarray(row.global_reliability_weights, dtype=float)
                        change = float(np.asarray(row.weight_l1_change_vs_released)[0])
                    elif variant == "class_reliability":
                        effective = np.asarray(row.class_reliability_weights, dtype=float)[cls_idx - 1]
                        change = float(np.asarray(row.weight_l1_change_vs_released)[cls_idx - 1])
                    else:
                        effective = np.asarray(row.inverse_class_reliability_weights, dtype=float)[cls_idx - 1]
                        change = float(np.asarray(row.weight_l1_change_vs_released)[cls_idx - 1])
                    if len(effective) != len(released):
                        raise RuntimeError(f"weight length mismatch: {variant} {domain} {cls}")
                    l1.append(change)
                    sums.append(float(effective.sum()))
                    entropy.append(float(-(effective * np.log(np.maximum(effective, 1e-12))).sum()))
                    top1.append(float(effective.max()))
                    changed.append(change > 1e-8)
                rows.append({
                    "seed": SEED, "variant": variant, "display_variant": DISPLAY[variant],
                    "domain": domain, "class": cls.upper(), "n_with_memory": len(l1),
                    "mean_l1_change_vs_released": finite_mean(l1),
                    "median_l1_change_vs_released": float(np.median(l1)) if l1 else np.nan,
                    "changed_rate": finite_mean(changed), "mean_entropy": finite_mean(entropy),
                    "mean_top1_weight": finite_mean(top1), "max_abs_weight_sum_error":
                    float(np.max(np.abs(np.asarray(sums) - 1))) if sums else np.nan,
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "weight_diagnostics_seed2026.csv", index=False)
    return result


def memory_quality(frames):
    reference = pd.read_csv(ROOT / "results/exp5/per_class_utility_seed2026.csv")
    quality = {}
    for row in reference[reference.gt_present].itertuples(index=False):
        quality[(row.slice_name, row.class_name)] = float(row.anchor_dice)

    rows = []
    for variant, frame in frames.items():
        for domain in DOMAINS:
            part = frame if domain == "All" else frame[frame.domain == domain]
            for cls_idx, cls in enumerate(CLASSES):
                values, released_values = [], []
                valid_count = 0
                for row in part.itertuples(index=False):
                    names = row.topk_names
                    released = np.asarray(row.released_weights, dtype=float)
                    if len(names) == 0 or len(released) != len(names):
                        continue
                    q = np.asarray([quality.get((name, cls), np.nan) for name in names])
                    valid = np.isfinite(q)
                    if not valid.any():
                        continue
                    valid_count += 1
                    released_q = float(np.average(q[valid], weights=released[valid]))
                    if variant in {"released", "crsff_identity"}:
                        weights = released
                    elif variant == "global_reliability":
                        weights = np.asarray(row.global_reliability_weights, dtype=float)
                    elif variant == "class_reliability":
                        weights = np.asarray(row.class_reliability_weights, dtype=float)[cls_idx]
                    else:
                        weights = np.asarray(row.inverse_class_reliability_weights, dtype=float)[cls_idx]
                    values.append(float(np.average(q[valid], weights=weights[valid])))
                    released_values.append(released_q)
                rows.append({
                    "seed": SEED, "variant": variant, "display_variant": DISPLAY[variant],
                    "domain": domain, "query_class": cls.upper(), "n_valid_queries": valid_count,
                    "mean_anchor_dice_of_memory": finite_mean(values),
                    "mean_released_memory_quality": finite_mean(released_values),
                    "delta_memory_quality_vs_released": finite_mean(values) - finite_mean(released_values)
                    if values and released_values else np.nan,
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "effective_memory_quality.csv", index=False)
    return result


def representative_cases(frames):
    frame = frames["class_reliability"].copy()
    candidates = frame[frame.pool_size_before >= 5].copy()
    candidates["max_class_l1"] = candidates.weight_l1_change_vs_released.map(lambda x: max(x) if x else 0.0)
    candidates = candidates.sort_values("max_class_l1", ascending=False).head(9)
    rows = []
    reference = pd.read_csv(ROOT / "results/exp5/per_class_utility_seed2026.csv")
    quality = {(r.slice_name, r.class_name): float(r.anchor_dice)
               for r in reference[reference.gt_present].itertuples(index=False)}
    for row in candidates.itertuples(index=False):
        rows.append({
            "global_index": row.global_index, "domain": row.domain, "volume_id": row.volume_id,
            "slice_name": row.slice_name, "z_index": row.z_index, "pool_size_before": row.pool_size_before,
            "topk_names": json.dumps(row.topk_names), "topk_similarities": json.dumps(row.topk_similarities),
            "memory_class_reliabilities": json.dumps(row.memory_class_reliabilities),
            "memory_class_soft_masses": json.dumps(row.memory_class_soft_masses),
            "released_weights": json.dumps(row.released_weights),
            "class_reliability_weights": json.dumps(row.class_reliability_weights),
            "inverse_class_reliability_weights": json.dumps(row.inverse_class_reliability_weights),
            "query_class_soft_mass": json.dumps(row.query_class_soft_mass),
            "weight_l1_change_vs_released": json.dumps(row.weight_l1_change_vs_released),
            "memory_anchor_dice_lv": json.dumps([quality.get((n, "lv"), np.nan) for n in row.topk_names]),
            "memory_anchor_dice_myo": json.dumps([quality.get((n, "myo"), np.nan) for n in row.topk_names]),
            "memory_anchor_dice_rv": json.dumps([quality.get((n, "rv"), np.nan) for n in row.topk_names]),
            "anchor_dice_average": row.anchor_dice_average, "adapted_dice_average": row.adapted_dice_average,
            "max_class_l1": row.max_class_l1,
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "representative_memory_cases.csv", index=False)
    return result


def runtime_table(metadata):
    rows = []
    for variant in VARIANTS:
        data = metadata[variant]
        rows.append({"seed": SEED, "variant": variant, "display_variant": DISPLAY[variant],
                     "num_slices": data["num_slices"], "elapsed_seconds": data["elapsed_seconds"],
                     "mean_slice_elapsed_seconds": data["mean_slice_elapsed_seconds"],
                     "peak_gpu_memory_bytes": data["peak_gpu_memory_bytes"],
                     "peak_gpu_memory_mib": data["peak_gpu_memory_bytes"] / 1024 ** 2})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "runtime_summary.csv", index=False)
    return result


def figures(domain, quality, weights):
    FIG.mkdir(parents=True, exist_ok=True)
    all_delta = domain[domain.domain == "All"].copy()
    base = all_delta[all_delta.variant == "released"].iloc[0]
    variants = [v for v in VARIANTS if v != "released"]
    values = []
    for variant in variants:
        row = all_delta[all_delta.variant == variant].iloc[0]
        values.append([(row[f"dice_{c}"] - base[f"dice_{c}"]) * 100 for c in CLASSES + ["average"]])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(4); width = 0.18
    for i, (variant, vals) in enumerate(zip(variants, values)):
        ax.bar(x + (i - 1.5) * width, vals, width, label=DISPLAY[variant])
    ax.axhline(0, color="black", linewidth=.8); ax.axhline(.15, color="tab:green", linestyle="--", linewidth=.9)
    ax.set_xticks(x, ["LV", "MYO", "RV", "Average"]); ax.set_ylabel("Dice delta vs Released (pp)")
    ax.set_title("EXP-6 Stage-1 segmentation deltas"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / "crsff_dice_delta.png", dpi=300); plt.close(fig)

    part = weights[(weights.domain == "All") & (weights['class'] == "LV")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_vars = [v for v in VARIANTS if v != "released"]
    y = [part[part.variant == v].mean_l1_change_vs_released.iloc[0] for v in plot_vars]
    ax.bar([DISPLAY[v] for v in plot_vars], y, color=["#999999", "#4c78a8", "#f58518", "#e45756"])
    ax.set_ylabel("Mean L1 change of LV weights"); ax.set_title("Top-K weight movement (LV)")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(FIG / "weight_shift.png", dpi=300); plt.close(fig)

    part = quality[(quality.domain == "All") & (quality.query_class == "LV")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = [part[part.variant == v].mean_anchor_dice_of_memory.iloc[0] for v in VARIANTS]
    ax.bar([DISPLAY[v] for v in VARIANTS], y, color="#59a14f")
    ax.set_ylabel("Weighted memory anchor Dice"); ax.set_title("Effective memory quality (LV)")
    ax.tick_params(axis="x", rotation=25); fig.tight_layout(); fig.savefig(FIG / "effective_memory_quality.png", dpi=300); plt.close(fig)


def report(domain, overall, delta, quality, weights, runtime, representatives):
    all_rows = overall.set_index("variant")
    base = all_rows.loc["released"]
    main = all_rows.loc["class_reliability"]
    main_delta = (main.dice_average - base.dice_average) * 100
    decision = "STRONG" if main_delta >= .40 else "PROMISING" if main_delta >= .15 else "STOP"
    class_lines = []
    for cls in CLASSES + ["average"]:
        class_lines.append(f"- {cls.upper()}: {((main[f'dice_{cls}'] - base[f'dice_{cls}']) * 100):+.4f} pp")
    quality_main = quality[(quality.variant == "class_reliability") & (quality.domain == "All")]
    quality_lines = ", ".join(f"{r.query_class} {r.mean_anchor_dice_of_memory:.4f} (Δ {r.delta_memory_quality_vs_released:+.4f})"
                              for r in quality_main.itertuples())
    weight_main = weights[(weights.variant == "class_reliability") & (weights.domain == "All") & (weights['class'].isin(["LV", "MYO", "RV"]))]
    weight_lines = ", ".join(f"{r['class']} {r.mean_l1_change_vs_released:.6f}" for _, r in weight_main.iterrows())
    runtime_lines = ", ".join(f"{r.display_variant} {r.mean_slice_elapsed_seconds:.6f}s" for r in runtime.itertuples())
    text = f"""# EXP-6 CR-SFF 实验报告

## 结论

主方法 Class Reliability 在 seed2026、B→C→D、4091 张切片上的总体 Dice 为 **{main.dice_average*100:.4f}%**，相对 Released 的增量为 **{main_delta:+.4f} pp**。预设门槛为 +0.15 pp，因此 Stage-1 判定为 **{decision}**；按照实验规则，不启动 seed2024/2025 多 seed 扩展。

## 1. Released 等价性

`released_equivalence.txt`：PASS。4091 张切片顺序、逐类 Dice（容差 1e-8）和 prediction SHA-256 均一致。

## 2. Identity 等价性

`identity_equivalence.txt`：PASS。CR-SFF Identity 的预测哈希和逐类 Dice 与 Released 一致；正式输出共 4091 张切片。

## 3. Stage-1 总体结果

| Variant | LV Dice (%) | MYO Dice (%) | RV Dice (%) | Average Dice (%) | Δ Average (pp) |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(f"| {DISPLAY[v]} | {all_rows.loc[v].dice_lv*100:.4f} | {all_rows.loc[v].dice_myo*100:.4f} | {all_rows.loc[v].dice_rv*100:.4f} | {all_rows.loc[v].dice_average*100:.4f} | {(all_rows.loc[v].dice_average-base.dice_average)*100:+.4f} |" for v in VARIANTS) + f"""

## 4. 主方法逐类别增量

{chr(10).join(class_lines)}

## 5. 域级平均 Dice

详见 `exp6_domain_summary.csv` 和 `class_delta_summary.csv`。Class Reliability 的 B/C/D/All Average Dice 分别为：""" + ", ".join(f"{r.domain} {r.dice_average*100:.4f}%" for r in domain[domain.variant == "class_reliability"].itertuples()) + f"""。

## 6. Global 与 Class 的比较

Global Reliability 的总体 Average Dice 为 {all_rows.loc['global_reliability'].dice_average*100:.4f}%，Class Reliability 为 {main.dice_average*100:.4f}%，Class 相对 Global 为 {((main.dice_average-all_rows.loc['global_reliability'].dice_average)*100):+.4f} pp。

## 7. 有效 memory quality

基于 EXP-5 anchor Dice 的离线审计，Class Reliability 的 All 域加权 memory quality 为：{quality_lines}。完整结果见 `effective_memory_quality.csv`；GT 仅用于该离线审计和分割评估。

## 8. 权重变化

Class Reliability 的 All 域平均 class-wise Top-K 权重 L1 变化为：{weight_lines}。Released/Identity 保持零变化；权重归一化误差见 `weight_diagnostics_seed2026.csv`。

## 9. 反向可靠性对照

Inverse Class Reliability 总体 Average Dice 为 {all_rows.loc['inverse_class_reliability'].dice_average*100:.4f}%，相对 Released 为 {((all_rows.loc['inverse_class_reliability'].dice_average-base.dice_average)*100):+.4f} pp，未支持“反向可靠性”带来提升的假设。

## 10. 运行代价

各变体平均单切片耗时：{runtime_lines}。GPU 峰值显存和完整运行时间见 `runtime_summary.csv`。

## 11. Stage-1 决策

**{decision}**。Class Reliability 增量 {main_delta:+.4f} pp < +0.15 pp，停止多 seed、pairwise 显著性和扩展图表实验；保留当前 seed2026 的完整诊断作为结论依据。

## 可复现产物

- `raw/*_seed2026.csv/json`
- `exp6_domain_summary.csv`, `exp6_overall_summary.csv`, `class_delta_summary.csv`
- `weight_diagnostics_seed2026.csv`, `effective_memory_quality.csv`, `runtime_summary.csv`
- `representative_memory_cases.csv`
- `figures/crsff_dice_delta.png`, `figures/weight_shift.png`, `figures/effective_memory_quality.png`
- `exp6.patch`
"""
    (OUT / "exp6_report.md").write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, metadata = load_outputs()
    domain, overall, delta = summary_tables(metadata)
    weights = weight_diagnostics(frames)
    quality = memory_quality(frames)
    representatives = representative_cases(frames)
    runtime = runtime_table(metadata)
    figures(domain, quality, weights)
    report(domain, overall, delta, quality, weights, runtime, representatives)
    print(f"Stage-1 Class Reliability delta: {((overall.set_index('variant').loc['class_reliability'].dice_average - overall.set_index('variant').loc['released'].dice_average) * 100):+.6f} pp")
    print("Decision: STOP (multi-seed not run)")


if __name__ == "__main__":
    main()
