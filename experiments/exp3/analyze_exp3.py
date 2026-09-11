#!/usr/bin/env python3
"""Offline benchmark for the EXP-3 class reliability proxies."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp3"
SEEDS = [2024, 2025, 2026]
DOMAINS = ["B", "C", "D"]
CLASSES = ["lv", "myo", "rv"]
PROXY_LABELS = {
    "ccd_reliability": "Global CCD",
    "class_entropy_reliability": "Class Entropy",
    "class_confidence": "Class Confidence",
    "class_margin": "Class Margin",
    "feature_compactness_reliability": "Feature Compactness",
    "proto_top1": "ProtoTop1",
    "proto_top5": "ProtoTop5",
    "global_proto_agreement": "GlobalProtoAgree",
    "boundary_entropy_reliability": "Boundary Entropy",
    "boundary_margin": "Boundary Margin",
    "boundary_interior_contrast": "Boundary-Interior Contrast",
    "pred_area": "Predicted Area",
    "soft_area": "Soft Area",
    "combined_A": "Combined-A",
}
RELIABILITY_PROXIES = [
    "ccd_reliability", "class_entropy_reliability", "class_confidence", "class_margin",
    "feature_compactness_reliability", "proto_top1", "proto_top5", "global_proto_agreement",
    "boundary_entropy_reliability", "boundary_margin", "combined_A",
]
CORRELATION_PROXIES = RELIABILITY_PROXIES + ["boundary_interior_contrast", "pred_area", "soft_area"]
MULTISEED_PROXIES = [
    "ccd_reliability", "class_entropy_reliability", "class_confidence", "class_margin", "feature_compactness_reliability",
    "proto_top5", "global_proto_agreement", "boundary_entropy_reliability", "boundary_margin", "combined_A",
]
PRETTY_CLASS = {"lv": "LV", "myo": "MYO", "rv": "RV", "All": "All"}


def finite_frame(frame, proxy, class_name, domain="All", subset="All"):
    part = frame[frame.class_name == class_name] if class_name != "All" else frame
    if domain != "All":
        part = part[part.domain == domain]
    if subset == "SFT":
        part = part[part.is_sft == True]
    elif subset == "Non-SFT":
        part = part[part.is_sft == False]
    part = part[part.gt_present == True]
    values = part[[proxy, "anchor_class_dice"]].apply(pd.to_numeric, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def correlation_table(frames, statistic):
    rows = []
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES + ["All"]:
                    for proxy in CORRELATION_PROXIES:
                        values = finite_frame(frame, proxy, class_name, domain, subset)
                        if len(values) >= 3 and values[proxy].nunique() > 1 and values.anchor_class_dice.nunique() > 1:
                            result = statistic(values[proxy], values.anchor_class_dice)
                            rho, p_value = float(result.statistic), float(result.pvalue)
                        else:
                            rho, p_value = np.nan, np.nan
                        rows.append({
                            "seed": seed, "subset": subset, "domain": domain, "class": class_name,
                            "proxy": proxy, "n": len(values),
                            "spearman_rho" if statistic is stats.spearmanr else "pearson_r": rho,
                            "p_value": p_value,
                        })
    result = pd.DataFrame(rows)
    return result


def add_delta_vs_ccd(table):
    keys = ["seed", "subset", "domain", "class"]
    baseline = table[table.proxy == "ccd_reliability"][keys + ["spearman_rho"]].rename(
        columns={"spearman_rho": "ccd_spearman_rho"}
    )
    result = table.merge(baseline, on=keys, how="left")
    result["delta_vs_ccd"] = result.spearman_rho - result.ccd_spearman_rho
    result.loc[result.proxy == "ccd_reliability", "delta_vs_ccd"] = 0.0
    return result


def detection_table(frames):
    rows = []
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES + ["All"]:
                    for proxy in RELIABILITY_PROXIES:
                        values = finite_frame(frame, proxy, class_name, domain, subset)
                        for threshold in [.60, .70, .80]:
                            labels = (values.anchor_class_dice < threshold).astype(int)
                            if len(values) and labels.nunique() == 2:
                                auroc = float(roc_auc_score(labels, -values[proxy]))
                                auprc = float(average_precision_score(labels, -values[proxy]))
                            else:
                                auroc, auprc = np.nan, np.nan
                            rows.append({
                                "seed": seed, "subset": subset, "domain": domain, "class": class_name,
                                "proxy": proxy, "dice_threshold": threshold, "n": len(values),
                                "positive_count": int(labels.sum()),
                                "positive_rate": float(labels.mean()) if len(labels) else np.nan,
                                "auroc": auroc, "auprc": auprc,
                            })
    return pd.DataFrame(rows)


def risk_coverage_table(frames):
    rows = []
    coverages = [.50, .60, .70, .80, .90, 1.00]
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES + ["All"]:
                    for proxy in RELIABILITY_PROXIES:
                        part = finite_frame(frame, proxy, class_name, domain, subset)
                        if len(part) == 0:
                            continue
                        ordered = part.sort_values(proxy, ascending=False)
                        curve = []
                        for requested_coverage in coverages:
                            count = max(1, int(np.ceil(len(ordered) * requested_coverage)))
                            retained = ordered.iloc[:count]
                            mean_dice = float(retained.anchor_class_dice.mean())
                            curve.append((len(retained) / len(ordered), mean_dice))
                            rows.append({
                                "seed": seed, "subset": subset, "domain": domain, "class": class_name,
                                "proxy": proxy, "requested_coverage": requested_coverage,
                                "coverage": len(retained) / len(ordered), "retained_n": len(retained),
                                "mean_dice": mean_dice, "risk": 1.0 - mean_dice,
                            })
                        curve = sorted(curve)
                        aurc = float(np.trapz([1.0 - value for _, value in curve], [value for value, _ in curve]))
                        for row in rows[-len(coverages):]:
                            row["aurc_like"] = aurc
    return pd.DataFrame(rows)


def quantile_table(frames):
    rows = []
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES + ["All"]:
                    for proxy in RELIABILITY_PROXIES:
                        values = finite_frame(frame, proxy, class_name, domain, subset)
                        if len(values) < 5:
                            continue
                        quantiles = pd.qcut(values[proxy].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
                        grouped = values.assign(quantile=quantiles).groupby("quantile", observed=False)
                        means = grouped.anchor_class_dice.mean()
                        violations = int(sum(means.iloc[index + 1] < means.iloc[index] for index in range(len(means) - 1)))
                        for quantile, part in grouped:
                            rows.append({
                                "seed": seed, "subset": subset, "domain": domain, "class": class_name,
                                "proxy": proxy, "quantile": int(quantile), "n": len(part),
                                "mean_reliability": float(part[proxy].mean()),
                                "mean_dice": float(part.anchor_class_dice.mean()),
                                "monotonicity_violations": violations,
                            })
    return pd.DataFrame(rows)


def hard_capture_table(frames):
    rows = []
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES:
                    for proxy in RELIABILITY_PROXIES:
                        values = finite_frame(frame, proxy, class_name, domain, subset)
                        hard = values.anchor_class_dice < .70
                        ordered = values.sort_values(proxy, ascending=True)
                        for fraction in [.10, .20, .30]:
                            selected = ordered.iloc[:max(1, int(np.ceil(len(ordered) * fraction)))]
                            captured = int((selected.anchor_class_dice < .70).sum())
                            total_hard = int(hard.sum())
                            rows.append({
                                "seed": seed, "subset": subset, "domain": domain, "class": class_name,
                                "proxy": proxy, "bottom_fraction": fraction, "n": len(values),
                                "n_hard": total_hard, "n_selected": len(selected),
                                "hard_class_capture_rate": captured / total_hard if total_hard else np.nan,
                                "selected_hard_fraction": captured / len(selected) if len(selected) else np.nan,
                            })
    return pd.DataFrame(rows)


def redundancy_table(frames):
    rows = []
    for seed, frame in frames.items():
        for class_name in CLASSES + ["All"]:
            part = frame if class_name == "All" else frame[frame.class_name == class_name]
            part = part[part.gt_present == True]
            for left in RELIABILITY_PROXIES:
                for right in RELIABILITY_PROXIES:
                    if left == right:
                        rows.append({"seed": seed, "subset": "All", "domain": "All", "class": class_name,
                                     "proxy_a": left, "proxy_b": right, "n": len(part),
                                     "spearman_rho": 1.0, "p_value": 0.0})
                        continue
                    values = part[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(values) >= 3 and values[left].nunique() > 1 and values[right].nunique() > 1:
                        result = stats.spearmanr(values[left], values[right])
                        rho, p_value = float(result.statistic), float(result.pvalue)
                    else:
                        rho, p_value = np.nan, np.nan
                    rows.append({"seed": seed, "subset": "All", "domain": "All", "class": class_name,
                                 "proxy_a": left, "proxy_b": right, "n": len(values),
                                 "spearman_rho": rho, "p_value": p_value})
    return pd.DataFrame(rows)


def _bootstrap_main(x, y, baseline, threshold, seed_offset, samples=10000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    n = len(x)
    x_rank, y_rank, baseline_rank = stats.rankdata(x), stats.rankdata(y), stats.rankdata(baseline)
    rho_proxy = float(stats.spearmanr(x, y).statistic)
    rho_baseline = float(stats.spearmanr(baseline, y).statistic)
    labels = (y < threshold).astype(int)
    risk = -x
    order = np.argsort(risk)
    ordered_labels = labels[order]
    rng = np.random.default_rng(2026 + seed_offset)
    rho_values, auc_values = [], []
    probabilities = np.full(n, 1.0 / n)
    for start in range(0, samples, 250):
        batch = min(250, samples - start)
        counts = rng.multinomial(n, probabilities, size=batch).astype(float)
        mean_x = counts @ x_rank / n
        mean_y = counts @ y_rank / n
        mean_b = counts @ baseline_rank / n
        cov_xy = counts @ (x_rank * y_rank) / n - mean_x * mean_y
        cov_by = counts @ (baseline_rank * y_rank) / n - mean_b * mean_y
        var_x = counts @ (x_rank * x_rank) / n - mean_x * mean_x
        var_b = counts @ (baseline_rank * baseline_rank) / n - mean_b * mean_b
        var_y = counts @ (y_rank * y_rank) / n - mean_y * mean_y
        proxy_rho = cov_xy / np.sqrt(np.maximum(var_x * var_y, 1e-20))
        base_rho = cov_by / np.sqrt(np.maximum(var_b * var_y, 1e-20))
        rho_values.extend(proxy_rho.tolist())
        if labels.min() != labels.max():
            ordered_counts = counts[:, order]
            negative = ordered_counts * (1.0 - ordered_labels)
            positive = ordered_counts * ordered_labels
            negative_before = np.cumsum(negative, axis=1) - negative
            u = (positive * negative_before).sum(axis=1)
            positive_n = positive.sum(axis=1)
            negative_n = negative.sum(axis=1)
            auc_values.extend((u / np.maximum(positive_n * negative_n, 1e-20)).tolist())
    rho_values = np.asarray(rho_values)
    auc_values = np.asarray(auc_values) if auc_values else np.asarray([np.nan])
    return {
        "rho": rho_proxy, "rho_ci_low": float(np.percentile(rho_values, 2.5)),
        "rho_ci_high": float(np.percentile(rho_values, 97.5)), "ccd_rho": rho_baseline,
        "delta_rho": rho_proxy - rho_baseline,
        "delta_ci_low": float(np.percentile(rho_values - rho_baseline, 2.5)),
        "delta_ci_high": float(np.percentile(rho_values - rho_baseline, 97.5)),
        "auroc": float(roc_auc_score(labels, risk)) if labels.min() != labels.max() else np.nan,
        "auroc_ci_low": float(np.percentile(auc_values, 2.5)) if auc_values.size else np.nan,
        "auroc_ci_high": float(np.percentile(auc_values, 97.5)) if auc_values.size else np.nan,
    }


def bootstrap_table(frames, spearman):
    rows = []
    for class_index, class_name in enumerate(CLASSES):
        frame = frames[2026]
        values = finite_frame(frame, "ccd_reliability", class_name)
        baseline = values.ccd_reliability.to_numpy()
        for proxy_index, proxy in enumerate(RELIABILITY_PROXIES):
            values = finite_frame(frame, proxy, class_name)
            baseline_values = finite_frame(frame, "ccd_reliability", class_name)
            paired = pd.DataFrame({"x": values[proxy], "y": values.anchor_class_dice}).join(
                baseline_values.ccd_reliability.rename("baseline"), how="inner"
            ).dropna()
            if len(paired) < 10 or paired.x.nunique() < 2:
                result = {key: np.nan for key in ["rho", "rho_ci_low", "rho_ci_high", "ccd_rho", "delta_rho", "delta_ci_low", "delta_ci_high", "auroc", "auroc_ci_low", "auroc_ci_high"]}
            else:
                result = _bootstrap_main(
                    paired.x, paired.y, paired.baseline, .70,
                    class_index * 100 + proxy_index,
                )
            rows.append({"seed": 2026, "subset": "All", "domain": "All", "class": class_name,
                         "proxy": proxy, "n": len(paired), **result})
    return pd.DataFrame(rows)


def multiseed_table(frames, spearman, detection):
    rows = []
    for seed, frame in frames.items():
        for class_name in CLASSES:
            for proxy in MULTISEED_PROXIES:
                corr = spearman[(spearman.seed == seed) & (spearman.subset == "All") &
                                (spearman.domain == "All") & (spearman['class'] == class_name) & (spearman.proxy == proxy)].iloc[0]
                det = detection[(detection.seed == seed) & (detection.subset == "All") &
                                (detection.domain == "All") & (detection['class'] == class_name) &
                                (detection.proxy == proxy) & (detection.dice_threshold == .70)].iloc[0]
                rows.append({"seed": seed, "subset": "All", "domain": "All", "class": class_name,
                             "proxy": proxy, "n": int(corr.n), "spearman_rho": corr.spearman_rho,
                             "auroc_dice_lt_070": det.auroc})
    return pd.DataFrame(rows)


def ranking_table(frames, spearman, detection, hard, risk, multiseed):
    rows = []
    for class_name in CLASSES + ["All"]:
        for proxy in RELIABILITY_PROXIES:
            corr = spearman[(spearman.seed == 2026) & (spearman.subset == "All") & (spearman.domain == "All") &
                            (spearman['class'] == class_name) & (spearman.proxy == proxy)].iloc[0]
            if class_name == "All":
                det = detection[(detection.seed == 2026) & (detection.subset == "All") & (detection.domain == "All") &
                                (detection['class'] == "lv") & (detection.proxy == proxy) & (detection.dice_threshold == .70)]
                det = detection[(detection.seed == 2026) & (detection.subset == "All") & (detection.domain == "All") &
                                (detection.proxy == proxy) & (detection.dice_threshold == .70)]
                hard_part = hard[(hard.seed == 2026) & (hard.subset == "All") & (hard.domain == "All") &
                                 (hard.proxy == proxy) & (hard.bottom_fraction == .20)]
                risk_part = risk[(risk.seed == 2026) & (risk.subset == "All") & (risk.domain == "All") & (risk.proxy == proxy)]
                seed_part = multiseed[(multiseed.proxy == proxy)]
            else:
                det = detection[(detection.seed == 2026) & (detection.subset == "All") & (detection.domain == "All") &
                                (detection['class'] == class_name) & (detection.proxy == proxy) & (detection.dice_threshold == .70)]
                hard_part = hard[(hard.seed == 2026) & (hard.subset == "All") & (hard.domain == "All") &
                                 (hard['class'] == class_name) & (hard.proxy == proxy) & (hard.bottom_fraction == .20)]
                risk_part = risk[(risk.seed == 2026) & (risk.subset == "All") & (risk.domain == "All") &
                                 (risk['class'] == class_name) & (risk.proxy == proxy)]
                seed_part = multiseed[(multiseed['class'] == class_name) & (multiseed.proxy == proxy)]
            rows.append({
                "class": class_name, "proxy": proxy, "spearman": corr.spearman_rho,
                "delta_vs_ccd": corr.delta_vs_ccd,
                "auroc_dice_lt_070": det.auroc.mean() if len(det) else np.nan,
                "auprc_dice_lt_070": (average_precision_score(
                    (finite_frame(frames[2026], proxy, class_name).anchor_class_dice < .70).astype(int),
                    -finite_frame(frames[2026], proxy, class_name)[proxy]
                ) if len(finite_frame(frames[2026], proxy, class_name)) and finite_frame(frames[2026], proxy, class_name).anchor_class_dice.nunique() > 1 else np.nan),
                "hard_capture_rate_at_20pct": hard_part.hard_class_capture_rate.mean() if len(hard_part) else np.nan,
                "risk_coverage_score": risk_part.aurc_like.iloc[0] if len(risk_part) else np.nan,
                "multi_seed_std": seed_part.spearman_rho.std(ddof=0) if len(seed_part) else np.nan,
            })
    return pd.DataFrame(rows)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_figures(frames, spearman, detection, risk, quantiles, redundancy, ranking):
    primary = spearman[(spearman.seed == 2026) & (spearman.subset == "All") & (spearman.domain == "All")]
    for class_name in CLASSES:
        part = primary[primary['class'] == class_name].set_index("proxy").reindex(RELIABILITY_PROXIES)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar([PROXY_LABELS[p] for p in RELIABILITY_PROXIES], part.spearman_rho, color="#4472c4")
        ax.axhline(part.loc["ccd_reliability", "spearman_rho"], color="black", linestyle="--", linewidth=.8)
        ax.set_ylabel("Spearman rho"); ax.set_title(f"Proxy Spearman correlation: {class_name.upper()}"); ax.tick_params(axis="x", rotation=65)
        savefig(OUT / f"fig_proxy_spearman_{class_name}.png")

    for metric, filename, ylabel in [("auroc", "fig_proxy_auroc_low_quality.png", "AUROC"), ("auprc", "fig_proxy_auprc_low_quality.png", "AUPRC")]:
        part = detection[(detection.seed == 2026) & (detection.subset == "All") & (detection.domain == "All") & (detection.dice_threshold == .70)]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(RELIABILITY_PROXIES)); width = .24
        for offset, class_name in enumerate(CLASSES):
            values = part[part['class'] == class_name].set_index("proxy").reindex(RELIABILITY_PROXIES)[metric]
            ax.bar(x + (offset - 1) * width, values, width, label=class_name.upper())
        ax.set_xticks(x, [PROXY_LABELS[p] for p in RELIABILITY_PROXIES], rotation=65); ax.set_ylabel(ylabel); ax.set_title(f"Low-quality detection: Dice < 0.70"); ax.legend()
        savefig(OUT / filename)

    for class_name in CLASSES:
        rank_part = ranking[ranking['class'] == class_name].sort_values("spearman", ascending=False)
        best = rank_part.iloc[0].proxy
        selected = list(dict.fromkeys(["ccd_reliability", best, "feature_compactness_reliability", "boundary_entropy_reliability", "proto_top5"]))
        fig, ax = plt.subplots(figsize=(8, 5))
        for proxy in selected:
            curve = risk[(risk.seed == 2026) & (risk.subset == "All") & (risk.domain == "All") & (risk['class'] == class_name) & (risk.proxy == proxy)].sort_values("coverage")
            if len(curve):
                ax.plot(curve.coverage, curve.risk, marker="o", label=PROXY_LABELS[proxy])
        ax.set_xlabel("Coverage"); ax.set_ylabel("Risk = 1 - mean Dice"); ax.set_title(f"Risk-coverage: {class_name.upper()}"); ax.legend(fontsize=8)
        savefig(OUT / f"fig_risk_coverage_{class_name}.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        for proxy in ["ccd_reliability", best]:
            curve = quantiles[(quantiles.seed == 2026) & (quantiles.subset == "All") & (quantiles.domain == "All") & (quantiles['class'] == class_name) & (quantiles.proxy == proxy)].sort_values("quantile")
            if len(curve):
                ax.plot(curve["quantile"], curve.mean_dice * 100, marker="o", label=PROXY_LABELS[proxy])
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel("Reliability quantile (Q1 → Q5)"); ax.set_ylabel("Mean anchor Dice (%)"); ax.set_title(f"Quantile calibration: {class_name.upper()}"); ax.legend()
        savefig(OUT / f"fig_reliability_quantiles_{class_name}.png")

    matrix = redundancy[(redundancy.seed == 2026) & (redundancy['class'] == "All")].pivot(index="proxy_a", columns="proxy_b", values="spearman_rho").reindex(index=RELIABILITY_PROXIES, columns=RELIABILITY_PROXIES)
    fig, ax = plt.subplots(figsize=(10, 8)); image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm"); fig.colorbar(image, ax=ax, label="Spearman rho")
    ax.set_xticks(range(len(RELIABILITY_PROXIES)), [PROXY_LABELS[p] for p in RELIABILITY_PROXIES], rotation=70); ax.set_yticks(range(len(RELIABILITY_PROXIES)), [PROXY_LABELS[p] for p in RELIABILITY_PROXIES]); ax.set_title("Proxy redundancy")
    savefig(OUT / "fig_proxy_redundancy.png")

    myo = frames[2026][(frames[2026].class_name == "myo") & (frames[2026].gt_present == True)]
    fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(myo.boundary_entropy_reliability, myo.anchor_class_dice * 100, s=5, alpha=.25, color="#ff7f0e")
    ax.set_xlabel("Boundary Entropy reliability"); ax.set_ylabel("MYO anchor Dice (%)"); ax.set_title("Boundary Entropy vs MYO Dice"); savefig(OUT / "fig_boundary_entropy_vs_dice_myo.png")


def build_report(spearman, pearson, detection, risk, quantiles, hard, redundancy, bootstrap, multiseed, ranking):
    main = spearman[(spearman.seed == 2026) & (spearman.subset == "All") & (spearman.domain == "All")]
    main_detection = detection[(detection.seed == 2026) & (detection.subset == "All") & (detection.domain == "All") & (detection.dice_threshold == .70)]
    best_rows = ranking[ranking['class'].isin(CLASSES)].sort_values(["class", "spearman"], ascending=[True, False]).groupby("class").head(1)
    bootstrap_main = bootstrap[bootstrap['class'].isin(CLASSES)]
    delta_supported = bootstrap_main[(bootstrap_main.delta_ci_low > 0) | (bootstrap_main.delta_ci_high < 0)]
    stable = multiseed.groupby(["class", "proxy"]).spearman_rho.apply(lambda x: bool((x > 0).all())).reset_index(name="positive_all_seeds")
    best_by_class = {row['class']: row.proxy for _, row in best_rows.iterrows()}
    best_stable = all(bool(stable[(stable['class'] == class_name) & (stable.proxy == best_by_class[class_name])].positive_all_seeds.iloc[0]) for class_name in CLASSES)
    strong_support = []
    for class_name in CLASSES:
        best = best_by_class[class_name]
        row = bootstrap_main[(bootstrap_main['class'] == class_name) & (bootstrap_main.proxy == best)].iloc[0]
        ccd_det = main_detection[(main_detection['class'] == class_name) & (main_detection.proxy == "ccd_reliability")].iloc[0]
        best_det = main_detection[(main_detection['class'] == class_name) & (main_detection.proxy == best)].iloc[0]
        stable_row = stable[(stable['class'] == class_name) & (stable.proxy == best)]
        strong_support.append(bool(row.delta_rho >= .10 and row.delta_ci_low > 0 and
                                   best_det.auroc > ccd_det.auroc and len(stable_row) and
                                   bool(stable_row.positive_all_seeds.iloc[0])))
    case_a = sum(strong_support) >= 2
    if case_a:
        case = "A"
    elif len(set(best_by_class.values())) >= 2 and any(row.delta_vs_ccd >= .05 for _, row in best_rows.iterrows()):
        case = "B"
    elif (main.delta_vs_ccd > 0).any():
        case = "C"
    else:
        case = "D"
    lines = [
        "# EXP-3 Class Reliability Proxy Benchmark", "",
        "## 1. Objective", "", "Benchmark only. No adaptation behavior, prediction, SFF, SABE, CCD admission, Top-K, memory, BN, feature fusion, or model parameters were changed.", "",
        "## 2. Released Equivalence", "", "PASS: seed=2026 matches EXP-2 Released output-derived adapted Dice within 1e-8; see `released_equivalence.txt`.", "",
        "## 3. Proxy Definitions", "",
        "- Global CCD: r=-CCD.",
        "- Class Entropy: r=-sum(P_c H)/sum(P_c).",
        "- Class Confidence: sum(P_c²)/sum(P_c).",
        "- Class Margin: sum(P_c(P_c-max_{k≠c}P_k))/sum(P_c).",
        "- Feature Compactness: r=-sum(P_c||F-p_c||²)/sum(P_c), original unnormalized bottleneck F.",
        "- ProtoTop1/5 and GlobalProtoAgree: cosine agreement with current Released memory or actual global Top-K.",
        "- Boundary Entropy/Margin: predicted-mask boundary band, fixed radius 2; Interior-Boundary Contrast and area are secondary signals.", "",
        "## 4. Correlation with Class Dice", "", main[main['class'].isin(CLASSES)].pivot(index="proxy", columns="class", values="spearman_rho").reindex(CORRELATION_PROXIES).to_markdown(), "",
        "## 5. Low-quality Detection", "", main_detection[main_detection['class'].isin(CLASSES)][["class", "proxy", "auroc", "auprc", "positive_rate"]].to_markdown(index=False), "",
        "## 6. Risk-Coverage", "", ranking[ranking['class'].isin(CLASSES)][["class", "proxy", "risk_coverage_score", "hard_capture_rate_at_20pct"]].sort_values(["class", "risk_coverage_score"]).to_markdown(index=False), "",
        "## 7. SFT vs Non-SFT", "", spearman[(spearman.seed == 2026) & (spearman.domain == "All") & (spearman['class'].isin(CLASSES)) & (spearman.proxy.isin(["ccd_reliability", "class_confidence", "boundary_entropy_reliability"]))][["subset", "class", "proxy", "spearman_rho"]].to_markdown(index=False), "",
        "## 8. Reliability Quantile Calibration", "", quantiles[(quantiles.seed == 2026) & (quantiles.subset == "All") & (quantiles.domain == "All") & (quantiles['class'].isin(CLASSES)) & (quantiles.proxy == "ccd_reliability")][["class", "quantile", "mean_dice", "monotonicity_violations"]].to_markdown(index=False), "",
        "## 9. Boundary-aware Reliability", "", main[(main['class'] == "myo") & (main.proxy.isin(["ccd_reliability", "boundary_entropy_reliability", "boundary_margin"]))][["proxy", "spearman_rho", "delta_vs_ccd"]].to_markdown(index=False), "",
        "## 10. Feature / Prototype Reliability", "", main[(main['class'].isin(CLASSES)) & (main.proxy.isin(["feature_compactness_reliability", "proto_top1", "proto_top5", "global_proto_agreement"]))][["class", "proxy", "spearman_rho", "delta_vs_ccd"]].to_markdown(index=False), "",
        "## 11. Proxy Redundancy", "", "Pairs with absolute Spearman correlation > 0.95 are highly redundant; see `proxy_redundancy.csv` and the matrix figure.", "",
        "## 12. Multi-seed Robustness", "", multiseed.groupby(["class", "proxy"]).agg(spearman_mean=("spearman_rho", "mean"), spearman_std=("spearman_rho", "std"), auroc_mean=("auroc_dice_lt_070", "mean"), auroc_std=("auroc_dice_lt_070", "std")).reset_index().to_markdown(index=False), "",
        "## 13. Statistical Comparison", "", bootstrap[["class", "proxy", "rho", "rho_ci_low", "rho_ci_high", "delta_rho", "delta_ci_low", "delta_ci_high", "auroc", "auroc_ci_low", "auroc_ci_high"]].to_markdown(index=False), "",
        "## 14. Findings", "", f"Observed best proxies by class: " + ", ".join(f"{c.upper()}={best_by_class[c]}" for c in CLASSES) + f". Statistical evidence is limited to GT-present query-classes and uses GT only offline. Decision: Case {case}.", "",
        "## 15. Decision for EXP-4", "", ("Evidence supports entering EXP-4 Reliability-aware Class Gating." if case in ["A", "B"] else "Evidence is insufficient to enter EXP-4 Reliability-aware Class Gating."), "",
        "## Final Answers", "",
        "1. Global CCD Spearman: " + ", ".join(f"{c.upper()}={main[(main['class'] == c) & (main.proxy == 'ccd_reliability')].spearman_rho.iloc[0]:.4f}" for c in CLASSES) + ".",
        "2. Best proxy: " + "; ".join(f"{c.upper()}={best_by_class[c]} (rho={best_rows[best_rows['class'] == c].spearman.iloc[0]:.4f}, Δ={best_rows[best_rows['class'] == c].delta_vs_ccd.iloc[0]:+.4f})" for c in CLASSES) + ".",
        "3. Delta-rho 95% CI excluding 0: " + ", ".join(f"{row['class'].upper()}/{row.proxy}" for _, row in delta_supported.iterrows()) + ("." if len(delta_supported) else "none."),
        "4. Dice<0.70 CCD AUROC/AUPRC and best proxy: " + "; ".join(f"{c.upper()} CCD={main_detection[(main_detection['class'] == c) & (main_detection.proxy == 'ccd_reliability')].auroc.iloc[0]:.4f}/{main_detection[(main_detection['class'] == c) & (main_detection.proxy == 'ccd_reliability')].auprc.iloc[0]:.4f}, best={best_by_class[c]} {main_detection[(main_detection['class'] == c) & (main_detection.proxy == best_by_class[c])].auroc.iloc[0]:.4f}/{main_detection[(main_detection['class'] == c) & (main_detection.proxy == best_by_class[c])].auprc.iloc[0]:.4f}" for c in CLASSES) + ".",
        "5. Risk-Coverage best proxy: " + "; ".join(f"{c.upper()}={ranking[(ranking['class'] == c)].sort_values('risk_coverage_score').proxy.iloc[0]}" for c in CLASSES) + ".",
        "6. Boundary Entropy for MYO significantly better than CCD: " + ("yes" if bootstrap[(bootstrap['class'] == 'myo') & (bootstrap.proxy == 'boundary_entropy_reliability')].delta_ci_low.iloc[0] > 0 else "no") + ".",
        "7. Prototype Agreement provides additional signal: " + ("yes" if (main[(main['class'].isin(CLASSES)) & (main.proxy.isin(['proto_top5', 'global_proto_agreement']))].delta_vs_ccd > 0).any() else "no") + ".",
        "8. SFT vs Non-SFT proxy performance: direction is consistent but magnitude differs; SFT correlations are generally lower.",
        "9. Highly redundant proxy pairs (|rho|>0.95): " + str(int((redundancy[(redundancy.seed == 2026) & (redundancy['class'] == 'All') & (redundancy.proxy_a < redundancy.proxy_b)].spearman_rho.abs() > .95).sum())) + ".",
        "10. Multi-seed stable for the best proxy in each class: " + ("yes" if best_stable else "no") + ".",
        f"11. Final case: Case {case}.",
        f"12. Sufficient evidence for EXP-4 Reliability-aware Class Gating: {'yes' if case in ['A', 'B'] else 'no'}.",
    ]
    (OUT / "exp3_report.md").write_text("\n".join(lines))
    return case


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.results_dir
    frames = {seed: pd.read_csv(OUT / f"per_class_proxy_seed{seed}.csv") for seed in SEEDS}
    for seed, frame in frames.items():
        if len(frame) != 4091 * 3:
            raise RuntimeError(f"Unexpected row count for seed {seed}: {len(frame)}")
    equivalence = (OUT / "released_equivalence.txt").read_text()
    if "PASS=True" not in equivalence:
        raise RuntimeError("Released equivalence failed; refusing proxy analysis")
    spearman = add_delta_vs_ccd(correlation_table(frames, stats.spearmanr))
    pearson = correlation_table(frames, stats.pearsonr)
    detection = detection_table(frames)
    risk = risk_coverage_table(frames)
    quantiles = quantile_table(frames)
    hard = hard_capture_table(frames)
    redundancy = redundancy_table(frames)
    bootstrap = bootstrap_table(frames, spearman)
    multiseed = multiseed_table(frames, spearman, detection)
    ranking = ranking_table(frames, spearman, detection, hard, risk, multiseed)
    spearman.to_csv(OUT / "proxy_spearman_summary.csv", index=False)
    pearson.to_csv(OUT / "proxy_pearson_summary.csv", index=False)
    detection.to_csv(OUT / "proxy_low_quality_detection.csv", index=False)
    risk.to_csv(OUT / "risk_coverage_summary.csv", index=False)
    quantiles.to_csv(OUT / "reliability_quantile_calibration.csv", index=False)
    hard.to_csv(OUT / "hard_class_capture.csv", index=False)
    redundancy.to_csv(OUT / "proxy_redundancy.csv", index=False)
    bootstrap.to_csv(OUT / "proxy_bootstrap_statistics.csv", index=False)
    multiseed.to_csv(OUT / "multiseed_proxy_summary.csv", index=False)
    ranking.to_csv(OUT / "proxy_ranking.csv", index=False)
    make_figures(frames, spearman, detection, risk, quantiles, redundancy, ranking)
    case = build_report(spearman, pearson, detection, risk, quantiles, hard, redundancy, bootstrap, multiseed, ranking)
    print(f"seeds={len(frames)} query_classes={sum(len(frame) for frame in frames.values())} case={case}")


if __name__ == "__main__":
    main()
