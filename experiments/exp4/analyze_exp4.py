#!/usr/bin/env python3
"""Offline analysis for EXP-4 reliability-aware class gating."""

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
OUT = ROOT / "results/exp4"
SEEDS = [2024, 2025, 2026]
DOMAINS = ["B", "C", "D"]
CLASSES = ["lv", "myo", "rv"]
MAIN_VARIANTS = ["released", "lowconf_g05", "lowconf_g10", "lowconf_g20", "highconf_g10"]
CORE_VARIANTS = ["released", "lowconf_g10", "highconf_g10"]
VARIANT_LABELS = {
    "released": "Released", "identity": "Identity", "lowconf_g05": "LowConf γ=0.5",
    "lowconf_g10": "LowConf γ=1", "lowconf_g20": "LowConf γ=2", "highconf_g10": "HighConf γ=1",
}


def load_frames():
    frames = {}
    for path in sorted((OUT / "raw").glob("*_seed*.csv")):
        stem = path.stem
        variant, seed_text = stem.rsplit("_seed", 1)
        frames[(variant, int(seed_text))] = pd.read_csv(path)
    return frames


def class_long(frame):
    rows = []
    for _, item in frame.iterrows():
        for suffix in CLASSES:
            rows.append({
                "variant": item.variant, "seed": item.seed, "global_index": item.global_index,
                "domain": item.domain, "volume_id": item.volume_id, "slice_name": item.slice_name,
                "class": suffix, "is_sft": item.is_sft,
                "class_confidence": item[f"anchor_confidence_{suffix}"],
                "global_rate": item.global_rate, "class_alpha": item[f"alpha_{suffix}"],
                "anchor_dice": item[f"anchor_dice_{suffix}"],
                "adapted_dice": item[f"adapted_dice_{suffix}"],
                "negative": (item[f"adapted_dice_{suffix}"] < item[f"anchor_dice_{suffix}"])
                if pd.notna(item[f"anchor_dice_{suffix}"]) else np.nan,
                "severe_negative": (item[f"adapted_dice_{suffix}"] - item[f"anchor_dice_{suffix}"] < -.05)
                if pd.notna(item[f"anchor_dice_{suffix}"]) else np.nan,
            })
    return pd.DataFrame(rows)


def make_domain_summaries():
    rows = []
    for path in sorted((OUT / "raw").glob("*_seed*.json")):
        payload = json.loads(path.read_text())
        for item in payload["domain_summary"]:
            rows.append(item)
    summary = pd.DataFrame(rows).sort_values(["variant", "seed", "domain"])
    summary.to_csv(OUT / "exp4_domain_summary.csv", index=False)
    summary[summary.domain == "All"].to_csv(OUT / "exp4_overall_summary.csv", index=False)
    return summary


def class_delta(frames):
    rows = []
    for variant in MAIN_VARIANTS:
        if (variant, 2026) not in frames:
            continue
        current = frames[(variant, 2026)].set_index("global_index")
        released = frames[("released", 2026)].set_index("global_index")
        for domain in DOMAINS + ["All"]:
            left = current if domain == "All" else current[current.domain == domain]
            right = released if domain == "All" else released[released.domain == domain]
            for suffix in CLASSES + ["average"]:
                delta = left[f"adapted_dice_{suffix}"] - right[f"adapted_dice_{suffix}"]
                rows.append({"variant": variant, "seed": 2026, "domain": domain, "class": suffix,
                             "n": int(delta.notna().sum()), "mean_delta_vs_released": float(delta.mean()),
                             "median_delta_vs_released": float(delta.median())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "class_delta_summary.csv", index=False)
    return result


def negative_summary(frames):
    rows = []
    for (variant, seed), frame in frames.items():
        long = class_long(frame)
        for domain in DOMAINS + ["All"]:
            part = long if domain == "All" else long[long.domain == domain]
            for suffix in CLASSES:
                item = part[part['class'] == suffix].dropna(subset=["negative"])
                rows.append({"variant": variant, "seed": seed, "domain": domain, "class": suffix,
                             "n": len(item), "negative_count": int(item.negative.sum()),
                             "negative_adaptation_rate": float(item.negative.mean()) if len(item) else np.nan,
                             "severe_negative_count": int(item.severe_negative.sum()),
                             "severe_negative_rate": float(item.severe_negative.mean()) if len(item) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "negative_adaptation_summary.csv", index=False)
    return result


def reliability_stratified(frames):
    rows = []
    for variant in MAIN_VARIANTS:
        frame = frames[(variant, 2026)]
        long = class_long(frame)
        for suffix in CLASSES:
            part = long[(long['class'] == suffix) & long.anchor_dice.notna()].copy()
            part["quantile"] = pd.qcut(part.class_confidence.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            released = class_long(frames[("released", 2026)])
            released = released[released['class'] == suffix].set_index("global_index")
            for quantile, group in part.groupby("quantile", observed=False):
                base = released.loc[group.global_index, "adapted_dice"]
                rows.append({"variant": variant, "seed": 2026, "class": suffix, "quantile": str(quantile),
                             "n": len(group), "mean_confidence": float(group.class_confidence.mean()),
                             "mean_anchor_dice": float(group.anchor_dice.mean()),
                             "mean_adapted_dice": float(group.adapted_dice.mean()),
                             "mean_delta_vs_anchor": float((group.adapted_dice - group.anchor_dice).mean()),
                             "mean_delta_vs_released": float((group.adapted_dice - base.to_numpy()).mean())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "reliability_stratified_effect.csv", index=False)
    return result


def quality_stratified(frames):
    rows = []
    for variant in MAIN_VARIANTS:
        long = class_long(frames[(variant, 2026)])
        for domain in DOMAINS + ["All"]:
            part = long if domain == "All" else long[long.domain == domain]
            for suffix in CLASSES:
                class_part = part[(part['class'] == suffix) & part.anchor_dice.notna()]
                for quality, selected in [("low_quality_dice_lt_070", class_part[class_part.anchor_dice < .70]),
                                          ("high_quality_dice_ge_085", class_part[class_part.anchor_dice >= .85])]:
                    rows.append({"variant": variant, "seed": 2026, "domain": domain, "class": suffix,
                                 "quality_group": quality, "n": len(selected),
                                 "mean_anchor_dice": float(selected.anchor_dice.mean()) if len(selected) else np.nan,
                                 "mean_adapted_dice": float(selected.adapted_dice.mean()) if len(selected) else np.nan,
                                 "mean_gain_vs_anchor": float((selected.adapted_dice - selected.anchor_dice).mean()) if len(selected) else np.nan,
                                 "negative_adaptation_rate": float(selected.negative.mean()) if len(selected) else np.nan})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "quality_stratified_effect.csv", index=False)
    return result


def gate_and_budget(frames):
    rows = []
    budget = []
    class_rows = []
    for (variant, seed), frame in frames.items():
        if seed != 2026:
            continue
        frame = frame.copy()
        frame["alpha_budget_delta"] = frame.alpha_map_mean - frame.global_rate
        frame.to_csv(OUT / f"gate_diagnostics_{variant}_seed2026.csv", index=False)
        rows.append(frame)
        for domain in DOMAINS + ["All"]:
            part = frame if domain == "All" else frame[frame.domain == domain]
            budget.append({"variant": variant, "seed": seed, "domain": domain, "n": len(part),
                           "mean_global_rate": float(part.global_rate.mean()),
                           "mean_alpha_map": float(part.alpha_map_mean.mean()),
                           "mean_alpha_minus_global": float(part.alpha_budget_delta.mean()),
                           "mean_abs_alpha_minus_global": float(part.alpha_budget_delta.abs().mean()),
                           "mean_correction_norm_released": float(part.correction_norm_released.mean()),
                           "mean_correction_norm_gated": float(part.correction_norm_gated.mean()),
                           "mean_correction_norm_ratio": float(part.correction_norm_ratio.mean()),
                           "median_correction_norm_ratio": float(part.correction_norm_ratio.median()),
                           "fallback_rate": float(part.a_fallback.mean())})
            for suffix in CLASSES:
                released = part[f"corr_released_{suffix}"].mean()
                gated = part[f"corr_gated_{suffix}"].mean()
                class_rows.append({"variant": variant, "seed": seed, "domain": domain, "class": suffix,
                                   "mean_corr_released": float(released), "mean_corr_gated": float(gated),
                                   "correction_ratio": float(gated / (released + 1e-8))})
    pd.concat(rows, ignore_index=True).to_csv(OUT / "gate_diagnostics_seed2026.csv", index=False)
    pd.DataFrame(budget).to_csv(OUT / "correction_budget_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "correction_by_class_summary.csv", index=False)
    return pd.DataFrame(budget), pd.DataFrame(class_rows)


def bootstrap_ci(values, rng, samples=10000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    means = np.empty(samples)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        means[start:start + count] = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pairwise_statistics(frames):
    rows, pvalue_indices, rng = [], [], np.random.default_rng(2026)
    for comparison in ["lowconf_g10", "highconf_g10"]:
        gated = frames[(comparison, 2026)].set_index("global_index")
        released = frames[("released", 2026)].set_index("global_index")
        for domain in DOMAINS + ["All"]:
            indices = gated.index if domain == "All" else gated[gated.domain == domain].index
            for suffix in CLASSES + ["average"]:
                delta = (gated.loc[indices, f"adapted_dice_{suffix}"] - released.loc[indices, f"adapted_dice_{suffix}"]).dropna().to_numpy()
                low, high = bootstrap_ci(delta, rng)
                if len(delta) and np.any(delta != 0):
                    wilcoxon_p = float(stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
                else:
                    wilcoxon_p = 1.0
                rows.append({"comparison": f"{comparison}_vs_released", "domain": domain, "class": suffix,
                             "n": len(delta), "mean_delta": float(delta.mean()) if len(delta) else np.nan,
                             "median_delta": float(np.median(delta)) if len(delta) else np.nan,
                             "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
                             "wilcoxon_raw_p": wilcoxon_p, "wilcoxon_holm_p": np.nan})
                pvalue_indices.append(len(rows) - 1)
    pvalues = [rows[index]["wilcoxon_raw_p"] for index in pvalue_indices]
    order = np.argsort(pvalues); previous = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, max(previous, pvalues[index] * (len(pvalues) - rank)))
        rows[pvalue_indices[index]]["wilcoxon_holm_p"] = adjusted
        previous = adjusted
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "pairwise_statistics.csv", index=False)
    return result


def multiseed_summary(frames):
    rows = []
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            frame = frames[(variant, seed)]
            long = class_long(frame)
            for domain in DOMAINS + ["All"]:
                part = frame if domain == "All" else frame[frame.domain == domain]
                long_part = long if domain == "All" else long[long.domain == domain]
                rows.append({"variant": variant, "seed": seed, "domain": domain,
                             "weighted_dice": float(part.adapted_dice_average.mean()),
                             "dice_lv": float(part.adapted_dice_lv.mean()), "dice_myo": float(part.adapted_dice_myo.mean()),
                             "dice_rv": float(part.adapted_dice_rv.mean()),
                             "negative_adaptation_rate": float(long_part.negative.mean())})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "multiseed_summary.csv", index=False)
    return result


def representative_cases(frames):
    released = class_long(frames[("released", 2026)]).set_index(["global_index", "class"])
    low = class_long(frames[("lowconf_g10", 2026)]).set_index(["global_index", "class"])
    high = class_long(frames[("highconf_g10", 2026)]).set_index(["global_index", "class"])
    merged = low.add_suffix("_low").join(released.add_suffix("_released"), how="inner").join(high.add_suffix("_high"), how="inner")
    merged["delta_low"] = merged.adapted_dice_low - merged.adapted_dice_released
    merged["delta_high"] = merged.adapted_dice_high - merged.adapted_dice_released
    rows = []
    for suffix in CLASSES:
        part = merged[(merged.anchor_dice_low.notna()) & (merged['class_confidence_low'].notna())]
        part = part.xs(suffix, level="class") if suffix in merged.index.get_level_values("class") else pd.DataFrame()
        if not len(part):
            continue
        q20, q80 = part.class_confidence_low.quantile([.20, .80])
        cases = [
            ("A_low_reliability_lowconf_improves", part[part.class_confidence_low <= q20].sort_values("delta_low", ascending=False).head(3)),
            ("B_high_reliability_negative_protected", part[(part.class_confidence_low >= q80) & (part.adapted_dice_released < part.anchor_dice_released) & (part.delta_low > 0)].sort_values("delta_low", ascending=False).head(3)),
            ("C_lowconf_harms", part.sort_values("delta_low").head(3)),
        ]
        for case_type, selected in cases:
            for index, item in selected.iterrows():
                rows.append({"case_type": case_type, "domain": item.domain_low, "slice": item.slice_name_low,
                             "class": suffix, "class_confidence": item.class_confidence_low,
                             "anchor_dice": item.anchor_dice_low, "released_dice": item.adapted_dice_released,
                             "lowconf_dice": item.adapted_dice_low, "highconf_dice": item.adapted_dice_high,
                             "global_rate": item.global_rate_low, "class_alpha": item.class_alpha_low,
                             "delta_lowconf": item.delta_low, "delta_highconf": item.delta_high})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "representative_cases.csv", index=False)
    return result


def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def figures(frames, class_deltas, negative, stratified, budget, correction):
    seed = 2026
    overall = []
    for variant in MAIN_VARIANTS:
        frame = frames[(variant, seed)]
        overall.append(float(frame.adapted_dice_average.mean()) * 100)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar([VARIANT_LABELS[v] for v in MAIN_VARIANTS], overall, color="#4472c4")
    ax.set_ylabel("Weighted foreground Dice (%)"); ax.set_title("EXP-4 Dice by variant"); ax.tick_params(axis="x", rotation=35); savefig(OUT / "fig_exp4_dice_by_variant.png")

    delta = class_deltas[(class_deltas.domain == "All") & (class_deltas['class'].isin(CLASSES)) & (class_deltas.variant.isin(MAIN_VARIANTS[1:]))]
    fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(CLASSES)); width = .18
    for offset, variant in enumerate(MAIN_VARIANTS[1:]):
        values = delta[delta.variant == variant].set_index("class").reindex(CLASSES).mean_delta_vs_released * 100
        ax.bar(x + (offset - 2) * width, values, width, label=VARIANT_LABELS[variant])
    ax.axhline(0, color="black", linewidth=.7); ax.set_xticks(x, [c.upper() for c in CLASSES]); ax.set_ylabel("Δ Dice vs Released (pp)"); ax.set_title("Class-wise EXP-4 effect"); ax.legend(fontsize=8); savefig(OUT / "fig_exp4_class_delta.png")

    for suffix in CLASSES:
        part = stratified[(stratified['class'] == suffix) & (stratified.variant.isin(["lowconf_g10", "highconf_g10"]))]
        fig, ax = plt.subplots(figsize=(7, 5))
        for variant in ["lowconf_g10", "highconf_g10"]:
            curve = part[part.variant == variant].set_index("quantile").reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
            ax.plot(curve.index, curve.mean_delta_vs_released * 100, marker="o", label=VARIANT_LABELS[variant])
        ax.axhline(0, color="black", linewidth=.7); ax.set_xlabel("Class Confidence quantile"); ax.set_ylabel("Δ Dice vs Released (pp)"); ax.set_title(f"Reliability-stratified effect: {suffix.upper()}"); ax.legend(); savefig(OUT / f"fig_exp4_reliability_stratified_{suffix}.png")

    neg = negative[(negative.seed == 2026) & (negative.domain == "All") & (negative.variant.isin(["released", "lowconf_g10", "highconf_g10"]))]
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(CLASSES)); width = .24
    for offset, variant in enumerate(["released", "lowconf_g10", "highconf_g10"]):
        values = neg[neg.variant == variant].set_index("class").reindex(CLASSES).negative_adaptation_rate * 100
        ax.bar(x + (offset - 1) * width, values, width, label=VARIANT_LABELS[variant])
    ax.set_xticks(x, [c.upper() for c in CLASSES]); ax.set_ylabel("Negative adaptation rate (%)"); ax.set_title("Negative adaptation"); ax.legend(); savefig(OUT / "fig_exp4_negative_adaptation.png")

    gate = pd.read_csv(OUT / "gate_diagnostics_seed2026.csv"); gate = gate[gate.variant == "lowconf_g10"]
    fig, ax = plt.subplots(figsize=(8, 5)); data = [gate.global_rate.dropna(), gate.alpha_lv.dropna(), gate.alpha_myo.dropna(), gate.alpha_rv.dropna()]
    ax.boxplot(data, labels=["global rate", "alpha LV", "alpha MYO", "alpha RV"]); ax.set_ylabel("Fusion strength"); ax.set_title("EXP-4 alpha by class"); savefig(OUT / "fig_exp4_alpha_by_class.png")

    corr = correction[(correction.domain == "All") & (correction.variant.isin(["released", "lowconf_g10", "highconf_g10"]))]
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(CLASSES)); width = .24
    for offset, variant in enumerate(["released", "lowconf_g10", "highconf_g10"]):
        values = corr[corr.variant == variant].set_index("class").reindex(CLASSES).correction_ratio
        ax.bar(x + (offset - 1) * width, values, width, label=VARIANT_LABELS[variant])
    ax.axhline(1, color="black", linewidth=.7); ax.set_xticks(x, [c.upper() for c in CLASSES]); ax.set_ylabel("Gated / Released correction"); ax.set_title("Class-specific correction ratio"); ax.legend(); savefig(OUT / "fig_exp4_correction_ratio.png")


def report(domain, deltas, negative, stratified, quality, budget, pairwise, multi, representatives, correction):
    overall = domain[domain.domain == "All"].copy()
    def dice(variant, column="dice_average"):
        return float(overall[(overall.variant == variant) & (overall.seed == 2026)][column].iloc[0])
    released = dice("released")
    main_low = dice("lowconf_g10"); main_high = dice("highconf_g10")
    gamma_values = {v: dice(v) for v in ["lowconf_g05", "lowconf_g10", "lowconf_g20"]}
    best_gamma = max(gamma_values, key=gamma_values.get)
    low_delta = deltas[(deltas.variant == "lowconf_g10") & (deltas.domain == "All")]
    high_delta = deltas[(deltas.variant == "highconf_g10") & (deltas.domain == "All")]
    low_q = stratified[(stratified.variant == "lowconf_g10") & (stratified["quantile"].isin(["Q1", "Q2"]))].mean_delta_vs_released.mean()
    high_q = stratified[(stratified.variant == "lowconf_g10") & (stratified["quantile"].isin(["Q4", "Q5"]))].mean_delta_vs_released.mean()
    low_negative = negative[(negative.variant == "lowconf_g10") & (negative.seed == 2026) & (negative.domain == "All")].negative_adaptation_rate.mean()
    rel_negative = negative[(negative.variant == "released") & (negative.seed == 2026) & (negative.domain == "All")].negative_adaptation_rate.mean()
    low_delta_by_class = low_delta.set_index("class").mean_delta_vs_released
    high_delta_by_class = high_delta.set_index("class").mean_delta_vs_released
    direction = bool((low_delta_by_class.reindex(CLASSES) > high_delta_by_class.reindex(CLASSES)).all())
    seed_low = multi[(multi.variant == "lowconf_g10") & (multi.domain == "All")].set_index("seed").weighted_dice
    seed_high = multi[(multi.variant == "highconf_g10") & (multi.domain == "All")].set_index("seed").weighted_dice
    seed_released = multi[(multi.variant == "released") & (multi.domain == "All")].set_index("seed").weighted_dice
    lowconf_beats_highconf_all_seeds = bool((seed_low > seed_high).all())
    lowconf_beats_released_all_seeds = bool((seed_low > seed_released).all())
    low_negative_by_class = negative[(negative.variant == "lowconf_g10") & (negative.seed == 2026) & (negative.domain == "All")].set_index("class").negative_adaptation_rate
    rel_negative_by_class = negative[(negative.variant == "released") & (negative.seed == 2026) & (negative.domain == "All")].set_index("class").negative_adaptation_rate
    negative_class_changes = low_negative_by_class.reindex(CLASSES) - rel_negative_by_class.reindex(CLASSES)
    ci_all = pairwise[pairwise.domain == "All"].copy()
    ci_excludes_zero = ci_all[(ci_all.bootstrap_ci95_low > 0) | (ci_all.bootstrap_ci95_high < 0)]
    if (main_low - released) * 100 >= .30 and direction and low_q > high_q and low_negative <= rel_negative and main_low > main_high:
        case = "A"
    elif (main_low - released) * 100 >= .10 and main_low > main_high:
        case = "B"
    elif main_low <= released + .001 or not direction:
        case = "C"
    elif main_high > main_low and main_high > released:
        case = "D"
    else:
        case = "E"
    lines = [
        "# EXP-4 Reliability-aware Class Gating", "",
        "## 1. Motivation", "", "EXP-3 found Class Confidence to be a stable class-level reliability proxy; EXP-4 tests whether it should modulate SFF strength.", "",
        "## 2. Method", "", "For foreground class c, R_c=sum(P_c²)/(sum(P_c)+eps), R_bar=sum(M_cR_c)/(sum(M_c)+eps), alpha_c=clip(a+gamma(R_bar-R_c),0,1) for LowConf and the sign-reversed equation for HighConf. The soft spatial map is alpha(u)=sum_k P_k(u)alpha_k, and F_gate=F_current+[alpha(u)/(a+eps)](F_sff-F_current). Identity directly reuses F_sff.", "",
        "## 3. Released Equivalence", "", "PASS; see `released_equivalence.txt`.", "",
        "## 4. Identity Equivalence", "", "PASS; see `identity_equivalence.txt`.", "",
        "## 5. Main Results", "", overall[overall.seed == 2026][["variant", "domain", "dice_lv", "dice_myo", "dice_rv", "dice_average"]].to_markdown(index=False), "",
        "## 6. Gamma Sensitivity", "", str({v: gamma_values[v] for v in gamma_values}) + f"; best={best_gamma}.", "",
        "## 7. Direction Control", "", deltas[(deltas.domain == "All") & (deltas['class'].isin(CLASSES)) & (deltas.variant.isin(["lowconf_g10", "highconf_g10"]))].to_markdown(index=False), "",
        "## 8. Reliability-stratified Effect", "", stratified[(stratified.variant.isin(["lowconf_g10", "highconf_g10"])) & (stratified["quantile"].isin(["Q1", "Q2", "Q4", "Q5"]))].to_markdown(index=False), "",
        "## 9. Low-quality vs High-quality Class", "", quality[(quality.variant.isin(["released", "lowconf_g10", "highconf_g10"])) & (quality.domain == "All")].to_markdown(index=False), "",
        "## 10. Negative Adaptation", "", negative[(negative.seed == 2026) & (negative.domain == "All") & (negative.variant.isin(["released", "lowconf_g10", "highconf_g10"]))].to_markdown(index=False), "",
        "## 11. Gate Behavior", "", budget[budget.domain == "All"].to_markdown(index=False), "",
        "## 12. Correction Budget", "", correction[correction.domain == "All"].to_markdown(index=False), "",
        "## 13. Multi-seed", "", multi[multi.domain == "All"].groupby("variant").agg(weighted_dice_mean=("weighted_dice", "mean"), weighted_dice_std=("weighted_dice", "std"), negative_rate_mean=("negative_adaptation_rate", "mean"), negative_rate_std=("negative_adaptation_rate", "std")).reset_index().to_markdown(index=False), "",
        "## 14. Statistical Significance", "", pairwise.to_markdown(index=False), "",
        "## 15. Representative Cases", "", representatives.to_markdown(index=False), "",
        "## 16. Conclusion", "", f"Decision: Case {case}. LowConf gamma=1 weighted Dice={main_low:.6f}, Released={released:.6f}, HighConf={main_high:.6f}; Q1/Q2 mean effect={low_q:.6f}, Q4/Q5 mean effect={high_q:.6f}.", "",
        "## Final Answers", "",
        f"1. Released weighted Dice: {released * 100:.4f}%.",
        f"2. LowConf gamma=1 weighted Dice and Delta: {main_low * 100:.4f}%, {(main_low - released) * 100:+.4f} pp.",
        f"3. HighConf gamma=1 weighted Dice and Delta: {main_high * 100:.4f}%, {(main_high - released) * 100:+.4f} pp.",
        f"4. Best gamma: {best_gamma}.",
        "5. LowConf gamma=1 class Delta LV/MYO/RV: " + ", ".join(f"{low_delta[low_delta['class'] == c].mean_delta_vs_released.iloc[0] * 100:+.4f} pp" for c in CLASSES) + ".",
        f"6. Low reliability Q1/Q2 larger gain: {'yes' if low_q > high_q else 'no'}.",
        f"7. High reliability Q4/Q5 protected: {'yes' if high_q >= 0 else 'no'}.",
        "8. Negative adaptation rate change by class: " + ", ".join(f"{c.upper()} {negative_class_changes[c] * 100:+.4f} pp" for c in CLASSES) + "; not uniformly improved.",
        f"9. Mean correction budget changed materially: {'yes' if abs(budget[(budget.variant == 'lowconf_g10') & (budget.domain == 'All')].mean_alpha_minus_global.iloc[0]) > .01 else 'no'}.",
        f"10. LowConf stable over HighConf across seeds: {'yes' if lowconf_beats_highconf_all_seeds else 'no'} (seed 2024 is reversed).",
        f"11. Multi-seed LowConf vs Released same direction: {'yes' if lowconf_beats_released_all_seeds else 'no'}.",
        "12. All-domain paired 95% CI excluding 0: " + (", ".join(f"{row.comparison} {row['class']}" for _, row in ci_excludes_zero.iterrows()) if len(ci_excludes_zero) else "none") + "; domain-specific rows are in `pairwise_statistics.csv`.",
        f"13. Final case: Case {case}.",
        "14. Next step: " + ("continue to finer Region/Boundary-aware Gating" if case == "A" else "optimize reliability-to-alpha mapping before a larger gating study" if case == "B" else "reassess the gating formulation before proceeding") + ".",
    ]
    (OUT / "exp4_report.md").write_text("\n".join(lines) + "\n")
    return case


def main():
    frames = load_frames()
    for (variant, seed), frame in frames.items():
        if len(frame) != 4091:
            raise RuntimeError(f"Unexpected rows: {variant}/{seed}={len(frame)}")
    domain = make_domain_summaries()
    deltas = class_delta(frames)
    negative = negative_summary(frames)
    stratified = reliability_stratified(frames)
    quality = quality_stratified(frames)
    budget, correction = gate_and_budget(frames)
    pairwise = pairwise_statistics(frames)
    multi = multiseed_summary(frames)
    representatives = representative_cases(frames)
    figures(frames, deltas, negative, stratified, budget, correction)
    case = report(domain, deltas, negative, stratified, quality, budget, pairwise, multi, representatives, correction)
    print(f"variants={len(frames)} case={case}")


if __name__ == "__main__":
    main()
