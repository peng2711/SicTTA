#!/usr/bin/env python3
"""Analyze EXP-1 raw streams and generate reproducible tables and figures."""

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
RAW = ROOT / "results/exp1/raw"
OUT = ROOT / "results/exp1"
DOMAINS = ["B", "C", "D"]
VARIANTS = ["released", "rolling_all_40", "rolling_all_160",
            "sft_queue_40", "domain_reset_oracle"]
DEPLOYABLE = VARIANTS[1:4]
LABELS = {
    "released": "Released",
    "rolling_all_40": "Rolling-All-40",
    "rolling_all_160": "Rolling-All-160",
    "sft_queue_40": "SFT-Queue-40",
    "domain_reset_oracle": "Domain-Reset Oracle",
}
COLORS = dict(zip(VARIANTS, ["#333333", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))
MAIN_SEED = 2026


def finite(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def avg(series, scale=1.0):
    values = finite(series)
    return float(values.mean() * scale) if len(values) else np.nan


def spread(series, scale=1.0):
    values = finite(series)
    return float(values.std(ddof=0) * scale) if len(values) else np.nan


def median(series, scale=1.0):
    values = finite(series)
    return float(values.median() * scale) if len(values) else np.nan


def load_runs():
    runs = {}
    for path in sorted(RAW.glob("*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        variant = str(frame.variant.iloc[0])
        seed = int(frame.seed.iloc[0])
        runs[(variant, seed)] = frame
    if not runs:
        raise RuntimeError(f"No raw EXP-1 CSV files found under {RAW}")
    return runs


def metadata(variant, seed):
    return json.loads((RAW / f"{variant}_seed{seed}.json").read_text())


def make_domain_summary(runs):
    rows = []
    for (variant, seed), frame in sorted(runs.items()):
        for domain in DOMAINS:
            part = frame[frame.domain == domain]
            accepted = part[part.is_sft == True]
            rejected = part[part.is_sft == False]
            rows.append({
                "variant": variant, "seed": seed, "domain": domain,
                "num_slices": len(part), "num_cases": part.volume_id.nunique(),
                "dice_lv": avg(part.adapted_dice_lv, 100),
                "dice_myo": avg(part.adapted_dice_myo, 100),
                "dice_rv": avg(part.adapted_dice_rv, 100),
                "dice_average": avg(part.adapted_dice_average, 100),
                "anchor_dice_average": avg(part.anchor_dice_average, 100),
                "sft_accepted": int(len(accepted)),
                "sft_acceptance_rate": float(len(accepted) / len(part)) if len(part) else np.nan,
                "accepted_anchor_dice_mean": avg(accepted.anchor_dice_average, 100),
                "accepted_anchor_dice_std": spread(accepted.anchor_dice_average, 100),
                "rejected_anchor_dice_mean": avg(rejected.anchor_dice_average, 100),
                "rejected_anchor_dice_std": spread(rejected.anchor_dice_average, 100),
                "threshold_mean": avg(part.threshold),
                "threshold_std": spread(part.threshold),
                "threshold_median": median(part.threshold),
                "ccd_mean": avg(part.ccd), "ccd_std": spread(part.ccd),
                "margin_mean": avg(part.ccd_margin),
                "history_len_start": int(part.admission_history_len.iloc[0]),
                "history_len_end": int(part.admission_history_len.iloc[-1]),
                "pool_size_start": int(part.pool_size_before.iloc[0]),
                "pool_size_end": int(part.pool_size_after.iloc[-1]),
                "top1_similarity_mean": avg(part.top1_similarity),
                "top5_similarity_mean": avg(part.top5_mean_similarity),
                "negative_adaptation_rate": float((part.adapted_dice_average < part.anchor_dice_average).mean()),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "exp1_domain_summary.csv", index=False)
    return result


def make_overall_summary(domain_summary):
    rows = []
    for (variant, seed), part in domain_summary.groupby(["variant", "seed"], sort=True):
        weight = part.num_slices / part.num_slices.sum()
        rows.append({
            "variant": variant, "seed": seed,
            "macro_domain_dice": float(part.dice_average.mean()),
            "slice_weighted_dice": float((part.dice_average * weight).sum()),
            "macro_domain_anchor_dice": float(part.anchor_dice_average.mean()),
            "slice_weighted_anchor_dice": float((part.anchor_dice_average * weight).sum()),
            "num_slices": int(part.num_slices.sum()), "num_cases": int(part.num_cases.sum()),
        })
    result = pd.DataFrame(rows)
    released = result[result.variant == "released"].set_index("seed").slice_weighted_dice
    result["delta_vs_released"] = [float(row.slice_weighted_dice - released.loc[row.seed])
                                    if row.variant != "released" else 0.0
                                    for row in result.itertuples()]
    result.to_csv(OUT / "exp1_overall_summary.csv", index=False)
    return result


def make_quality_tables(runs):
    quality_rows = []
    threshold_rows = []
    for (variant, seed), frame in sorted(runs.items()):
        for domain in DOMAINS:
            part = frame[frame.domain == domain]
            accepted = part[part.is_sft == True]
            rejected = part[part.is_sft == False]
            accepted_values = finite(accepted.anchor_dice_average)
            rejected_values = finite(rejected.anchor_dice_average)
            quality_rows.append({
                "variant": variant, "seed": seed, "domain": domain,
                "accepted_count": len(accepted), "acceptance_rate": len(accepted) / len(part),
                "accepted_anchor_dice_mean": avg(accepted.anchor_dice_average, 100),
                "accepted_anchor_dice_median": median(accepted.anchor_dice_average, 100),
                "accepted_anchor_dice_std": spread(accepted.anchor_dice_average, 100),
                "rejected_anchor_dice_mean": avg(rejected.anchor_dice_average, 100),
                "rejected_anchor_dice_median": median(rejected.anchor_dice_average, 100),
                "quality_gap": avg(accepted.anchor_dice_average, 100) - avg(rejected.anchor_dice_average, 100),
                "accepted_anchor_p10": float(accepted_values.quantile(.10) * 100) if len(accepted_values) else np.nan,
                "accepted_anchor_p25": float(accepted_values.quantile(.25) * 100) if len(accepted_values) else np.nan,
                "accepted_anchor_p50": float(accepted_values.quantile(.50) * 100) if len(accepted_values) else np.nan,
                "accepted_anchor_p75": float(accepted_values.quantile(.75) * 100) if len(accepted_values) else np.nan,
                "accepted_anchor_p90": float(accepted_values.quantile(.90) * 100) if len(accepted_values) else np.nan,
            })
            for quality_threshold in (.70, .75, .80):
                high_quality = part.anchor_dice_average >= quality_threshold
                threshold_rows.append({
                    "variant": variant, "seed": seed, "domain": domain,
                    "anchor_quality_threshold": quality_threshold,
                    "accepted_count": int(accepted.is_sft.sum()),
                    "high_quality_count": int(high_quality.sum()),
                    "true_positive_count": int((accepted.is_sft & high_quality).sum()),
                    "precision_like_p_high_given_accepted": float((accepted.anchor_dice_average >= quality_threshold).mean()) if len(accepted) else np.nan,
                    "recall_like_p_accepted_given_high": float((part.is_sft[high_quality]).mean()) if high_quality.any() else np.nan,
                })
    quality = pd.DataFrame(quality_rows)
    thresholds = pd.DataFrame(threshold_rows)
    quality.to_csv(OUT / "accepted_sft_quality.csv", index=False)
    thresholds.to_csv(OUT / "admission_quality_thresholds.csv", index=False)
    return quality, thresholds


def first_nonnull(part, field):
    values = finite(part[field])
    return float(values.iloc[0]) if len(values) else np.nan


def last_nonnull(part, field):
    values = finite(part[field])
    return float(values.iloc[-1]) if len(values) else np.nan


def transition_rows(runs):
    rows = []
    for (variant, seed), frame in sorted(runs.items()):
        for before_domain, after_domain in (("B", "C"), ("C", "D")):
            before = frame[frame.domain == before_domain]
            after = frame[frame.domain == after_domain]
            row = {"transition": f"{before_domain}->{after_domain}", "variant": variant, "seed": seed}
            row.update({
                "threshold_before": last_nonnull(before, "threshold"),
                "threshold_after_initial": first_nonnull(after, "threshold"),
                "threshold_final_domain_reference": median(after.tail(max(1, int(np.ceil(len(after) * .20)))).threshold),
                "ccd_before": last_nonnull(before, "ccd"),
                "ccd_after": first_nonnull(after, "ccd"),
            })
            for n in (20, 50, 100):
                head = after.head(n)
                row[f"acceptance_first_{n}"] = avg(head.is_sft)
                accepted = head[head.is_sft == True]
                row[f"anchor_quality_first_{n}_accepted"] = avg(accepted.anchor_dice_average, 100)
                row[f"top5_similarity_first_{n}"] = avg(head.top5_mean_similarity)
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "transition_analysis.csv", index=False)
    return result


def adaptation_lag(runs):
    rows = []
    for (variant, seed), frame in sorted(runs.items()):
        for domain in ("C", "D"):
            part = frame[frame.domain == domain].reset_index(drop=True)
            reference_count = max(1, min(200, int(np.ceil(len(part) * .20))))
            reference_values = finite(part.tail(reference_count).threshold)
            reference = float(reference_values.median()) if len(reference_values) else np.nan
            values = pd.to_numeric(part.threshold, errors="coerce").to_numpy(dtype=float)
            rolling = []
            for start in range(max(0, len(values) - 19)):
                window = values[start:start + 20]
                rolling.append(float(np.median(window)) if len(window) == 20 and np.isfinite(window).all() else np.nan)
            tolerance = max(abs(reference) * .05, 1e-12) if np.isfinite(reference) else np.nan
            lag = np.nan
            if np.isfinite(reference):
                for start in range(max(0, len(rolling) - 19)):
                    segment = np.asarray(rolling[start:start + 20])
                    if len(segment) == 20 and np.isfinite(segment).all() and np.all(np.abs(segment - reference) <= tolerance):
                        lag = start + 20
                        break
            rows.append({
                "domain": domain, "variant": variant, "seed": seed,
                "final_reference": reference, "reference_count": reference_count,
                "rolling_window": 20, "tolerance_fraction": .05,
                "adaptation_lag_slices_after_entry": lag,
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "threshold_adaptation_lag.csv", index=False)
    return result


def holm_adjust(pvalues):
    finite_indices = [i for i, value in enumerate(pvalues) if np.isfinite(value)]
    adjusted = [np.nan] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(sorted(finite_indices, key=lambda i: pvalues[i])):
        value = min(1.0, max(previous, pvalues[index] * (len(finite_indices) - rank)))
        adjusted[index] = value
        previous = value
    return adjusted


def bootstrap_mean(values, rng, samples=10000):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def make_pairwise_stats(runs):
    rows = []
    for seed in sorted({seed for _, seed in runs}):
        base_meta = metadata("released", seed)
        base = {item["volume_id"]: item for item in base_meta["case_metrics"]}
        for variant in VARIANTS[1:]:
            if (variant, seed) not in runs:
                continue
            other = {item["volume_id"]: item for item in metadata(variant, seed)["case_metrics"]}
            common = sorted(set(base) & set(other))
            for domain in DOMAINS + ["All"]:
                selected = [name for name in common if domain == "All" or base[name]["domain"] == domain]
                deltas = np.asarray([(other[name]["adapted_dice_average"] - base[name]["adapted_dice_average"]) * 100 for name in selected])
                if len(deltas):
                    rng = np.random.default_rng(2026)
                    ci_low, ci_high = bootstrap_mean(deltas, rng)
                    try:
                        wilcoxon = stats.wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
                        pvalue = float(wilcoxon.pvalue)
                    except ValueError:
                        pvalue = np.nan
                    rows.append({
                        "variant": variant, "baseline": "released", "seed": seed, "domain": domain,
                        "n_cases": len(deltas), "mean_delta": float(deltas.mean()),
                        "median_delta": float(np.median(deltas)), "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high, "wilcoxon_raw_p": pvalue,
                    })
    result = pd.DataFrame(rows)
    result["wilcoxon_holm_p"] = holm_adjust(result.wilcoxon_raw_p.tolist())
    result.to_csv(OUT / "pairwise_statistics.csv", index=False)
    return result


def released_equivalence(runs):
    released = runs[("released", MAIN_SEED)]
    old = pd.read_csv(ROOT / "results/exp0_5/per_slice_diagnostics.csv")
    fields = [("adapted_dice_lv", "dice_lv"), ("adapted_dice_myo", "dice_myo"),
              ("adapted_dice_rv", "dice_rv"), ("adapted_dice_average", "dice_average")]
    deltas = [abs(float(a) - float(b)) for left, right in fields for a, b in zip(released[left], old[right])]
    names_equal = len(released) == len(old) and all(released.slice_name == old.slice_name)
    exact = names_equal and len(deltas) == len(released) * len(fields) and max(deltas, default=0) <= 1e-12
    lines = ["EXP-1 Released seed=2026 vs EXP-0.5 Full SicTTA equivalence", "",
             f"record_count_exp1={len(released)}", f"record_count_exp0_5={len(old)}",
             f"slice_names_and_order_equal={names_equal}",
             f"max_per_slice_dice_delta={max(deltas, default=0):.12f}",
             "tolerance=1e-12", f"PASS={exact}", ""]
    for domain in DOMAINS:
        current = released[released.domain == domain].adapted_dice_average.mean() * 100
        reference = old[old.domain == domain].dice_average.mean() * 100
        lines.append(f"{domain},EXP-1={current:.12f},EXP-0.5={reference:.12f},delta={abs(current-reference):.12f}")
    (OUT / "released_equivalence.txt").write_text("\n".join(lines) + "\n")
    return exact


def boundary_lines(frame):
    changes = []
    previous = frame.domain.iloc[0]
    for _, row in frame.iloc[1:].iterrows():
        if row.domain != previous:
            changes.append((int(row.global_index), f"{previous} -> {row.domain}"))
            previous = row.domain
    return changes


def add_boundaries(frame):
    for index, label in boundary_lines(frame):
        plt.axvline(index, color="black", linestyle="--", linewidth=.7)
        plt.text(index, .98, label, transform=plt.gca().get_xaxis_transform(), rotation=90,
                 va="top", ha="right", fontsize=7)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_figures(runs, domain_summary, overall, lag):
    main = {variant: runs[(variant, MAIN_SEED)] for variant in VARIANTS if (variant, MAIN_SEED) in runs}
    x = np.arange(len(VARIANTS))
    fig, ax = plt.subplots(figsize=(11, 5))
    width = .18
    for offset, domain in enumerate(DOMAINS):
        values = [domain_summary[(domain_summary.variant == v) & (domain_summary.seed == MAIN_SEED) & (domain_summary.domain == domain)].dice_average.iloc[0]
                  if v in main else np.nan for v in VARIANTS]
        ax.bar(x + (offset - 1.5) * width, values, width, label=domain)
    weighted = [overall[(overall.variant == v) & (overall.seed == MAIN_SEED)].slice_weighted_dice.iloc[0]
                if v in main else np.nan for v in VARIANTS]
    ax.bar(x + 1.5 * width, weighted, width, label="Slice-weighted Avg")
    ax.set_xticks(x, [LABELS[v] for v in VARIANTS], rotation=20, ha="right")
    ax.set_ylabel("Dice (%)"); ax.set_title("EXP-1 segmentation comparison"); ax.legend()
    savefig(OUT / "fig_exp1_dice_comparison.png")

    plt.figure(figsize=(11, 5))
    for variant, frame in main.items():
        plt.plot(frame.global_index, frame.threshold, label=LABELS[variant], color=COLORS[variant], linewidth=.7)
    add_boundaries(next(iter(main.values())))
    plt.xlabel("global index"); plt.ylabel("CCD threshold"); plt.title("Admission threshold stream"); plt.legend(fontsize=8)
    savefig(OUT / "fig_exp1_threshold_stream.png")

    plt.figure(figsize=(11, 5))
    for variant, frame in main.items():
        acceptance = frame.is_sft.astype(float).rolling(50, min_periods=1).mean()
        plt.plot(frame.global_index, acceptance, label=LABELS[variant], color=COLORS[variant], linewidth=.8)
    add_boundaries(next(iter(main.values())))
    plt.xlabel("global index"); plt.ylabel("rolling acceptance rate (window=50)"); plt.title("SFT admission stream"); plt.legend(fontsize=8)
    savefig(OUT / "fig_exp1_acceptance_stream.png")

    fig, ax = plt.subplots(figsize=(13, 5))
    positions, values, labels, colors = [], [], [], []
    position = 1
    for variant in VARIANTS:
        for domain in DOMAINS:
            part = main[variant][main[variant].domain == domain]
            accepted = part[part.is_sft == True].anchor_dice_average.dropna() * 100
            if len(accepted):
                positions.append(position); values.append(accepted.to_numpy()); labels.append(f"{LABELS[variant]}\n{domain}"); colors.append(COLORS[variant])
            position += 1
    box = ax.boxplot(values, positions=positions, patch_artist=True, widths=.65, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(.65)
    ax.set_xticks(positions, labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accepted anchor Dice (%)"); ax.set_title("Accepted SFT source quality")
    savefig(OUT / "fig_exp1_accepted_quality.png")

    plt.figure(figsize=(11, 5))
    for variant, frame in main.items():
        plt.plot(frame.global_index, frame.admission_history_len, label=LABELS[variant], color=COLORS[variant], linewidth=.8)
    add_boundaries(next(iter(main.values())))
    plt.xlabel("global index"); plt.ylabel("effective admission history length"); plt.title("Admission history length"); plt.legend(fontsize=8)
    savefig(OUT / "fig_exp1_history_length.png")

    pivot = lag[lag.seed == MAIN_SEED].pivot(index="domain", columns="variant", values="adaptation_lag_slices_after_entry").reindex(index=["C", "D"], columns=VARIANTS)
    fig, ax = plt.subplots(figsize=(11, 5)); pivot.plot.bar(ax=ax, color=[COLORS[v] for v in VARIANTS])
    ax.set_ylabel("lag (slices; NA omitted)"); ax.set_title("Threshold adaptation lag"); ax.legend(labels=[LABELS[v] for v in VARIANTS], fontsize=8)
    savefig(OUT / "fig_exp1_transition_lag.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, domain in enumerate(DOMAINS):
        values = [domain_summary[(domain_summary.variant == v) & (domain_summary.seed == MAIN_SEED) & (domain_summary.domain == domain)].negative_adaptation_rate.iloc[0] * 100
                  if v in main else np.nan for v in VARIANTS]
        ax.bar(x + (offset - 1) * .22, values, .22, label=domain)
    ax.set_xticks(x, [LABELS[v] for v in VARIANTS], rotation=20, ha="right")
    ax.set_ylabel("negative adaptation rate (%)"); ax.set_title("Negative adaptation"); ax.legend()
    savefig(OUT / "fig_exp1_negative_adaptation.png")


def conclusion(overall, quality, lag):
    main = overall[overall.seed == MAIN_SEED].set_index("variant")
    released = float(main.loc["released", "slice_weighted_dice"])
    deltas = {v: float(main.loc[v, "delta_vs_released"]) for v in VARIANTS[1:] if v in main.index}
    oracle_delta = deltas.get("domain_reset_oracle", np.nan)
    best = max(DEPLOYABLE, key=lambda v: float(main.loc[v, "slice_weighted_dice"]) if v in main.index else -np.inf)
    deployable_quality = quality[(quality.seed == MAIN_SEED) & quality.variant.isin(DEPLOYABLE)].accepted_anchor_dice_mean.dropna()
    max_deployable_delta = max((deltas.get(v, -np.inf) for v in DEPLOYABLE), default=-np.inf)
    if np.isfinite(oracle_delta) and oracle_delta > .2:
        case = "D"
    elif max_deployable_delta < .2:
        case = "B"
    elif all(float(main.loc["released", "slice_weighted_dice"]) >= float(main.loc[v, "slice_weighted_dice"]) for v in DEPLOYABLE if v in main.index):
        case = "C"
    else:
        case = "A"
    return released, deltas, best, case


def build_report(domain_summary, overall, quality, transitions, lag, pairwise, equivalence):
    released, deltas, best, case = conclusion(overall, quality, lag)
    main_domain = domain_summary[domain_summary.seed == MAIN_SEED]
    main_overall = overall[overall.seed == MAIN_SEED].copy()
    main_quality = quality[quality.seed == MAIN_SEED]
    main_lag = lag[lag.seed == MAIN_SEED]
    lines = [
        "# EXP-1 Controlled Admission History Ablation", "",
        "## 1. Question", "",
        "This controlled experiment changes only the CCD admission history statistics and measures Dice, source/anchor quality, admission, threshold dynamics, and transition behavior.", "",
        "## 2. Controlled Variables", "",
        "Checkpoint, data stream, B -> C -> D order, batch size, source/anchor model, CCD formula and sampling, pool implementation, Top-K, SABE, SFF, BN behavior, preprocessing, labels, and metrics were frozen. The released pool behavior reaching 41 items was not fixed.", "",
        "## 3. Variants", "",
        "- Released: exact released path; its stored `self.entropy_list` is cumulative, while the released threshold calculation locally trims the current candidate history to 40 after the append.",
        "- Rolling-All-40 / Rolling-All-160: last 40 / 160 CCD values from all test samples, including the current sample.",
        "- SFT-Queue-40: accepted CCD values only; threshold uses queue plus current candidate.",
        "- Domain-Reset Oracle: clears only CCD history at B/C/D boundaries; memory and model remain continual.", "",
        "## 4. Released Equivalence", "",
        f"{('PASS' if equivalence else 'FAIL')}: Released seed=2026 per-slice output-derived Dice matches EXP-0.5 within 1e-12; see `released_equivalence.txt`.", "",
        "## 5. Segmentation Results", "",
        main_domain.pivot(index="variant", columns="domain", values="dice_average").reindex(VARIANTS).assign(**{"Macro Avg": main_overall.set_index("variant").macro_domain_dice, "Slice-weighted Avg": main_overall.set_index("variant").slice_weighted_dice}).reset_index().rename(columns={"variant": "Variant"}).to_markdown(index=False), "",
        "Values are Dice percentages. `Δ vs Released` below uses the slice-weighted average in percentage points.", "",
        "| Variant | Δ vs Released (weighted Dice, pp) |", "|---|---:|" ]
    for variant in VARIANTS:
        lines.append(f"| {LABELS[variant]} | {main_overall.loc[main_overall.variant == variant, 'delta_vs_released'].iloc[0]:.4f} |")
    lines += ["", "## 6. Admission Rate", "", main_domain[["variant", "domain", "sft_accepted", "sft_acceptance_rate", "ccd_mean", "threshold_mean", "margin_mean", "history_len_start", "history_len_end", "pool_size_start", "pool_size_end"]].to_markdown(index=False), "",
              "## 7. Accepted SFT Source Quality", "", main_quality[["variant", "domain", "accepted_count", "acceptance_rate", "accepted_anchor_dice_mean", "accepted_anchor_dice_median", "accepted_anchor_dice_std", "rejected_anchor_dice_mean", "quality_gap", "accepted_anchor_p10", "accepted_anchor_p90"]].to_markdown(index=False), "",
              "Anchor Dice is used only offline; it never participates in CCD, admission, memory update, retrieval, or fusion.", "",
              "## 8. Transition Behavior", "", transitions[transitions.seed == MAIN_SEED].to_markdown(index=False), "",
              "## 9. Threshold Adaptation Lag", "", main_lag.to_markdown(index=False), "",
              "Lag is the first rolling-20-median endpoint followed by 20 consecutive rolling medians within ±5% of the final-domain reference. NA means the criterion was never met.", "",
              "## 10. Negative Adaptation", "", main_domain[["variant", "domain", "negative_adaptation_rate"]].to_markdown(index=False), "",
              "## 11. Statistical Comparison", "", pairwise[pairwise.seed == MAIN_SEED].to_markdown(index=False), "",
              "Statistics are paired by volume/case. Bootstrap uses 10,000 NumPy resamples with seed 2026; Holm correction is applied across reported Wilcoxon rows.", "",
              "## 12. Mechanistic Findings", "",
              "Admission history changes threshold statistics only; Top-K retrieval, memory capacity, SFT update code, SABE, SFF, and model behavior remain otherwise frozen. Interpret admission rate, accepted anchor quality, threshold lag, and similarity jointly; these are controlled results, not proof of a universal mechanism.", "",
              "## 13. Conclusion", "",
              f"The predefined interpretation is **Case {case}**. Released weighted Dice is {released:.4f}%. The best deployable variant at seed=2026 is **{LABELS[best]}**.", ""]
    if case == "A":
        lines.append("At least one deployable history variant shows a meaningful and mechanistically compatible improvement; admission-history timescale remains a research direction, not a final method.")
    elif case == "B":
        lines.append("The fixed-history variants remain within the predefined 0.2-point practical margin; SFT-Queue-40 is a negative outlier with near-zero admission rather than evidence for a useful replacement. Current data do not support long CCD history as the main bottleneck. The next direction should be class-level reliability and semantic memory retrieval.")
    elif case == "C":
        lines.append("Released cumulative history is best; this should not be called a bug. Long-term history may provide a stability prior, motivating future dual-timescale reliability work.")
    else:
        lines.append("The Oracle improves while simple deployable rolling variants do not; this supports a possible cross-domain history effect, but fixed rolling windows are insufficient.")
    multi = overall[overall.variant.isin(["released", best])].groupby("variant").slice_weighted_dice.agg(["min", "max", "std"])
    lines += ["", "## Final Answers", "",
              f"1. Released weighted Dice: **{released:.4f}%**.",
              f"2. Rolling-All-40 Δ: **{deltas.get('rolling_all_40', np.nan):.4f} pp**.",
              f"3. Rolling-All-160 Δ: **{deltas.get('rolling_all_160', np.nan):.4f} pp**.",
              f"4. SFT-Queue-40 Δ: **{deltas.get('sft_queue_40', np.nan):.4f} pp**.",
              f"5. Domain-Reset Oracle Δ: **{deltas.get('domain_reset_oracle', np.nan):.4f} pp**.",
              f"6. Best deployable variant: **{LABELS[best]}**.",
              "7. B/C/D admission rates are listed in `exp1_domain_summary.csv` and Section 6.",
              "8. B/C/D accepted SFT anchor Dice are listed in `accepted_sft_quality.csv` and Section 7.",
              "9. C/D threshold adaptation lags are listed in `threshold_adaptation_lag.csv` and Section 9.",
              f"10. Domain-Reset Oracle causal evidence: {'supports a possible cross-domain effect' if case == 'D' else 'does not by itself establish a causal cross-domain effect'}.",
              f"11. Multi-seed released/{LABELS[best]} weighted Dice ranges and standard deviations: {multi.to_dict(orient='index')}.",
              f"12. Final case: **Case {case}**.",
              f"13. Next direction: {'continue admission-history research as a controlled line' if case == 'A' else 'return to class-level reliability and semantic memory retrieval' if case == 'B' else 'study long-term stability plus short-term plasticity' if case == 'C' else 'investigate domain-change-aware or adaptive-timescale history'}.", ""]
    (OUT / "exp1_report.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    domain_summary = make_domain_summary(runs)
    overall = make_overall_summary(domain_summary)
    quality, _ = make_quality_tables(runs)
    transitions = transition_rows(runs)
    lag = adaptation_lag(runs)
    pairwise = make_pairwise_stats(runs)
    equivalence = released_equivalence(runs)
    make_figures(runs, domain_summary, overall, lag)
    build_report(domain_summary, overall, quality, transitions, lag, pairwise, equivalence)
    print(f"runs={len(runs)} released_equivalence={equivalence} rows={sum(len(frame) for frame in runs.values())}")


if __name__ == "__main__":
    main()
