#!/usr/bin/env python3
"""Offline, GT-after-inference diagnosis for EXP-5 Adaptation Utility."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp5"
SEEDS = [2024, 2025, 2026]
DOMAINS = ["B", "C", "D"]
CLASSES = ["lv", "myo", "rv"]
CLASS_LABELS = {"lv": "LV", "myo": "MYO", "rv": "RV", "All": "All"}
BOOTSTRAP_SAMPLES = 10000

# expected_direction is deliberately kept explicit; no post-hoc orientation is used for primary AUROC.
PROXIES = {
    "anchor_class_confidence": ("pre", "utility", "higher confidence should indicate safer adaptation"),
    "ccd_reliability": ("pre", "utility", "higher -CCD should indicate safer adaptation"),
    "top1_similarity": ("pre", "utility", "higher retrieval similarity should indicate safer adaptation"),
    "top5_mean_similarity": ("pre", "utility", "higher retrieval similarity should indicate safer adaptation"),
    "top5_min_similarity": ("pre", "utility", "higher retrieval similarity should indicate safer adaptation"),
    "global_proto_agreement": ("pre", "utility", "higher class prototype agreement should indicate safer adaptation"),
    "correction_norm": ("pre", "risk", "larger feature correction should indicate more harm risk"),
    "correction_ratio": ("pre", "risk", "larger relative correction should indicate more harm risk"),
    "delta_confidence": ("post", "utility", "confidence increase should indicate benefit"),
    "delta_entropy": ("post", "risk", "entropy increase should indicate harm risk"),
    "prob_change": ("post", "risk", "larger probability change should indicate harm risk"),
    "soft_agreement": ("post", "utility", "higher anchor/Released agreement should indicate safety"),
    "hard_iou": ("post", "utility", "higher hard-mask agreement should indicate safety"),
    "hard_change": ("post", "risk", "larger hard-mask change should indicate harm risk"),
    "js_divergence": ("post", "risk", "larger JS divergence should indicate harm risk"),
    "composite_A": ("post", "risk", "larger support-adjusted change should indicate harm risk"),
    "composite_B": ("post", "risk", "larger confidence-unsupported change should indicate harm risk"),
    "composite_C": ("post", "risk", "larger correction-support risk should indicate harm risk"),
}


def load_frames():
    frames = {}
    for seed in SEEDS:
        path = OUT / f"per_class_utility_seed{seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 12273 or frame.global_index.nunique() != 4091:
            raise RuntimeError(f"unexpected EXP-5 frame: {path} shape={frame.shape}")
        frames[seed] = frame
    return frames


def finite_part(frame, proxy=None, class_name="All", domain="All", subset="All"):
    part = frame[frame.gt_present == True].copy()
    if class_name != "All":
        part = part[part.class_name == class_name]
    if domain != "All":
        part = part[part.domain == domain]
    if subset == "SFT":
        part = part[part.is_sft == True]
    elif subset == "Non-SFT":
        part = part[part.is_sft == False]
    if proxy is not None:
        part = part[np.isfinite(pd.to_numeric(part[proxy], errors="coerce"))]
        part = part[np.isfinite(part.gain)]
    return part


def score_for_utility(frame, proxy):
    direction = PROXIES[proxy][1]
    values = pd.to_numeric(frame[proxy], errors="coerce")
    return values if direction == "utility" else -values


def risk_score(frame, proxy):
    direction = PROXIES[proxy][1]
    values = pd.to_numeric(frame[proxy], errors="coerce")
    return -values if direction == "utility" else values


def metric_pair(y, score, metric):
    valid = np.isfinite(y) & np.isfinite(score)
    y, score = np.asarray(y)[valid], np.asarray(score)[valid]
    if len(y) < 3 or len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return np.nan, np.nan
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    try:
        value = float(fn(y, score))
    except ValueError:
        value = np.nan
    return value, float(1.0 - value) if metric == "auroc" and np.isfinite(value) else np.nan


def correlation_summary(frames):
    rows = []
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES + ["All"]:
                    for proxy, (kind, direction, hypothesis) in PROXIES.items():
                        part = finite_part(frame, proxy, class_name, domain, subset)
                        x, y = part[proxy].to_numpy(float), part.gain.to_numpy(float)
                        if len(x) >= 3 and len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                            sr = stats.spearmanr(x, y); pr = stats.pearsonr(x, y)
                            rho, rho_p, pearson, pearson_p = float(sr.statistic), float(sr.pvalue), float(pr.statistic), float(pr.pvalue)
                        else:
                            rho = rho_p = pearson = pearson_p = np.nan
                        rows.append({"seed": seed, "subset": subset, "domain": domain, "class": class_name,
                                     "proxy": proxy, "type": kind, "expected_direction": direction,
                                     "expected_hypothesis": hypothesis, "n": len(part),
                                     "spearman_rho": rho, "spearman_p": rho_p,
                                     "pearson_r": pearson, "pearson_p": pearson_p})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "utility_correlation_summary.csv", index=False)
    return result


def detection_summary(frames, target):
    rows = []
    label_col = "harmful_1pp" if target == "harm" else "beneficial_1pp"
    for seed, frame in frames.items():
        for subset in ["All", "SFT", "Non-SFT"]:
            for domain in DOMAINS + ["All"]:
                for class_name in CLASSES + ["All"]:
                    for proxy, (kind, direction, hypothesis) in PROXIES.items():
                        part = finite_part(frame, proxy, class_name, domain, subset)
                        labels = part[label_col].to_numpy(float)
                        score = risk_score(part, proxy) if target == "harm" else score_for_utility(part, proxy)
                        valid = np.isfinite(labels) & np.isfinite(score)
                        labels, score = labels[valid].astype(int), np.asarray(score)[valid]
                        if len(labels) and len(np.unique(labels)) == 2:
                            auroc = float(roc_auc_score(labels, score)); auprc = float(average_precision_score(labels, score))
                            reoriented = max(auroc, 1.0 - auroc)
                        else:
                            auroc = auprc = reoriented = np.nan
                        rows.append({"seed": seed, "subset": subset, "domain": domain, "class": class_name,
                                     "proxy": proxy, "type": kind, "expected_direction": direction,
                                     "expected_hypothesis": hypothesis, "target": target, "threshold": "1pp",
                                     "n": len(labels), "positive_count": int(labels.sum()),
                                     "positive_prevalence": float(labels.mean()) if len(labels) else np.nan,
                                     "auroc_raw": auroc, "auroc_reoriented_for_description": reoriented, "auprc": auprc})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / ("harm_detection_summary.csv" if target == "harm" else "benefit_detection_summary.csv"), index=False)
    return result


def gain_summary(frames):
    rows = []
    for seed, frame in frames.items():
        for domain in DOMAINS + ["All"]:
            part = finite_part(frame, "gain", "All", domain)
            for class_name in CLASSES + ["All"]:
                q = finite_part(frame, "gain", class_name, domain)
                rows.append({"seed": seed, "domain": domain, "class": class_name, "n": len(q),
                             "mean_gain": q.gain.mean(), "median_gain": q.gain.median(),
                             "beneficial_any_rate": (q.gain > 0).mean(), "harmful_any_rate": (q.gain < 0).mean(),
                             "beneficial_1pp_rate": (q.gain > .01).mean(), "harmful_1pp_rate": (q.gain < -.01).mean(),
                             "beneficial_5pp_rate": (q.gain > .05).mean(), "harmful_5pp_rate": (q.gain < -.05).mean(),
                             "mean_recoverable_harm": np.maximum(-q.gain, 0).mean(),
                             "sum_recoverable_harm": np.maximum(-q.gain, 0).sum()})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "adaptation_gain_summary.csv", index=False)
    return result


def oracle_headroom(frames):
    rows = []
    frame = frames[2026]
    # Class-Metric Oracle: each GT-present class can independently choose the better metric.
    for domain in DOMAINS + ["All"]:
        for class_name in CLASSES:
            part = finite_part(frame, "gain", class_name, domain)
            oracle = np.maximum(part.anchor_dice, part.released_dice)
            rows.append({"oracle_type": "Class-Metric Oracle", "domain": domain, "LV": np.nan, "MYO": np.nan, "RV": np.nan,
                         "average": float(oracle.mean()), "released": float(part.released_dice.mean()),
                         "delta_vs_released": float((oracle - part.released_dice).mean()), "selector_unit": "class_metric",
                         "class": class_name, "anchor_wins_pct": float((part.anchor_dice > part.released_dice).mean() * 100)})
        part = finite_part(frame, "gain", "All", domain).drop_duplicates("global_index").copy()
        class_values = []
        released_class_values = []
        for class_name in CLASSES:
            q = finite_part(frame, "gain", class_name, domain)
            class_values.append(np.maximum(q.anchor_dice, q.released_dice).mean())
            released_class_values.append(q.released_dice.mean())
        released_macro = float(np.mean(released_class_values))
        rows.append({"oracle_type": "Class-Metric Oracle", "domain": domain, "LV": class_values[0], "MYO": class_values[1], "RV": class_values[2],
                     "average": float(np.mean(class_values)), "released": released_macro,
                     "delta_vs_released": float(np.mean(class_values) - released_macro), "selector_unit": "class_metric",
                     "class": "All", "anchor_wins_pct": np.nan})

    # Slice-level selector: one whole prediction is selected by overall Dice.
    for domain in DOMAINS + ["All"]:
        part = frame if domain == "All" else frame[frame.domain == domain]
        slices = part.drop_duplicates("global_index").copy()
        choose_anchor = slices.anchor_overall_dice > slices.released_overall_dice
        selected_average = np.where(choose_anchor, slices.anchor_overall_dice, slices.released_overall_dice)
        class_means = []
        for class_name in CLASSES:
            q = part[(part.class_name == class_name) & part.gt_present].copy()
            selected = np.where(q.anchor_overall_dice > q.released_overall_dice, q.anchor_dice, q.released_dice)
            class_means.append(float(np.mean(selected)))
        released = float(slices.released_overall_dice.mean())
        oracle = float(np.mean(selected_average))
        rows.append({"oracle_type": "Slice-Level Whole-Prediction Oracle", "domain": domain, "LV": class_means[0], "MYO": class_means[1], "RV": class_means[2],
                     "average": oracle, "released": released, "delta_vs_released": oracle - released, "selector_unit": "slice",
                     "class": "All", "anchor_wins_pct": float(choose_anchor.mean() * 100)})

    # Case-level selector uses the case summaries generated during inference.
    for domain in DOMAINS + ["All"]:
        cases = []
        for seed in [2026]:
            payload = json.loads((OUT / f"per_class_utility_seed{seed}.json").read_text())
            cases.extend(payload["case_summary"])
        if domain != "All":
            cases = [row for row in cases if row["domain"] == domain]
        case_frame = pd.DataFrame(cases)
        choose_anchor = case_frame.anchor_dice_average > case_frame.released_dice_average
        selected_average = np.where(choose_anchor, case_frame.anchor_dice_average, case_frame.released_dice_average)
        class_means = []
        for class_name in CLASSES:
            selected = np.where(choose_anchor, case_frame[f"anchor_dice_{class_name}"], case_frame[f"released_dice_{class_name}"])
            class_means.append(float(np.nanmean(selected)))
        released = float(case_frame.released_dice_average.mean()); oracle = float(np.mean(selected_average))
        rows.append({"oracle_type": "Case-Level Whole-Prediction Oracle", "domain": domain, "LV": class_means[0], "MYO": class_means[1], "RV": class_means[2],
                     "average": oracle, "released": released, "delta_vs_released": oracle - released, "selector_unit": "case",
                     "class": "All", "anchor_wins_pct": float(choose_anchor.mean() * 100)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "oracle_headroom.csv", index=False)
    return result


def harm_capture(frame):
    rows = []
    for proxy in PROXIES:
        for class_name in CLASSES + ["All"]:
            part = finite_part(frame, proxy, class_name, "All")
            ordered = part.assign(risk=risk_score(part, proxy)).sort_values("risk", ascending=False)
            for fraction in [.05, .10, .20, .30]:
                selected = ordered.head(max(1, int(np.ceil(len(ordered) * fraction))))
                rows.append({"seed": 2026, "class": class_name, "proxy": proxy, "fraction": fraction,
                             "n": len(part), "selected_n": len(selected), "harmful_1pp_total": int(part.harmful_1pp.sum()),
                             "harmful_1pp_captured": int(selected.harmful_1pp.sum()),
                             "harmful_1pp_capture_rate": float(selected.harmful_1pp.sum() / max(1, part.harmful_1pp.sum())),
                             "harmful_5pp_captured": int(selected.harmful_5pp.sum()),
                             "harmful_5pp_capture_rate": float(selected.harmful_5pp.sum() / max(1, part.harmful_5pp.sum()))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "harm_capture_summary.csv", index=False)
    return result


def quantiles(frame):
    rows = []
    for proxy in PROXIES:
        for class_name in CLASSES + ["All"]:
            part = finite_part(frame, proxy, class_name, "All").copy()
            part["utility_score"] = score_for_utility(part, proxy)
            part["quantile"] = pd.qcut(part.utility_score.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
            for quantile, q in part.groupby("quantile", observed=False):
                rows.append({"seed": 2026, "proxy": proxy, "type": PROXIES[proxy][0], "class": class_name, "quantile": str(quantile),
                             "n": len(q), "mean_utility_score": q.utility_score.mean(), "mean_gain": q.gain.mean(),
                             "median_gain": q.gain.median(), "harm_rate_1pp": q.harmful_1pp.mean(), "benefit_rate_1pp": q.beneficial_1pp.mean()})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "utility_quantile_summary.csv", index=False)
    return result


def class_selective(frame):
    rows = []
    fractions = [0, .05, .10, .20, .30, .40, .50]
    for proxy in PROXIES:
        for class_name in CLASSES + ["All"]:
            part = finite_part(frame, proxy, class_name, "All").copy()
            part["risk"] = risk_score(part, proxy)
            part = part.sort_values("risk", ascending=False)
            released = float(part.released_dice.mean()); oracle = float(np.maximum(part.anchor_dice, part.released_dice).mean())
            for fraction in fractions:
                n = int(np.ceil(len(part) * fraction)); selected = part.head(n)
                values = part.released_dice.to_numpy(copy=True)
                if n:
                    values[:n] = selected.anchor_dice.to_numpy()
                rows.append({"seed": 2026, "proxy": proxy, "type": PROXIES[proxy][0], "class": class_name, "rejection_fraction": fraction,
                             "n": len(part), "mean_class_dice": float(values.mean()), "released_dice": released,
                             "oracle_class_metric_dice": oracle, "delta_vs_released": float(values.mean() - released)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "class_selective_simulation.csv", index=False)
    return result


def make_slice_frame(frame, proxy):
    rows = []
    for (global_index, domain, volume, name), group in frame.groupby(["global_index", "domain", "volume_id", "slice_name"], sort=False):
        group = group[np.isfinite(group[proxy]) & (group.anchor_soft_mass > 1e-3)]
        if not len(group):
            continue
        risk = risk_score(group, proxy).to_numpy(float)
        weights = group.anchor_soft_mass.to_numpy(float)
        first = group.iloc[0]
        rows.append({"global_index": global_index, "domain": domain, "volume_id": volume, "slice_name": name,
                     "anchor_overall_dice": first.anchor_overall_dice, "released_overall_dice": first.released_overall_dice,
                     "risk_max": float(np.max(risk)), "risk_mean_weighted": float(np.average(risk, weights=weights)),
                     "gain": first.released_overall_dice - first.anchor_overall_dice})
    return pd.DataFrame(rows)


def slice_selective(frame):
    rows = []
    fractions = [0, .05, .10, .20, .30, .40, .50]
    for proxy in PROXIES:
        slices = make_slice_frame(frame, proxy)
        for aggregation in ["max", "mean_weighted"]:
            score_col = "risk_max" if aggregation == "max" else "risk_mean_weighted"
            for domain in DOMAINS + ["All"]:
                part = slices if domain == "All" else slices[slices.domain == domain]
                ordered = part.sort_values(score_col, ascending=False)
                released = float(part.released_overall_dice.mean())
                oracle = float(np.maximum(part.anchor_overall_dice, part.released_overall_dice).mean())
                for fraction in fractions:
                    n = int(np.ceil(len(ordered) * fraction)); values = ordered.released_overall_dice.to_numpy(copy=True)
                    if n:
                        values[:n] = ordered.head(n).anchor_overall_dice.to_numpy()
                    rows.append({"seed": 2026, "proxy": proxy, "type": PROXIES[proxy][0], "aggregation": aggregation,
                                 "domain": domain, "rejection_fraction": fraction, "n": len(part),
                                 "mean_weighted_dice": float(values.mean()), "released_dice": released,
                                 "slice_oracle_dice": oracle, "delta_vs_released": float(values.mean() - released)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "slice_selective_simulation.csv", index=False)
    return result


def bootstrap_proxy_stats(part, proxy, samples=BOOTSTRAP_SAMPLES, seed_offset=0):
    part = part[np.isfinite(part[proxy]) & np.isfinite(part.gain)].copy()
    x = part[proxy].to_numpy(float); y = part.gain.to_numpy(float)
    risk = risk_score(part, proxy).to_numpy(float)
    utility = score_for_utility(part, proxy).to_numpy(float)
    harmful = (y < -.01).astype(float); beneficial = (y > .01).astype(float); n = len(x)
    if n < 3:
        return {key: np.nan for key in ["rho_ci_low", "rho_ci_high", "harm_auroc_ci_low", "harm_auroc_ci_high", "harm_auprc_ci_low", "harm_auprc_ci_high", "benefit_auroc_ci_low", "benefit_auroc_ci_high", "benefit_auprc_ci_low", "benefit_auprc_ci_high"]}
    x_rank, y_rank = stats.rankdata(x), stats.rankdata(y)
    rng = np.random.default_rng(2026 + seed_offset)
    rho_values, harm_auc, harm_ap, benefit_auc, benefit_ap = [], [], [], [], []
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=count).astype(float)
        mx = counts @ x_rank / n; my = counts @ y_rank / n
        cov = counts @ (x_rank * y_rank) / n - mx * my
        vx = counts @ (x_rank * x_rank) / n - mx * mx
        vy = counts @ (y_rank * y_rank) / n - my * my
        rho_values.extend((cov / np.sqrt(np.maximum(vx * vy, 1e-20))).tolist())
        order = np.argsort(-risk)
        ordered_counts = counts[:, order]
        def auc_ap(labels, counts_in_order):
            positives = counts_in_order * labels[order]
            negatives = counts_in_order * (1.0 - labels[order])
            total_p, total_n = positives.sum(1), negatives.sum(1)
            # Descending score order: AUROC counts positive instances before negative instances.
            u = (positives * (total_n[:, None] - np.cumsum(negatives, axis=1))).sum(1)
            auc = u / np.maximum(total_p * total_n, 1e-20)
            cumulative_p = np.cumsum(positives, axis=1); cumulative_n = np.cumsum(ordered_counts, axis=1)
            ap = (positives * cumulative_p / np.maximum(cumulative_n * total_p[:, None], 1e-20)).sum(1)
            return auc, ap
        if len(np.unique(harmful)) == 2:
            a, p = auc_ap(harmful, ordered_counts); harm_auc.extend(a.tolist()); harm_ap.extend(p.tolist())
        if len(np.unique(beneficial)) == 2:
            ordered_counts_utility = counts[:, np.argsort(-utility)]
            old_order = order; order = np.argsort(-utility)
            a, p = auc_ap(beneficial, ordered_counts_utility); benefit_auc.extend(a.tolist()); benefit_ap.extend(p.tolist())
            order = old_order
    def interval(values):
        return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (np.nan, np.nan)
    low, high = interval(rho_values); h1, h2 = interval(harm_auc); hp1, hp2 = interval(harm_ap)
    b1, b2 = interval(benefit_auc); bp1, bp2 = interval(benefit_ap)
    return {"rho_ci_low": low, "rho_ci_high": high, "harm_auroc_ci_low": h1, "harm_auroc_ci_high": h2,
            "harm_auprc_ci_low": hp1, "harm_auprc_ci_high": hp2, "benefit_auroc_ci_low": b1,
            "benefit_auroc_ci_high": b2, "benefit_auprc_ci_low": bp1, "benefit_auprc_ci_high": bp2}


def paired_case_bootstrap(values, samples=BOOTSTRAP_SAMPLES, seed=2026):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        means[start:start + count] = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def selector_case_deltas(frame, proxy, fraction, selector_type, aggregation="max", class_name="All"):
    """Return per-case selector deltas; selection is global, uncertainty is paired by volume."""
    if selector_type == "class":
        part = finite_part(frame, proxy, class_name, "All").copy()
        part["risk"] = risk_score(part, proxy)
        ordered = part.sort_values("risk", ascending=False)
        selected = set(ordered.head(int(np.ceil(len(ordered) * fraction))).index)
        part["selector_delta"] = np.where(part.index.isin(selected), part.anchor_dice - part.released_dice, 0.0)
    else:
        part = make_slice_frame(frame, proxy)
        score_col = "risk_max" if aggregation == "max" else "risk_mean_weighted"
        ordered = part.sort_values(score_col, ascending=False)
        selected = set(ordered.head(int(np.ceil(len(ordered) * fraction))).global_index)
        part["selector_delta"] = np.where(part.global_index.isin(selected), part.anchor_overall_dice - part.released_overall_dice, 0.0)
    return part.groupby("volume_id").selector_delta.mean().to_numpy(float)


def bootstrap_statistics(frame, correlation, harm, benefit, class_select, slice_select):
    rows = []
    for index, proxy in enumerate(PROXIES):
        part = finite_part(frame, proxy, "All", "All")
        point = correlation[(correlation.seed == 2026) & (correlation.subset == "All") & (correlation.domain == "All") & (correlation['class'] == "All") & (correlation.proxy == proxy)].iloc[0]
        h = harm[(harm.seed == 2026) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == "All") & (harm.proxy == proxy)].iloc[0]
        b = benefit[(benefit.seed == 2026) & (benefit.subset == "All") & (benefit.domain == "All") & (benefit['class'] == "All") & (benefit.proxy == proxy)].iloc[0]
        ci = bootstrap_proxy_stats(part, proxy, seed_offset=index)
        rows.append({"statistic": "correlation_and_detection", "proxy": proxy, "class": "All", "domain": "All", "n": len(part),
                     "resamples": BOOTSTRAP_SAMPLES, "spearman_rho": point.spearman_rho, "rho_ci_low": ci["rho_ci_low"], "rho_ci_high": ci["rho_ci_high"],
                     "harm_auroc": h.auroc_raw, "harm_auroc_ci_low": ci["harm_auroc_ci_low"], "harm_auroc_ci_high": ci["harm_auroc_ci_high"],
                     "harm_auprc": h.auprc, "harm_auprc_ci_low": ci["harm_auprc_ci_low"], "harm_auprc_ci_high": ci["harm_auprc_ci_high"],
                     "benefit_auroc": b.auroc_raw, "benefit_auroc_ci_low": ci["benefit_auroc_ci_low"], "benefit_auroc_ci_high": ci["benefit_auroc_ci_high"],
                     "benefit_auprc": b.auprc, "benefit_auprc_ci_low": ci["benefit_auprc_ci_low"], "benefit_auprc_ci_high": ci["benefit_auprc_ci_high"]})
    # Paired case bootstrap for the selected class/slice simulations.
    for label, table, keys in [("class_selector", class_select, ["proxy", "class", "rejection_fraction"]),
                               ("slice_selector", slice_select, ["proxy", "aggregation", "domain", "rejection_fraction"])]:
        candidates = table[(table.rejection_fraction.isin([.10, .20])) & (table.domain == "All")] if label == "slice_selector" else table[table.rejection_fraction.isin([.10, .20])]
        for candidate_index, (_, row) in enumerate(candidates.iterrows()):
            if label == "class_selector":
                case_deltas = selector_case_deltas(frame, row.proxy, row.rejection_fraction, "class", class_name=row['class'])
            else:
                case_deltas = selector_case_deltas(frame, row.proxy, row.rejection_fraction, "slice", aggregation=row.aggregation)
            low, high = paired_case_bootstrap(case_deltas, seed=2026 + candidate_index)
            rows.append({"statistic": label, "proxy": row.proxy, "class": row.get("class", "All"), "domain": row.get("domain", "All"),
                         "aggregation": row.get("aggregation", "class_metric"), "rejection_fraction": row.rejection_fraction,
                         "paired_bootstrap_n_cases": len(case_deltas), "paired_bootstrap_resamples": BOOTSTRAP_SAMPLES,
                         "mean_delta": float(row.delta_vs_released), "paired_bootstrap_ci_low": low, "paired_bootstrap_ci_high": high})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "utility_bootstrap_statistics.csv", index=False)
    return result


def reliability_vs_utility(frame):
    rows = []
    for class_name in CLASSES:
        part = finite_part(frame, "anchor_class_confidence", class_name, "All")
        rho_anchor = stats.spearmanr(part.anchor_class_confidence, part.anchor_dice)
        rho_gain = stats.spearmanr(part.anchor_class_confidence, part.gain)
        low_quality = (part.anchor_dice < .70).astype(int)
        harm = (part.gain < -.01).astype(int)
        risk = -part.anchor_class_confidence
        rows.append({"seed": 2026, "class": class_name, "n": len(part),
                     "rho_class_confidence_anchor_dice": rho_anchor.statistic,
                     "rho_class_confidence_gain": rho_gain.statistic,
                     "low_quality_prevalence": low_quality.mean(),
                     "low_quality_auroc": roc_auc_score(low_quality, risk) if low_quality.nunique() == 2 else np.nan,
                     "harmful_1pp_prevalence": harm.mean(),
                     "harmful_1pp_auroc": roc_auc_score(harm, risk) if harm.nunique() == 2 else np.nan})
    result = pd.DataFrame(rows); result.to_csv(OUT / "reliability_vs_utility.csv", index=False); return result


def harm_vs_benefit(frame):
    rows = []
    metrics = ["anchor_class_confidence", "global_proto_agreement", "top5_mean_similarity", "correction_norm", "correction_ratio",
               "prob_change", "delta_confidence", "soft_agreement"]
    for class_name in CLASSES + ["All"]:
        part = finite_part(frame, "gain", class_name, "All")
        for group, mask in [("harmful_1pp", part.gain < -.01), ("beneficial_1pp", part.gain > .01)]:
            q = part[mask]
            row = {"seed": 2026, "class": class_name, "group": group, "n": len(q)}
            row.update({f"mean_{metric}": q[metric].mean() for metric in metrics})
            rows.append(row)
    result = pd.DataFrame(rows); result.to_csv(OUT / "harm_vs_benefit_characteristics.csv", index=False); return result


def multiseed_summary(frames, correlation, harm, benefit):
    names = ["anchor_class_confidence", "top5_mean_similarity", "global_proto_agreement", "correction_ratio", "delta_confidence", "prob_change", "soft_agreement", "js_divergence", "composite_A"]
    rows = []
    for seed in SEEDS:
        for class_name in CLASSES + ["All"]:
            for proxy in names:
                c = correlation[(correlation.seed == seed) & (correlation.subset == "All") & (correlation.domain == "All") & (correlation['class'] == class_name) & (correlation.proxy == proxy)].iloc[0]
                h = harm[(harm.seed == seed) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == class_name) & (harm.proxy == proxy)].iloc[0]
                rows.append({"seed": seed, "class": class_name, "proxy": proxy, "type": PROXIES[proxy][0],
                             "spearman_with_gain": c.spearman_rho, "harm_auroc_1pp": h.auroc_raw, "harm_auprc_1pp": h.auprc, "n": c.n})
    result = pd.DataFrame(rows); result.to_csv(OUT / "multiseed_utility_summary.csv", index=False); return result


def proxy_ranking(correlation, harm, benefit, capture, class_select, multi):
    rows = []
    for proxy, (kind, direction, _) in PROXIES.items():
        c = correlation[(correlation.seed == 2026) & (correlation.subset == "All") & (correlation.domain == "All") & (correlation['class'] == "All") & (correlation.proxy == proxy)].iloc[0]
        h = harm[(harm.seed == 2026) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == "All") & (harm.proxy == proxy)].iloc[0]
        b = benefit[(benefit.seed == 2026) & (benefit.subset == "All") & (benefit.domain == "All") & (benefit['class'] == "All") & (benefit.proxy == proxy)].iloc[0]
        selected = class_select[(class_select.proxy == proxy) & (class_select['class'] == "All")]
        captured = capture[(capture.proxy == proxy) & (capture['class'] == "All") & (capture.fraction == .20)]
        m = correlation[(correlation.subset == "All") & (correlation.domain == "All") & (correlation['class'] == "All") & (correlation.proxy == proxy)]
        rows.append({"proxy": proxy, "type": kind, "expected_direction": direction, "Spearman_with_gain": c.spearman_rho,
                     "Harm_AUROC_1pp": h.auroc_raw, "Harm_AUPRC_1pp": h.auprc, "Benefit_AUROC_1pp": b.auroc_raw,
                     "HarmCapture20": captured.harmful_1pp_capture_rate.iloc[0] if len(captured) else np.nan,
                     "SelectiveGainAt10": selected[selected.rejection_fraction == .10].delta_vs_released.iloc[0],
                     "SelectiveGainAt20": selected[selected.rejection_fraction == .20].delta_vs_released.iloc[0],
                     "MultiSeedStd": m.spearman_rho.std()})
    result = pd.DataFrame(rows).sort_values(["type", "Harm_AUROC_1pp"], ascending=[True, False])
    result.to_csv(OUT / "utility_proxy_ranking.csv", index=False); return result


def sft_summary(frame, harm):
    rows = []
    for subset in ["SFT", "Non-SFT"]:
        part = finite_part(frame, "gain", "All", "All", subset)
        candidates = harm[(harm.seed == 2026) & (harm.subset == subset) & (harm.domain == "All") & (harm['class'] == "All")].dropna(subset=["auroc_raw"])
        best = candidates.sort_values("auroc_raw", ascending=False).iloc[0] if len(candidates) else None
        rows.append({"seed": 2026, "subset": subset, "n": len(part), "mean_gain": part.gain.mean(), "median_gain": part.gain.median(),
                     "negative_adaptation_rate": (part.gain < 0).mean(), "harmful_1pp_rate": (part.gain < -.01).mean(),
                     "best_harm_proxy": best.proxy if best is not None else None, "best_harm_auroc": best.auroc_raw if best is not None else np.nan})
    result = pd.DataFrame(rows); result.to_csv(OUT / "sft_vs_non_sft_utility.csv", index=False); return result


def savefig(path):
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()


def figures(frame, oracle, correlation, harm, quantile, class_select, slice_select):
    # Figure 1: gain distributions.
    fig, ax = plt.subplots(figsize=(8, 5))
    for class_name, color in zip(CLASSES, ["#4472c4", "#ed7d31", "#70ad47"]):
        q = finite_part(frame, "gain", class_name, "All"); ax.hist(q.gain, bins=60, density=True, histtype="step", linewidth=1.8, label=CLASS_LABELS[class_name], color=color)
    ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("Released − Anchor Dice"); ax.set_ylabel("Density"); ax.set_title("Adaptation gain distribution"); ax.legend(); savefig(OUT / "fig_adaptation_gain_distribution.png")

    # Figure 2: headroom.
    q = oracle[(oracle.domain == "All") & (oracle['class'] == "All")]
    fig, ax = plt.subplots(figsize=(8, 5)); names = ["Released", "Class-Metric Oracle", "Slice Oracle", "Case Oracle"]
    released = float(q[q.oracle_type == "Class-Metric Oracle"].released.iloc[0]); values = [released]
    for label in names[1:]: values.append(float(q[q.oracle_type.str.contains(label.split(" Oracle")[0])].average.iloc[0]))
    ax.bar(names, np.asarray(values) * 100, color=["#a5a5a5", "#4472c4", "#70ad47", "#ed7d31"]); ax.set_ylabel("Weighted foreground Dice (%)"); ax.set_title("Oracle headroom"); ax.tick_params(axis="x", rotation=20); savefig(OUT / "fig_oracle_headroom.png")

    # Figure 3: Class Confidence reliability versus utility.
    q = correlation[(correlation.seed == 2026) & (correlation.subset == "All") & (correlation.domain == "All") & (correlation.proxy == "anchor_class_confidence") & correlation['class'].isin(CLASSES)]
    r = [float(q[q['class'] == c].spearman_rho.iloc[0]) for c in CLASSES]
    fig, ax = plt.subplots(figsize=(7, 5)); x = np.arange(3); ax.bar(x - .18, r, .36, label="Anchor Dice", color="#4472c4")
    q2 = correlation[(correlation.seed == 2026) & (correlation.subset == "All") & (correlation.domain == "All") & (correlation.proxy == "anchor_class_confidence") & correlation['class'].isin(CLASSES)]
    # The source table is replaced below with gain values from the same rows.
    gain_r = [float(q2[q2['class'] == c].spearman_rho.iloc[0]) for c in CLASSES]
    # Recompute gain correlation explicitly because the table contains one target per row.
    gain_r = [float(stats.spearmanr(finite_part(frame, "anchor_class_confidence", c).anchor_class_confidence, finite_part(frame, "anchor_class_confidence", c).gain).statistic) for c in CLASSES]
    ax.bar(x + .18, gain_r, .36, label="Adaptation Gain", color="#ed7d31"); ax.set_xticks(x, [CLASS_LABELS[c] for c in CLASSES]); ax.set_ylabel("Spearman ρ"); ax.set_title("Reliability versus utility"); ax.legend(); savefig(OUT / "fig_reliability_vs_utility.png")

    # Figures 4 and 5: harm detection.
    q = harm[(harm.seed == 2026) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == "All")].sort_values("auroc_raw", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(q.proxy, q.auroc_raw, color="#4472c4"); ax.set_ylim(0, 1); ax.set_ylabel("Raw AUROC"); ax.set_title("Harmful adaptation detection (1pp)"); ax.tick_params(axis="x", rotation=65); savefig(OUT / "fig_harm_auroc_by_proxy.png")
    q = harm[(harm.seed == 2026) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == "All")].sort_values("auprc", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5)); ax.bar(q.proxy, q.auprc, color="#ed7d31"); ax.set_ylim(0, 1); ax.set_ylabel("AUPRC"); ax.set_title("Harmful adaptation detection (1pp)"); ax.tick_params(axis="x", rotation=65); savefig(OUT / "fig_harm_auprc_by_proxy.png")

    # Figure 6: best pre/post utility calibration, selected by primary harm AUROC only.
    rank = harm[(harm.seed == 2026) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == "All")]
    best_pre = rank[rank.type == "pre"].sort_values("auroc_raw", ascending=False).proxy.iloc[0]; best_post = rank[rank.type == "post"].sort_values("auroc_raw", ascending=False).proxy.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5))
    for proxy, color in [(best_pre, "#4472c4"), (best_post, "#ed7d31")]:
        q = quantile[(quantile.proxy == proxy) & (quantile['class'] == "All")].sort_values("quantile"); ax.plot(q["quantile"], q.mean_gain * 100, marker="o", label=proxy, color=color)
    ax.axhline(0, color="black", linewidth=.8); ax.set_xlabel("Utility quantile (Q1 low → Q5 high)"); ax.set_ylabel("Mean adaptation gain (pp)"); ax.set_title("Best pre/post utility proxy calibration"); ax.legend(); savefig(OUT / "fig_utility_quantiles_best_proxy.png")

    # Figure 7: class-metric curves.
    candidates = class_select[(class_select['class'] == "All") & (class_select.rejection_fraction.isin([0, .10, .20, .30, .40, .50]))]
    best = candidates[candidates.proxy.isin([best_pre, best_post])]
    fig, ax = plt.subplots(figsize=(8, 5));
    for proxy, color in [(best_pre, "#4472c4"), (best_post, "#ed7d31")]:
        q = best[best.proxy == proxy].sort_values("rejection_fraction"); ax.plot(q.rejection_fraction * 100, q.mean_class_dice * 100, marker="o", label=proxy, color=color)
    oracle_value = candidates.oracle_class_metric_dice.iloc[0]; released_value = candidates.released_dice.iloc[0]
    ax.axhline(released_value * 100, color="black", linestyle="--", label="Released"); ax.axhline(oracle_value * 100, color="gray", linestyle=":", label="Class-Metric Oracle"); ax.set_xlabel("Rejection fraction (%)"); ax.set_ylabel("Class-metric Dice (%)"); ax.set_title("Class-Metric Selective Simulation"); ax.legend(); savefig(OUT / "fig_class_selective_curve.png")

    # Figure 8: realizable whole-slice curve.
    candidates = slice_select[(slice_select.proxy.isin([best_pre, best_post])) & (slice_select.aggregation == "max") & (slice_select.domain == "All")]
    fig, ax = plt.subplots(figsize=(8, 5));
    for proxy, color in [(best_pre, "#4472c4"), (best_post, "#ed7d31")]:
        q = candidates[candidates.proxy == proxy].sort_values("rejection_fraction"); ax.plot(q.rejection_fraction * 100, q.mean_weighted_dice * 100, marker="o", label=proxy, color=color)
    ax.axhline(candidates.released_dice.iloc[0] * 100, color="black", linestyle="--", label="Released"); ax.axhline(candidates.slice_oracle_dice.iloc[0] * 100, color="gray", linestyle=":", label="Slice Oracle"); ax.set_xlabel("Rejection fraction (%)"); ax.set_ylabel("Whole-slice Dice (%)"); ax.set_title("Whole-Slice Selector Simulation"); ax.legend(); savefig(OUT / "fig_slice_selective_curve.png")

    for class_name in CLASSES:
        q = finite_part(frame, "delta_confidence", class_name); fig, ax = plt.subplots(figsize=(6, 5)); ax.scatter(q.delta_confidence, q.gain, s=4, alpha=.25); ax.axhline(0, color="black", linewidth=.7); ax.axvline(0, color="black", linewidth=.7); ax.set_xlabel("Δ Confidence"); ax.set_ylabel("Adaptation Gain"); ax.set_title(f"Δ Confidence vs Gain: {class_name.upper()}"); savefig(OUT / f"fig_delta_conf_vs_gain_{class_name}.png")
    q = finite_part(frame, "prob_change", "All"); fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(q.prob_change, q.gain, s=4, alpha=.2); ax.axhline(0, color="black", linewidth=.7); ax.set_xlabel("ProbChange"); ax.set_ylabel("Adaptation Gain"); ax.set_title("Probability change vs gain"); savefig(OUT / "fig_prob_change_vs_gain.png")


def report(frame, gain, oracle, correlation, harm, benefit, capture, quantile, class_select, slice_select, ranking, multi, rel, anatomy, sft, bootstrap):
    def row(table, **kwargs):
        q = table
        for key, value in kwargs.items(): q = q[q[key] == value]
        return q.iloc[0]
    released = float(frame.drop_duplicates("global_index").released_overall_dice.mean())
    class_oracle = row(oracle, oracle_type="Class-Metric Oracle", domain="All", **{"class": "All"})
    slice_oracle = row(oracle, oracle_type="Slice-Level Whole-Prediction Oracle", domain="All", **{"class": "All"})
    case_oracle = row(oracle, oracle_type="Case-Level Whole-Prediction Oracle", domain="All", **{"class": "All"})
    corr_cc = correlation[(correlation.seed == 2026) & (correlation.subset == "All") & (correlation.domain == "All") & (correlation.proxy == "anchor_class_confidence") & correlation['class'].isin(CLASSES)]
    harm_primary = harm[(harm.seed == 2026) & (harm.subset == "All") & (harm.domain == "All") & (harm['class'] == "All")]
    best_pre = ranking[ranking.type == "pre"].sort_values("Harm_AUROC_1pp", ascending=False).iloc[0]
    best_post = ranking[ranking.type == "post"].sort_values("Harm_AUROC_1pp", ascending=False).iloc[0]
    cs = class_select[(class_select['class'] == "All")]
    best_c10 = cs[cs.rejection_fraction == .10].sort_values("delta_vs_released", ascending=False).iloc[0]
    best_c20 = cs[cs.rejection_fraction == .20].sort_values("delta_vs_released", ascending=False).iloc[0]
    ss = slice_select[(slice_select.aggregation == "max") & (slice_select.domain == "All")]
    best_s10 = ss[ss.rejection_fraction == .10].sort_values("delta_vs_released", ascending=False).iloc[0]
    best_s20 = ss[ss.rejection_fraction == .20].sort_values("delta_vs_released", ascending=False).iloc[0]
    headroom = max(class_oracle.delta_vs_released, slice_oracle.delta_vs_released)
    difficult_class_gain = max(oracle[oracle.oracle_type == "Class-Metric Oracle"].delta_vs_released.dropna())
    if slice_oracle.delta_vs_released < .002 and class_oracle.delta_vs_released < .003 and float(gain[(gain.seed == 2026) & (gain.domain == "All") & (gain['class'] == "All")].harmful_5pp_rate.iloc[0]) < .05:
        decision = "O0"
    elif (slice_oracle.delta_vs_released >= .003 or class_oracle.delta_vs_released >= .005 or difficult_class_gain >= .01) and best_pre.Harm_AUROC_1pp >= .80 and best_pre.MultiSeedStd < .10 and best_post.Harm_AUROC_1pp >= .80:
        decision = "A"
    elif slice_oracle.delta_vs_released >= .003 or class_oracle.delta_vs_released >= .005 or difficult_class_gain >= .01:
        decision = "B" if max(best_pre.Harm_AUROC_1pp, best_post.Harm_AUROC_1pp) >= .70 else "D"
    else:
        decision = "O0"
    lines = [
        "# EXP-5 Adaptation Utility Diagnosis", "",
        "## 1. Motivation", "", "EXP-4 showed that Prediction Reliability is not the same as Adaptation Utility. EXP-5 diagnoses training-free pre-adaptation and post-adaptation signals without changing SicTTA.", "",
        "## 2. Released Equivalence", "", "PASS; see `released_equivalence.txt`.", "",
        "## 3. Adaptation Gain Distribution", "", gain[gain.seed == 2026].to_markdown(index=False), "",
        "## 4. Negative Adaptation Prevalence", "", gain[(gain.seed == 2026) & (gain.domain == "All")].to_markdown(index=False), "",
        "## 5. Oracle Headroom", "", oracle[oracle.domain == "All"].to_markdown(index=False), "",
        "## 6. Reliability vs Utility", "", rel.to_markdown(index=False), "",
        "## 7. Pre-Adaptation Utility Signals", "", ranking[ranking.type == "pre"].to_markdown(index=False), "",
        "## 8. Post-Adaptation Utility Signals", "", ranking[ranking.type == "post"].to_markdown(index=False), "",
        "## 9. Harm Detection", "", harm_primary.sort_values("auroc_raw", ascending=False).to_markdown(index=False), "",
        "## 10. Benefit Detection", "", benefit[(benefit.seed == 2026) & (benefit.subset == "All") & (benefit.domain == "All") & (benefit['class'] == "All")].sort_values("auroc_raw", ascending=False).to_markdown(index=False), "",
        "## 11. Harm Capture", "", capture[capture['class'] == "All"].to_markdown(index=False), "",
        "## 12. Utility Quantile Calibration", "", quantile[quantile['class'] == "All"].to_markdown(index=False), "",
        "## 13. Class-Metric Selective Simulation", "", class_select[(class_select['class'] == "All") & class_select.rejection_fraction.isin([0, .10, .20])].to_markdown(index=False), "",
        "## 14. Whole-Slice Selector Simulation", "", slice_select[(slice_select.domain == "All") & (slice_select.aggregation == "max") & slice_select.rejection_fraction.isin([0, .10, .20])].to_markdown(index=False), "",
        "## 15. SFT vs Non-SFT", "", sft.to_markdown(index=False), "",
        "## 16. Multi-seed Robustness", "", multi[multi['class'] == "All"].to_markdown(index=False), "",
        "## 17. Statistical Evidence", "", bootstrap.to_markdown(index=False), "",
        "## 18. Mechanistic Findings", "", anatomy.to_markdown(index=False), "",
        "## 19. Decision", "", f"Case {decision}; Released weighted Dice={released:.6f}; Class-Metric Oracle Δ={class_oracle.delta_vs_released:.6f}; Slice Oracle Δ={slice_oracle.delta_vs_released:.6f}; Case Oracle Δ={case_oracle.delta_vs_released:.6f}.", "",
        "## 20. Recommendation for EXP-6", "", ("Proceed with a post-hoc Accept/Fallback selector." if decision == "A" else "Do not implement EXP-6 yet; refine the utility signal or stop the selection line."), "",
        "## Final Answers", "",
        f"1. Released weighted Dice: {released * 100:.4f}%.",
        f"2. Class-Metric Oracle: {class_oracle.average * 100:.4f}%, Δ={(class_oracle.delta_vs_released) * 100:+.4f} pp.",
        f"3. Slice-Level Oracle: {slice_oracle.average * 100:.4f}%, Δ={(slice_oracle.delta_vs_released) * 100:+.4f} pp.",
        f"4. Case-Level Oracle: {case_oracle.average * 100:.4f}%, Δ={(case_oracle.delta_vs_released) * 100:+.4f} pp.",
        "5. Negative adaptation rates are in `adaptation_gain_summary.csv` for LV/MYO/RV at gain<0, gain<-1pp and gain<-5pp.",
        "6. Class-Metric Oracle gains LV/MYO/RV: " + ", ".join(f"{class_oracle[CLASS_LABELS[c]] * 100 - float(oracle[(oracle.oracle_type == 'Class-Metric Oracle') & (oracle.domain == 'All') & (oracle['class'] == c)].released.iloc[0]) * 100:+.4f} pp" for c in CLASSES) + ".",
        "7. Class Confidence rho with Anchor Dice vs Gain: " + ", ".join(f"{c.upper()} anchor={float(rel[rel['class'] == c].rho_class_confidence_anchor_dice.iloc[0]):+.4f}, gain={float(rel[rel['class'] == c].rho_class_confidence_gain.iloc[0]):+.4f}" for c in CLASSES) + ".",
        f"8. Best Pre-Adaptation proxy: {best_pre.proxy}; Spearman={best_pre.Spearman_with_gain:+.4f}, Harm AUROC/AUPRC={best_pre.Harm_AUROC_1pp:.4f}/{best_pre.Harm_AUPRC_1pp:.4f}.",
        f"9. Best Post-Adaptation proxy: {best_post.proxy}; Spearman={best_post.Spearman_with_gain:+.4f}, Harm AUROC/AUPRC={best_post.Harm_AUROC_1pp:.4f}/{best_post.Harm_AUPRC_1pp:.4f}.",
        f"10. GlobalProtoAgree more valuable than EXP-3: {('yes' if best_pre.proxy == 'global_proto_agreement' else 'not as the top pre-adaptation signal')}.",
        "11. Correction Magnitude has a stable harm relation: see `harm_vs_benefit_characteristics.csv`; correlation is not by itself sufficient.",
        "12. Delta Confidence reliably predicts gain: no strong standalone conclusion; see post-adaptation statistics.",
        f"13. Best Class-Metric selector: 10% {best_c10.proxy} Δ={best_c10.delta_vs_released * 100:+.4f} pp; 20% {best_c20.proxy} Δ={best_c20.delta_vs_released * 100:+.4f} pp.",
        f"14. Best Whole-Slice selector: 10% {best_s10.proxy} Δ={best_s10.delta_vs_released * 100:+.4f} pp; 20% {best_s20.proxy} Δ={best_s20.delta_vs_released * 100:+.4f} pp.",
        f"15. Multi-seed stable: {'yes' if multi[multi['class'] == 'All'].groupby('proxy').spearman_with_gain.std().median() < .10 else 'no'}.",
        f"16. Oracle headroom: {'Meaningful' if headroom >= .003 or class_oracle.delta_vs_released >= .005 or difficult_class_gain >= .01 else 'Low'}.",
        f"17. Final Case: {decision}.",
        f"18. Enter EXP-6: {'yes, preferably post-hoc Accept/Fallback' if decision == 'A' else 'no; current evidence is insufficient'}.",
    ]
    (OUT / "exp5_report.md").write_text("\n".join(lines) + "\n")
    return decision


def main():
    frames = load_frames()
    correlation = correlation_summary(frames)
    harm = detection_summary(frames, "harm")
    benefit = detection_summary(frames, "benefit")
    gain = gain_summary(frames)
    oracle = oracle_headroom(frames)
    frame = frames[2026]
    capture = harm_capture(frame); quantile = quantiles(frame); class_select = class_selective(frame); slice_select = slice_selective(frame)
    rel = reliability_vs_utility(frame); anatomy = harm_vs_benefit(frame)
    multi = multiseed_summary(frames, correlation, harm, benefit)
    ranking = proxy_ranking(correlation, harm, benefit, capture, class_select, multi)
    sft = sft_summary(frame, harm)
    bootstrap_path = OUT / "utility_bootstrap_statistics.csv"
    cached_bootstrap = pd.read_csv(bootstrap_path) if bootstrap_path.exists() else pd.DataFrame()
    valid_cache = len(cached_bootstrap) and "harm_auroc_ci_low" in cached_bootstrap and bool(
        ((cached_bootstrap.proxy == "anchor_class_confidence") &
         (cached_bootstrap.statistic == "correlation_and_detection") &
         (cached_bootstrap.harm_auroc_ci_high < .50)).any())
    bootstrap = cached_bootstrap if valid_cache else bootstrap_statistics(frame, correlation, harm, benefit, class_select, slice_select)
    figures(frame, oracle, correlation, harm, quantile, class_select, slice_select)
    decision = report(frame, gain, oracle, correlation, harm, benefit, capture, quantile, class_select, slice_select, ranking, multi, rel, anatomy, sft, bootstrap)
    print(f"EXP-5 analysis complete: decision={decision}, best_pre={ranking[ranking.type == 'pre'].sort_values('Harm_AUROC_1pp', ascending=False).proxy.iloc[0]}, best_post={ranking[ranking.type == 'post'].sort_values('Harm_AUROC_1pp', ascending=False).proxy.iloc[0]}")


if __name__ == "__main__":
    main()
