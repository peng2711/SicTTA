#!/usr/bin/env python3
"""Analyze EXP-9 output-only confidence-delta selection runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp9"
RAW = OUT / "raw"
CLASSES = ("lv", "myo", "rv")
BOOTSTRAP_SAMPLES = 10000


def load_runs():
    runs = {}
    for path in sorted(RAW.glob("*_seed*.csv")):
        if path.stem.startswith("smoke_"):
            continue
        variant, seed_text = path.stem.rsplit("_seed", 1)
        runs[(variant, int(seed_text))] = pd.read_csv(path)
    return runs


def load_metadata():
    payloads = {}
    for path in sorted(RAW.glob("*_seed*.json")):
        if path.stem.startswith("smoke_"):
            continue
        variant, seed_text = path.stem.rsplit("_seed", 1)
        payloads[(variant, int(seed_text))] = json.loads(path.read_text())
    return payloads


def bootstrap_ci(values, seed=2026):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(BOOTSTRAP_SAMPLES)
    for start in range(0, BOOTSTRAP_SAMPLES, 250):
        count = min(250, BOOTSTRAP_SAMPLES - start)
        means[start:start + count] = rng.choice(
            values, size=(count, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def overall_summary(metadata, runs):
    rows = []
    for (variant, seed), payload in metadata.items():
        for item in payload["domain_summary"]:
            item = dict(item)
            frame = runs[(variant, seed)]
            part = frame if item["domain"] == "All" else frame[frame.domain == item["domain"]]
            active = np.zeros(len(part), dtype=bool)
            for suffix in CLASSES:
                active |= part[f"class_gate_{suffix}"].to_numpy(float) < 1.0 - 1e-8
            item["fallback_slice_rate"] = float(active.mean())
            item["fallback_pixel_mass"] = float(part.fallback_pixel_mass.mean())
            rows.append(item)
    result = pd.DataFrame(rows).sort_values(["seed", "variant", "domain"])
    result.to_csv(OUT / "exp9_domain_summary.csv", index=False)
    result[result.domain == "All"].to_csv(OUT / "exp9_overall_summary.csv", index=False)
    return result


def class_and_gate_summary(runs):
    class_rows, gate_rows = [], []
    for (variant, seed), frame in runs.items():
        for domain in ["B", "C", "D", "All"]:
            part = frame if domain == "All" else frame[frame.domain == domain]
            active = np.zeros(len(part), dtype=bool)
            for suffix in CLASSES:
                active |= part[f"class_gate_{suffix}"].to_numpy(float) < 1.0 - 1e-8
            gate_rows.append({
                "variant": variant, "seed": seed, "domain": domain, "n": len(part),
                "fallback_slice_rate": float(active.mean()),
                "fallback_pixel_mass": float(part.fallback_pixel_mass.mean()),
                "gate_map_mean": float(part.gate_map_mean.mean()),
            })
            for suffix in CLASSES:
                delta = part[f"selected_dice_{suffix}"] - part[f"released_dice_{suffix}"]
                class_rows.append({
                    "variant": variant, "seed": seed, "domain": domain, "class": suffix,
                    "n": len(delta), "mean_delta_pp": float(delta.mean() * 100.0),
                    "median_delta_pp": float(delta.median() * 100.0),
                    "class_gate_mean": float(part[f"class_gate_{suffix}"].mean()),
                    "class_rejection_rate": float((part[f"class_gate_{suffix}"] < 1.0 - 1e-8).mean()),
                })
    class_result = pd.DataFrame(class_rows)
    gate_result = pd.DataFrame(gate_rows)
    class_result.to_csv(OUT / "class_delta_summary.csv", index=False)
    gate_result.to_csv(OUT / "gate_summary.csv", index=False)
    return class_result, gate_result


def pairwise_case_statistics(metadata):
    rows = []
    keys = sorted(metadata)
    for variant, seed in keys:
        if variant in {"released", "identity"} or ("released", seed) not in metadata:
            continue
        current = pd.DataFrame(metadata[(variant, seed)]["per_case"])
        baseline = pd.DataFrame(metadata[("released", seed)]["per_case"])
        merged = current.merge(baseline, on=["domain", "volume_id"], suffixes=("_current", "_released"))
        for domain in ["B", "C", "D", "All"]:
            part = merged if domain == "All" else merged[merged.domain == domain]
            delta = (part.dice_average_current - part.dice_average_released).to_numpy(float)
            low, high = bootstrap_ci(delta, seed + len(rows))
            if len(delta) and np.any(np.abs(delta) > 0):
                try:
                    pvalue = float(stats.wilcoxon(delta).pvalue)
                except ValueError:
                    pvalue = np.nan
            else:
                pvalue = np.nan
            rows.append({
                "variant": variant, "seed": seed, "domain": domain, "n_cases": len(delta),
                "mean_delta_pp": float(np.mean(delta) * 100.0) if len(delta) else np.nan,
                "median_delta_pp": float(np.median(delta) * 100.0) if len(delta) else np.nan,
                "bootstrap_ci95_low_pp": low * 100.0, "bootstrap_ci95_high_pp": high * 100.0,
                "wilcoxon_p": pvalue,
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "pairwise_case_statistics.csv", index=False)
    return result


def build_report(overall, class_summary, gate_summary, pairwise):
    all_rows = overall[overall.domain == "All"].copy()
    lines = [
        "# EXP-9 Post-Adaptation Confidence-Delta Selection", "",
        "## Protocol", "",
        "Output-only selection; Released SicTTA memory, admission, retrieval, SABE, and SFF remain unchanged. ",
        "The selector uses only anchor/adapted probabilities and past confidence-delta history; GT is evaluation-only.", "",
        "## Overall Results", "",
        all_rows.to_markdown(index=False), "", "## Class-wise Delta", "",
        class_summary[class_summary.domain == "All"].to_markdown(index=False), "",
        "## Gate Behavior", "", gate_summary[gate_summary.domain == "All"].to_markdown(index=False), "",
        "## Paired Case Statistics", "", pairwise.to_markdown(index=False) if len(pairwise) else "Not available.", "",
    ]
    if (all_rows.variant == "released").any():
        released = all_rows[all_rows.variant == "released"].set_index("seed").dice_average
        decision_rows = []
        for _, row in all_rows.iterrows():
            if row.variant in {"released", "identity"} or row.seed not in released:
                continue
            delta_pp = (row.dice_average - released.loc[row.seed]) * 100.0
            decision_rows.append((row.variant, row.seed, delta_pp))
        lines.extend(["## Stage-1 Decision", ""])
        if decision_rows:
            best = max(decision_rows, key=lambda item: item[2])
            lines.append(f"Best variant: **{best[0]}**, seed={best[1]}, delta={best[2]:+.4f} pp.")
            lines.append("GO threshold: +0.15 pp and paired-case 95% CI excluding zero.")
            no_rv = class_summary[
                (class_summary.variant == "mad_soft_no_rv")
                & (class_summary.seed == 2026)
                & (class_summary.domain == "All")
                & (class_summary["class"].isin(["lv", "myo"]))
            ]
            if len(no_rv) == 2:
                lv_myo_delta = no_rv.mean_delta_pp.mean()
                decision = "GO" if lv_myo_delta > 0.15 else "STOP"
                lines.append(
                    "Time-boxed No-RV diagnostic: "
                    f"LV/MYO mean delta={lv_myo_delta:+.4f} pp; "
                    f"required >+0.15 pp. Decision: **{decision}**."
                )
        else:
            lines.append("No selector variant is available for a decision.")
        lines.append("")
    (OUT / "exp9_report.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs(); metadata = load_metadata()
    if not runs or not metadata:
        raise RuntimeError("no EXP-9 runs found")
    overall = overall_summary(metadata, runs)
    class_summary, gate_summary = class_and_gate_summary(runs)
    pairwise = pairwise_case_statistics(metadata)
    build_report(overall, class_summary, gate_summary, pairwise)
    print((OUT / "exp9_report.md").read_text())


if __name__ == "__main__":
    main()
