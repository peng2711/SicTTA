#!/usr/bin/env python3
"""Run minimal reliability-aware SFF gating variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp0"))
from run_exp0 import ProcessedStream, assd, dice, json_safe, load_model


DOMAINS = ["B", "C", "D"]
CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
VARIANTS = {
    "released": ("released", 1.0), "identity": ("identity", 1.0),
    "lowconf_g05": ("lowconf", 0.5), "lowconf_g10": ("lowconf", 1.0),
    "lowconf_g20": ("lowconf", 2.0), "highconf_g10": ("highconf", 1.0),
}
FIELDS = [
    "variant", "seed", "global_index", "domain", "volume_id", "slice_name", "z_index", "is_sft",
    "ccd", "ccd_threshold", "ccd_margin", "r_lv", "r_myo", "r_rv", "r_bar", "global_rate",
    "alpha_lv", "alpha_myo", "alpha_rv", "alpha_map_mean", "alpha_map_std", "alpha_map_min", "alpha_map_max",
    "fraction_alpha_clipped_0", "fraction_alpha_clipped_1", "scale_map_mean", "scale_map_max", "a_fallback",
    "correction_norm_released", "correction_norm_gated", "correction_norm_ratio",
    "corr_released_lv", "corr_released_myo", "corr_released_rv", "corr_gated_lv", "corr_gated_myo", "corr_gated_rv",
    "anchor_confidence_lv", "anchor_confidence_myo", "anchor_confidence_rv",
    "anchor_dice_lv", "anchor_dice_myo", "anchor_dice_rv", "anchor_dice_overall_present",
    "adapted_dice_lv", "adapted_dice_myo", "adapted_dice_rv", "adapted_dice_average",
]


def class_confidence(probability):
    # EXP4_GATE: exact EXP-3 Class Confidence, computed without GT.
    probability = probability[0].detach().float()
    masses = probability[1:].sum(dim=(1, 2))
    confidence = probability[1:].square().sum(dim=(1, 2)) / (masses + 1e-8)
    return confidence.detach().cpu().tolist()


def present_dice(prediction, target):
    values, present = [], []
    for class_id, _ in CLASSES:
        exists = bool(np.any(target == class_id))
        present.append(exists)
        values.append(dice(prediction, target, class_id) if exists else np.nan)
    valid = [value for value in values if np.isfinite(value)]
    return values, float(np.mean(valid)) if valid else np.nan


def prediction_digest_update(digest, domain, name, prediction):
    digest.update(f"{domain}\t{name}\n".encode())
    digest.update(np.asarray(prediction, dtype=np.int64).tobytes())


def verify_reference(records, metadata, reference_csv, reference_json, output_path, label):
    reference = pd.read_csv(reference_csv).sort_values("global_index").drop_duplicates("global_index")
    current = pd.DataFrame(records).sort_values("global_index").drop_duplicates("global_index")
    same_order = len(reference) == len(current) and reference.slice_name.tolist() == current.slice_name.tolist()
    delta = np.abs(reference.adapted_overall_dice.to_numpy() - current.adapted_dice_average.to_numpy()) if same_order else np.asarray([np.inf])
    max_difference = float(np.nanmax(delta))
    reference_hash = json.loads(reference_json.read_text()).get("prediction_sha256") if reference_json.exists() else None
    hash_equal = metadata.get("prediction_sha256") == reference_hash if reference_hash else None
    passed = bool(same_order and max_difference <= 1e-8 and (hash_equal is not False))
    lines = [
        f"EXP-4 {label} seed=2026 vs EXP-3 Released equivalence", f"record_count_current={len(current)}",
        f"record_count_reference={len(reference)}", f"slice_names_and_order_equal={same_order}",
        f"max_adapted_dice_difference={max_difference:.12f}", "tolerance=1e-8",
        f"prediction_sha256_equal={hash_equal}", f"PASS={passed}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError(f"{label} equivalence failed; stopping EXP-4")


def summarize(variant, seed, records, volume_masks, elapsed):
    rows, per_case = [], []
    for domain in DOMAINS:
        subset = [row for row in records if row["domain"] == domain]
        cases = []
        for (case_domain, volume), slices in volume_masks.items():
            if case_domain != domain:
                continue
            ordered = [slices[z] for z in sorted(slices)]
            prediction = np.stack([item[0] for item in ordered])
            target = np.stack([item[1] for item in ordered])
            dice_values = [dice(prediction, target, class_id) for class_id, _ in CLASSES]
            assd_values = [assd(prediction, target, class_id) for class_id, _ in CLASSES]
            case = {"variant": variant, "seed": seed, "domain": domain, "volume_id": volume,
                    "dice_lv": dice_values[0], "dice_myo": dice_values[1], "dice_rv": dice_values[2],
                    "dice_average": float(np.mean(dice_values)), "assd_lv": assd_values[0],
                    "assd_myo": assd_values[1], "assd_rv": assd_values[2],
                    "assd_average": float(np.nanmean(assd_values)), "num_slices": len(ordered)}
            per_case.append(case); cases.append(case)
        rows.append({
            "variant": variant, "seed": seed, "domain": domain, "num_slices": len(subset),
            "dice_lv": float(np.mean([row["adapted_dice_lv"] for row in subset])),
            "dice_myo": float(np.mean([row["adapted_dice_myo"] for row in subset])),
            "dice_rv": float(np.mean([row["adapted_dice_rv"] for row in subset])),
            "dice_average": float(np.mean([row["adapted_dice_average"] for row in subset])),
            "assd_lv": float(np.nanmean([row["assd_lv"] for row in cases])),
            "assd_myo": float(np.nanmean([row["assd_myo"] for row in cases])),
            "assd_rv": float(np.nanmean([row["assd_rv"] for row in cases])),
            "assd_average": float(np.nanmean([row["assd_average"] for row in cases])),
        })
    all_rows = [row for row in records]
    all_cases = per_case
    rows.append({
        "variant": variant, "seed": seed, "domain": "All", "num_slices": len(all_rows),
        "dice_lv": float(np.mean([row["adapted_dice_lv"] for row in all_rows])),
        "dice_myo": float(np.mean([row["adapted_dice_myo"] for row in all_rows])),
        "dice_rv": float(np.mean([row["adapted_dice_rv"] for row in all_rows])),
        "dice_average": float(np.mean([row["adapted_dice_average"] for row in all_rows])),
        "assd_lv": float(np.nanmean([row["assd_lv"] for row in all_cases])),
        "assd_myo": float(np.nanmean([row["assd_myo"] for row in all_cases])),
        "assd_rv": float(np.nanmean([row["assd_rv"] for row in all_cases])),
        "assd_average": float(np.nanmean([row["assd_average"] for row in all_cases])),
    })
    return rows, per_case


def run(data_root, checkpoint, output, seed, variant, exp3_reference):
    gate_mode, gamma = VARIANTS[variant]
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, gate_mode=gate_mode, gate_gamma=gamma)
    records, volume_masks = [], defaultdict(dict)
    digest = hashlib.sha256(); started = time.time()
    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        with torch.inference_mode():
            adapted_output = model(image.to(device), [name])
        prediction = adapted_output.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        anchor_probability = model.last_anchor_probability
        anchor_prediction = anchor_probability.argmax(1).cpu().numpy()[0]
        anchor_values, anchor_overall = present_dice(anchor_prediction, target)
        adapted_values = [dice(prediction, target, class_id) for class_id, _ in CLASSES]
        gate = dict(model.last_diag.get("gate", {}))
        confidence = class_confidence(anchor_probability)
        row = {"variant": variant, "seed": seed, "global_index": global_index, "domain": domain,
               "volume_id": volume, "slice_name": name, "z_index": z_index,
               "is_sft": bool(model.last_diag.get("is_sft", False)), "ccd": model.last_diag.get("ccd"),
               "ccd_threshold": model.last_diag.get("ccd_threshold"), "ccd_margin": model.last_diag.get("ccd_margin"),
               "r_lv": gate.get("r_lv"), "r_myo": gate.get("r_myo"), "r_rv": gate.get("r_rv"),
               "r_bar": gate.get("r_bar"), "global_rate": gate.get("global_rate", 0.0),
               "alpha_lv": gate.get("alpha_lv"), "alpha_myo": gate.get("alpha_myo"), "alpha_rv": gate.get("alpha_rv"),
               "alpha_map_mean": gate.get("alpha_map_mean"), "alpha_map_std": gate.get("alpha_map_std"),
               "alpha_map_min": gate.get("alpha_map_min"), "alpha_map_max": gate.get("alpha_map_max"),
               "fraction_alpha_clipped_0": gate.get("fraction_alpha_clipped_0"), "fraction_alpha_clipped_1": gate.get("fraction_alpha_clipped_1"),
               "scale_map_mean": gate.get("scale_map_mean"), "scale_map_max": gate.get("scale_map_max"),
               "a_fallback": gate.get("a_fallback", False),
               "correction_norm_released": gate.get("correction_norm_released"), "correction_norm_gated": gate.get("correction_norm_gated"),
               "correction_norm_ratio": gate.get("correction_norm_ratio"),
               "corr_released_lv": gate.get("corr_released_lv"), "corr_released_myo": gate.get("corr_released_myo"), "corr_released_rv": gate.get("corr_released_rv"),
               "corr_gated_lv": gate.get("corr_gated_lv"), "corr_gated_myo": gate.get("corr_gated_myo"), "corr_gated_rv": gate.get("corr_gated_rv"),
               "anchor_confidence_lv": confidence[0], "anchor_confidence_myo": confidence[1], "anchor_confidence_rv": confidence[2],
               "anchor_dice_lv": anchor_values[0], "anchor_dice_myo": anchor_values[1], "anchor_dice_rv": anchor_values[2],
               "anchor_dice_overall_present": anchor_overall,
               "adapted_dice_lv": adapted_values[0], "adapted_dice_myo": adapted_values[1], "adapted_dice_rv": adapted_values[2],
               "adapted_dice_average": float(np.mean(adapted_values))}
        records.append(row); volume_masks[(domain, volume)][z_index] = (prediction, target)
        prediction_digest_update(digest, domain, name, prediction)
        if (global_index + 1) % 200 == 0:
            print(f"EXP-4 {variant} seed={seed}: {global_index + 1}/{len(stream.items)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(records)
    domain_rows, per_case = summarize(variant, seed, records, volume_masks, time.time() - started)
    metadata = {"variant": variant, "gate_mode": gate_mode, "gamma": gamma, "seed": seed,
                "num_slices": len(records), "data_root": str(data_root), "checkpoint": str(checkpoint),
                "stream_order": "B -> C -> D", "device": str(device), "elapsed_seconds": time.time() - started,
                "prediction_sha256": digest.hexdigest(), "domain_summary": domain_rows, "per_case": per_case}
    output.with_suffix(".json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False))
    if seed == 2026 and variant == "released":
        verify_reference(records, metadata, exp3_reference, exp3_reference.with_suffix(".json"),
                         output.parent.parent / "released_equivalence.txt", "Released")
    if seed == 2026 and variant == "identity":
        verify_reference(records, metadata, exp3_reference, exp3_reference.with_suffix(".json"),
                         output.parent.parent / "identity_equivalence.txt", "Identity")
    print(json.dumps({"variant": variant, "seed": seed, "num_slices": len(records),
                      "elapsed_seconds": metadata["elapsed_seconds"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exp3-reference", type=Path, default=ROOT / "results/exp3/per_class_proxy_seed2026.csv")
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.seed, args.variant, args.exp3_reference)


if __name__ == "__main__":
    main()
