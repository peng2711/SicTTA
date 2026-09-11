#!/usr/bin/env python3
"""Run EXP-6 CR-SFF variants without changing the released admission pipeline."""

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
VARIANTS = {
    "released": "released",
    "crsff_identity": "crsff_identity",
    "global_reliability": "global_reliability",
    "class_reliability": "class_reliability",
    "inverse_class_reliability": "inverse_class_reliability",
}
CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
FIELDS = [
    "variant", "seed", "global_index", "domain", "volume_id", "slice_name", "z_index", "is_sft",
    "ccd", "ccd_threshold", "ccd_margin", "pool_size_before", "pool_size_after", "bank_lengths",
    "topk_indices", "topk_names", "topk_similarities", "released_weights", "memory_class_reliabilities",
    "memory_class_soft_masses", "memory_global_reliabilities", "global_reliability_weights",
    "class_reliability_weights", "inverse_class_reliability_weights", "query_class_soft_mass",
    "weight_l1_change_vs_released", "weight_entropy_released", "weight_entropy_class",
    "top1_weight_released", "top1_weight_class", "global_rate",
    "anchor_dice_lv", "anchor_dice_myo", "anchor_dice_rv", "anchor_dice_average",
    "adapted_dice_lv", "adapted_dice_myo", "adapted_dice_rv", "adapted_dice_average",
    "slice_elapsed_seconds",
]


def prediction_digest_update(digest, domain, name, prediction):
    digest.update(f"{domain}\t{name}\n".encode())
    digest.update(np.asarray(prediction, dtype=np.int64).tobytes())


def class_dice(prediction, target):
    return [float(dice(prediction, target, class_id)) for class_id, _ in CLASSES]


def serialize(value):
    return json.dumps(json_safe(value), separators=(",", ":"))


def summarize(records, volume_masks, variant, seed):
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
            values = class_dice(prediction, target)
            case = {"variant": variant, "seed": seed, "domain": domain, "volume_id": volume,
                    "dice_lv": values[0], "dice_myo": values[1], "dice_rv": values[2],
                    "dice_average": float(np.mean(values)), "assd_lv": assd(prediction, target, 1),
                    "assd_myo": assd(prediction, target, 2), "assd_rv": assd(prediction, target, 3),
                    "assd_average": float(np.nanmean([assd(prediction, target, c) for c, _ in CLASSES])),
                    "num_slices": len(ordered)}
            cases.append(case); per_case.append(case)
        rows.append({"variant": variant, "seed": seed, "domain": domain, "num_slices": len(subset),
                     "dice_lv": float(np.mean([r["adapted_dice_lv"] for r in subset])),
                     "dice_myo": float(np.mean([r["adapted_dice_myo"] for r in subset])),
                     "dice_rv": float(np.mean([r["adapted_dice_rv"] for r in subset])),
                     "dice_average": float(np.mean([r["adapted_dice_average"] for r in subset])),
                     "assd_lv": float(np.nanmean([r["assd_lv"] for r in cases])),
                     "assd_myo": float(np.nanmean([r["assd_myo"] for r in cases])),
                     "assd_rv": float(np.nanmean([r["assd_rv"] for r in cases])),
                     "assd_average": float(np.nanmean([r["assd_average"] for r in cases]))})
    rows.append({"variant": variant, "seed": seed, "domain": "All", "num_slices": len(records),
                 "dice_lv": float(np.mean([r["adapted_dice_lv"] for r in records])),
                 "dice_myo": float(np.mean([r["adapted_dice_myo"] for r in records])),
                 "dice_rv": float(np.mean([r["adapted_dice_rv"] for r in records])),
                 "dice_average": float(np.mean([r["adapted_dice_average"] for r in records])),
                 "assd_lv": float(np.nanmean([r["assd_lv"] for r in per_case])),
                 "assd_myo": float(np.nanmean([r["assd_myo"] for r in per_case])),
                 "assd_rv": float(np.nanmean([r["assd_rv"] for r in per_case])),
                 "assd_average": float(np.nanmean([r["assd_average"] for r in per_case]))})
    return rows, per_case


def verify_reference(frame, metadata, reference_csv, reference_json, output_path, label):
    reference = pd.read_csv(reference_csv)
    pivot = reference.pivot(index="global_index", columns="class_name", values="released_dice").sort_index()
    current = frame.sort_values("global_index").drop_duplicates("global_index").set_index("global_index")
    same_order = len(current) == len(pivot) and current.slice_name.tolist() == reference.sort_values("global_index").drop_duplicates("global_index").slice_name.tolist()
    deltas = []
    for class_name in ["lv", "myo", "rv"]:
        deltas.extend((current[f"adapted_dice_{class_name}"] - pivot[class_name]).dropna().abs().tolist())
    max_difference = float(max(deltas)) if deltas else float("inf")
    reference_hash = json.loads(reference_json.read_text())["prediction_sha256"]
    hash_equal = metadata["prediction_sha256"] == reference_hash
    passed = bool(same_order and max_difference <= 1e-8 and hash_equal)
    lines = [f"EXP-6 {label} seed=2026 vs EXP-5 Released equivalence", f"record_count_current={len(current)}",
             f"record_count_reference={len(pivot)}", f"slice_names_and_order_equal={same_order}",
             f"max_class_dice_difference={max_difference:.12f}", "tolerance=1e-8",
             f"prediction_sha256_equal={hash_equal}", f"PASS={passed}"]
    output_path.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError(f"{label} equivalence failed; stopping EXP-6")


def run(data_root, checkpoint, output, seed, variant, limit=None):
    fusion_mode = VARIANTS[variant]
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy="released", fusion_mode=fusion_mode)
    records, volume_masks = [], defaultdict(dict)
    digest = hashlib.sha256(); started = time.time()
    total = min(limit, len(stream.items)) if limit is not None else len(stream.items)
    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if global_index >= total:
            break
        slice_started = time.perf_counter()
        with torch.inference_mode():
            adapted_output = model(image.to(device), [name])
        diag = dict(model.last_diag); crsff = dict(diag.get("crsff", {}))
        prediction = adapted_output.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        anchor_prediction = model.last_anchor_probability.argmax(1).cpu().numpy()[0]
        anchor_values = class_dice(anchor_prediction, target)
        adapted_values = class_dice(prediction, target)
        prediction_digest_update(digest, domain, name, prediction)
        records.append({
            "variant": variant, "seed": seed, "global_index": global_index, "domain": domain,
            "volume_id": volume, "slice_name": name, "z_index": z_index, "is_sft": bool(diag.get("is_sft", False)),
            "ccd": diag.get("ccd"), "ccd_threshold": diag.get("ccd_threshold"), "ccd_margin": diag.get("ccd_margin"),
            "pool_size_before": crsff.get("pool_size", diag.get("pool_before", {}).get("feature", 0)),
            "pool_size_after": diag.get("pool_after", {}).get("feature", 0), "bank_lengths": serialize(crsff.get("bank_lengths", {})),
            "topk_indices": serialize(crsff.get("topk_indices", [])), "topk_names": serialize(crsff.get("topk_names", [])),
            "topk_similarities": serialize(crsff.get("topk_similarities", [])), "released_weights": serialize(crsff.get("released_weights", [])),
            "memory_class_reliabilities": serialize(crsff.get("memory_class_reliabilities", [])),
            "memory_class_soft_masses": serialize(crsff.get("memory_class_soft_masses", [])),
            "memory_global_reliabilities": serialize(crsff.get("memory_global_reliabilities", [])),
            "global_reliability_weights": serialize(crsff.get("global_reliability_weights", [])),
            "class_reliability_weights": serialize(crsff.get("class_reliability_weights", [])),
            "inverse_class_reliability_weights": serialize(crsff.get("inverse_class_reliability_weights", [])),
            "query_class_soft_mass": serialize(crsff.get("query_class_soft_mass", [])),
            "weight_l1_change_vs_released": serialize(crsff.get("weight_l1_change_vs_released", [])),
            "weight_entropy_released": crsff.get("weight_entropy_released"), "weight_entropy_class": serialize(crsff.get("weight_entropy_class", [])),
            "top1_weight_released": crsff.get("top1_weight_released"), "top1_weight_class": serialize(crsff.get("top1_weight_class", [])),
            "global_rate": crsff.get("global_rate", 0.0), "anchor_dice_lv": anchor_values[0], "anchor_dice_myo": anchor_values[1],
            "anchor_dice_rv": anchor_values[2], "anchor_dice_average": float(np.mean(anchor_values)),
            "adapted_dice_lv": adapted_values[0], "adapted_dice_myo": adapted_values[1], "adapted_dice_rv": adapted_values[2],
            "adapted_dice_average": float(np.mean(adapted_values)), "slice_elapsed_seconds": time.perf_counter() - slice_started,
        })
        volume_masks[(domain, volume)][z_index] = (prediction, target)
        if (global_index + 1) % 200 == 0:
            print(f"EXP-6 {variant} seed={seed}: {global_index + 1}/{total}", flush=True)
    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[FIELDS].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    domain_rows, per_case = summarize(records, volume_masks, variant, seed)
    peak_memory = int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
    metadata = {"variant": variant, "fusion_mode": fusion_mode, "seed": seed, "batch_size": 1,
                "data_root": str(data_root), "checkpoint": str(checkpoint), "stream_order": "B -> C -> D",
                "num_slices": len(records), "device": str(device), "elapsed_seconds": time.time() - started,
                "mean_slice_elapsed_seconds": float(frame.slice_elapsed_seconds.mean()), "peak_gpu_memory_bytes": peak_memory,
                "prediction_sha256": digest.hexdigest(), "domain_summary": domain_rows, "per_case": per_case,
                "topk": 5, "reliability_formula": "sum(P_c^2)/(sum(P_c)+1e-8)", "gt_used_only_for": ["Dice", "ASSD"]}
    output.with_suffix(".json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False))
    if limit is None and seed == 2026 and variant in {"released", "crsff_identity"}:
        label = "Released" if variant == "released" else "Identity"
        verify_reference(frame, metadata, ROOT / "results/exp5/per_class_utility_seed2026.csv",
                         ROOT / "results/exp5/per_class_utility_seed2026.json",
                         output.parent.parent / ("released_equivalence.txt" if variant == "released" else "identity_equivalence.txt"), label)
    print(json.dumps({"variant": variant, "seed": seed, "num_slices": len(records),
                      "elapsed_seconds": metadata["elapsed_seconds"], "peak_gpu_memory_bytes": peak_memory}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.seed, args.variant, args.limit)


if __name__ == "__main__":
    main()
