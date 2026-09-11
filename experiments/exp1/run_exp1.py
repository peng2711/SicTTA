#!/usr/bin/env python3
"""Run one controlled SicTTA admission-history variant."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP0_ROOT = ROOT / "experiments/exp0"
sys.path.insert(0, str(EXP0_ROOT))
from run_exp0 import ProcessedStream, dice, load_model


DOMAINS = ["B", "C", "D"]
VARIANTS = ["released", "rolling_all_40", "rolling_all_160",
            "sft_queue_40", "domain_reset_oracle"]
FIELDS = [
    "seed", "variant", "global_index", "domain", "volume_id", "slice_name", "z_index",
    "ccd", "threshold", "ccd_margin", "admission_history_len",
    "admission_history_len_before", "admission_history_len_after",
    "admission_history_type", "is_sft", "pool_size_before", "pool_size_after",
    "memory_written", "top1_similarity", "top5_mean_similarity",
    "top5_min_similarity", "top5_max_similarity",
]
for rank in range(1, 6):
    FIELDS.extend([f"retrieved_rank{rank}_name", f"retrieved_rank{rank}_domain"])
FIELDS.extend([
    "anchor_dice_lv", "anchor_dice_myo", "anchor_dice_rv", "anchor_dice_average",
    "adapted_dice_lv", "adapted_dice_myo", "adapted_dice_rv", "adapted_dice_average",
    "memory_B_count", "memory_C_count", "memory_D_count",
])


def score(prediction, target):
    values = [dice(prediction, target, class_id) for class_id in (1, 2, 3)]
    return values + [float(np.mean(values))]


def memory_snapshot(model, domain, global_index, name_domain):
    pool = model.pool
    counts = Counter(name_domain.get(name, "unknown") for name in pool.name_list)
    return {
        "domain": domain,
        "global_index": global_index,
        "admission_history_length": len(model.entropy_list),
        "accepted_queue_length": len(model.accepted_ccd_queue),
        "feature_bank_size": int(pool.feature_bank.shape[0]),
        "image_bank_size": int(pool.image_bank.shape[0]),
        "mask_bank_size": int(pool.mask_bank.shape[0]),
        "name_list_size": len(pool.name_list),
        "memory_domain_counts": {key: int(counts.get(key, 0)) for key in DOMAINS},
        "memory_names": list(pool.name_list),
    }


def run(data_root: Path, checkpoint: Path, output: Path, variant: str, seed: int):
    if variant not in VARIANTS:
        raise ValueError(variant)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy=variant)
    records = []
    volume_masks = defaultdict(dict)
    volume_anchor_masks = defaultdict(dict)
    snapshots = []
    name_domain = {}
    current_domain = None
    started = time.time()

    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if domain != current_domain:
            if current_domain is not None:
                snapshots.append(memory_snapshot(model, current_domain, global_index - 1, name_domain))
            current_domain = domain
            if variant == "domain_reset_oracle":
                model.reset_admission_history()
            snapshots.append(memory_snapshot(model, domain, global_index, name_domain))

        with torch.inference_mode():
            adapted_output = model(image.to(device), [name])
            anchor_output = model.model_anchor.eval()(image.to(device))
        diag = dict(model.last_diag)
        retrieved_names = list(diag.get("retrieved_names", []))
        retrieved_similarities = list(diag.get("retrieved_similarities", []))
        retrieved_domains = [name_domain.get(value) if value is not None else None
                             for value in retrieved_names]
        anchor_prediction = anchor_output.argmax(1).cpu().numpy()[0]
        adapted_prediction = adapted_output.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        anchor_scores = score(anchor_prediction, target)
        adapted_scores = score(adapted_prediction, target)
        volume_masks[(domain, volume)][z_index] = (adapted_prediction, target)
        volume_anchor_masks[(domain, volume)][z_index] = (anchor_prediction, target)

        if diag.get("is_sft"):
            name_domain[name] = domain
        counts = Counter(name_domain.get(value, "unknown") for value in model.pool.name_list)
        row = {
            "seed": seed, "variant": variant, "global_index": global_index,
            "domain": domain, "volume_id": volume, "slice_name": name, "z_index": z_index,
            "ccd": diag.get("ccd"), "threshold": diag.get("ccd_threshold"),
            "ccd_margin": diag.get("ccd_margin"),
            "admission_history_len": diag.get("admission_history_len"),
            "admission_history_len_before": diag.get("admission_history_len_before"),
            "admission_history_len_after": diag.get("admission_history_len_after"),
            "admission_history_type": diag.get("admission_history_type"),
            "is_sft": diag.get("is_sft"),
            "pool_size_before": diag.get("pool_before", {}).get("name"),
            "pool_size_after": diag.get("pool_after", {}).get("name"),
            "memory_written": diag.get("memory_written"),
            "top1_similarity": retrieved_similarities[0] if retrieved_similarities else None,
            "top5_mean_similarity": float(np.mean(retrieved_similarities)) if retrieved_similarities else None,
            "top5_min_similarity": min(retrieved_similarities) if retrieved_similarities else None,
            "top5_max_similarity": max(retrieved_similarities) if retrieved_similarities else None,
            "anchor_dice_lv": anchor_scores[0], "anchor_dice_myo": anchor_scores[1],
            "anchor_dice_rv": anchor_scores[2], "anchor_dice_average": anchor_scores[3],
            "adapted_dice_lv": adapted_scores[0], "adapted_dice_myo": adapted_scores[1],
            "adapted_dice_rv": adapted_scores[2], "adapted_dice_average": adapted_scores[3],
            "memory_B_count": int(counts.get("B", 0)),
            "memory_C_count": int(counts.get("C", 0)),
            "memory_D_count": int(counts.get("D", 0)),
        }
        for rank in range(5):
            row[f"retrieved_rank{rank + 1}_name"] = retrieved_names[rank] if rank < len(retrieved_names) else None
            row[f"retrieved_rank{rank + 1}_domain"] = retrieved_domains[rank] if rank < len(retrieved_domains) else None
        records.append(row)
        if (global_index + 1) % 200 == 0:
            print(f"{variant} seed={seed}: {global_index + 1}/{len(stream.items)}", flush=True)

    if current_domain is not None:
        snapshots.append(memory_snapshot(model, current_domain, len(records) - 1, name_domain))
    case_metrics = []
    for domain, volume in volume_masks:
        ordered = [volume_masks[(domain, volume)][z] for z in sorted(volume_masks[(domain, volume)])]
        ordered_anchor = [volume_anchor_masks[(domain, volume)][z] for z in sorted(volume_anchor_masks[(domain, volume)])]
        adapted_prediction = np.stack([item[0] for item in ordered])
        target = np.stack([item[1] for item in ordered])
        anchor_prediction = np.stack([item[0] for item in ordered_anchor])
        case_metrics.append({
            "domain": domain, "volume_id": volume, "num_slices": len(ordered),
            "anchor_dice_average": score(anchor_prediction, target)[3],
            "adapted_dice_average": score(adapted_prediction, target)[3],
            "anchor_dice_lv": score(anchor_prediction, target)[0],
            "anchor_dice_myo": score(anchor_prediction, target)[1],
            "anchor_dice_rv": score(anchor_prediction, target)[2],
            "adapted_dice_lv": score(adapted_prediction, target)[0],
            "adapted_dice_myo": score(adapted_prediction, target)[1],
            "adapted_dice_rv": score(adapted_prediction, target)[2],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "variant": variant, "seed": seed, "batch_size": 1,
        "data_root": str(data_root), "checkpoint": str(checkpoint),
        "stream_order": "B -> C -> D", "num_slices": len(records),
        "elapsed_seconds": time.time() - started, "device": str(device),
        "snapshots": snapshots,
        "case_metrics": case_metrics,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: value for key, value in metadata.items() if key != "snapshots"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.variant, args.seed)


if __name__ == "__main__":
    main()
