#!/usr/bin/env python3
"""Run one diagnostic-only Full SicTTA stream without changing its behavior."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP0_ROOT = ROOT / "experiments/exp0"
sys.path.insert(0, str(EXP0_ROOT))
from run_exp0 import ProcessedStream, dice, load_model


DOMAINS = ["B", "C", "D"]
DIAG_FIELDS = [
    "global_index", "domain", "volume_id", "slice_name", "z_index",
    "ccd", "ccd_threshold", "ccd_margin", "is_sft",
    "ccd_history_len_before", "ccd_history_len_after",
    "pool_size_before", "pool_size_after", "memory_written", "pool_grew",
    "current_domain_index", "top1_similarity", "top5_mean_similarity",
    "top5_min_similarity", "top5_max_similarity",
]
for rank in range(1, 6):
    DIAG_FIELDS.extend([f"retrieved_rank{rank}_name", f"retrieved_rank{rank}_domain"])
DIAG_FIELDS.extend(["dice_lv", "dice_myo", "dice_rv", "dice_average",
                    "memory_B_count", "memory_C_count", "memory_D_count"])


def snapshot(model, domain: str, global_index: int, name_domain: dict[str, str]) -> dict:
    pool = model.pool
    counts = Counter(name_domain.get(name, "unknown") for name in pool.name_list)
    return {
        "domain": domain,
        "global_index": global_index,
        "ccd_history_length": len(model.entropy_list),
        "feature_bank_size": int(pool.feature_bank.shape[0]),
        "image_bank_size": int(pool.image_bank.shape[0]),
        "mask_bank_size": int(pool.mask_bank.shape[0]),
        "name_list_size": len(pool.name_list),
        "memory_domain_counts": {key: int(counts.get(key, 0)) for key in DOMAINS},
        "memory_names": list(pool.name_list),
    }


def run(data_root: Path, checkpoint: Path, output: Path, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device)
    records = []
    snapshots = []
    name_domain: dict[str, str] = {}
    started = time.time()
    current_domain = None

    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if domain != current_domain:
            if current_domain is not None:
                snapshots.append(snapshot(model, current_domain, global_index - 1, name_domain))
            current_domain = domain
            snapshots.append(snapshot(model, domain, global_index, name_domain))

        # EXP0_5_DIAG: only the current stream item is passed to the frozen path.
        with torch.inference_mode():
            result_tensor = model(image.to(device), [name])
        diag = dict(model.last_diag)
        retrieved_names = list(diag.get("retrieved_names", []))
        retrieved_similarities = list(diag.get("retrieved_similarities", []))
        retrieved_domains = [name_domain.get(value) if value is not None else None
                             for value in retrieved_names]
        prediction = result_tensor.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        class_dice = [dice(prediction, target, class_id) for class_id in (1, 2, 3)]
        # EXP0_5_DIAG: map every admitted item, including FIFO replacements.
        if diag.get("is_sft"):
            name_domain[name] = domain
        pool_counts = Counter(name_domain.get(value, "unknown") for value in model.pool.name_list)
        row = {
            "global_index": global_index, "domain": domain, "volume_id": volume,
            "slice_name": name, "z_index": z_index,
            "ccd": diag.get("ccd"), "ccd_threshold": diag.get("ccd_threshold"),
            "ccd_margin": diag.get("ccd_margin"), "is_sft": diag.get("is_sft"),
            "ccd_history_len_before": diag.get("ccd_history_len_before"),
            "ccd_history_len_after": diag.get("ccd_history_len_after"),
            "pool_size_before": diag.get("pool_before", {}).get("name"),
            "pool_size_after": diag.get("pool_after", {}).get("name"),
            "memory_written": diag.get("memory_written"),
            "pool_grew": diag.get("pool_grew"),
            "current_domain_index": DOMAINS.index(domain),
            "top1_similarity": retrieved_similarities[0] if retrieved_similarities else None,
            "top5_mean_similarity": float(np.mean(retrieved_similarities)) if retrieved_similarities else None,
            "top5_min_similarity": min(retrieved_similarities) if retrieved_similarities else None,
            "top5_max_similarity": max(retrieved_similarities) if retrieved_similarities else None,
            "dice_lv": class_dice[0], "dice_myo": class_dice[1], "dice_rv": class_dice[2],
            "dice_average": float(np.mean(class_dice)),
            "memory_B_count": int(pool_counts.get("B", 0)),
            "memory_C_count": int(pool_counts.get("C", 0)),
            "memory_D_count": int(pool_counts.get("D", 0)),
        }
        for rank in range(5):
            row[f"retrieved_rank{rank + 1}_name"] = retrieved_names[rank] if rank < len(retrieved_names) else None
            row[f"retrieved_rank{rank + 1}_domain"] = retrieved_domains[rank] if rank < len(retrieved_domains) else None
        records.append(row)

        if (global_index + 1) % 200 == 0:
            print(f"Full SicTTA diagnostics: {global_index + 1}/{len(stream.items)}", flush=True)

    if current_domain is not None:
        snapshots.append(snapshot(model, current_domain, len(records) - 1, name_domain))

    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / "per_slice_diagnostics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIAG_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    (output.parent / "memory_snapshots.json").write_text(json.dumps(snapshots, indent=2))
    metadata = {
        "method": "SABE+SFF", "seed": seed, "batch_size": 1,
        "data_root": str(data_root), "checkpoint": str(checkpoint),
        "stream_order": "B -> C -> D", "num_slices": len(records),
        "elapsed_seconds": time.time() - started,
    }
    (output.parent / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Marker path; diagnostics are written beside it")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.seed)


if __name__ == "__main__":
    main()
