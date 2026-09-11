#!/usr/bin/env python3
"""Collect test-time class reliability proxies without changing Released SicTTA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp2"))
from run_exp2 import CLASSES, ProcessedStream, class_dice_present, conventional_dice, load_model

from proxies import compute_class_proxies


FIELDS = [
    "seed", "global_index", "domain", "volume_id", "slice_name", "z_index", "class_id", "class_name",
    "is_sft", "gt_present", "anchor_class_dice", "anchor_overall_present_dice", "ccd", "ccd_reliability",
    "soft_mass", "class_entropy", "class_entropy_reliability", "class_confidence", "class_margin",
    "feature_compactness", "feature_compactness_reliability", "proto_top1", "proto_top5",
    "global_proto_agreement", "boundary_entropy", "boundary_entropy_reliability", "boundary_margin",
    "interior_entropy", "boundary_interior_contrast", "pred_area", "soft_area", "combined_A",
    "adapted_class_dice", "adapted_overall_dice",
]


def _digest_prediction(digest, domain, name, prediction):
    digest.update(f"{domain}\t{name}\n".encode())
    digest.update(np.asarray(prediction, dtype=np.int64).tobytes())


def verify_equivalence(records, exp2_path, output_path):
    reference = pd.read_csv(exp2_path)
    current = pd.DataFrame(records).drop_duplicates(["global_index"])
    current = current.sort_values("global_index")
    same_order = len(reference) == len(current) and reference.slice_name.tolist() == current.slice_name.tolist()
    if same_order:
        delta = np.abs(reference.adapted_dice_average.to_numpy() - current.adapted_overall_dice.to_numpy())
        max_difference = float(np.nanmax(delta))
    else:
        max_difference = np.inf
    passed = bool(same_order and max_difference <= 1e-8)
    lines = [
        "EXP-3 Released seed=2026 vs EXP-2 Released equivalence",
        "record_count_exp3=" + str(len(current)),
        "record_count_exp2=" + str(len(reference)),
        "slice_names_and_order_equal=" + str(same_order),
        f"max_adapted_dice_difference={max_difference:.12f}",
        "tolerance=1e-8",
        "PASS=" + str(passed),
    ]
    output_path.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("Released equivalence failed; proxy analysis was not started")


def add_combined_proxy(frame):
    components = [
        "class_entropy_reliability", "class_margin",
        "feature_compactness_reliability", "boundary_entropy_reliability",
    ]
    ranked = []
    for column in components:
        ranked_column = frame.groupby("class_name")[column].rank(method="average", pct=True)
        ranked.append(ranked_column.rename(column))
    frame["combined_A"] = pd.concat(ranked, axis=1).mean(axis=1, skipna=True)
    return frame


def run(data_root: Path, checkpoint: Path, output: Path, seed: int, exp2_reference: Path):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, ["B", "C", "D"])
    model = load_model("full", checkpoint, device, admission_policy="released")
    records = []
    prediction_digest = hashlib.sha256()
    started = time.time()

    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        with torch.inference_mode():
            adapted_output = model(image.to(device), [name])
        diag = dict(model.last_diag)
        anchor_probability = model.last_anchor_probability
        anchor_prediction = anchor_probability.argmax(1).cpu().numpy()[0]
        adapted_prediction = adapted_output.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        anchor_values, present, anchor_overall = class_dice_present(anchor_prediction, target)
        adapted_values = [conventional_dice(adapted_prediction, target, class_id) for class_id, _ in CLASSES]
        adapted_overall = float(np.mean(adapted_values))
        _digest_prediction(prediction_digest, domain, name, adapted_prediction)

        memory = model.last_class_memory_prototypes
        query = model.last_class_prototypes
        global_indices = list(diag.get("retrieved_indices", []))
        proxy_values = compute_class_proxies(
            anchor_probability, model.last_latent_feature_map, query, memory, global_indices
        )
        for class_id, class_name in CLASSES:
            values = proxy_values[class_name]
            records.append({
                "seed": seed, "global_index": global_index, "domain": domain, "volume_id": volume,
                "slice_name": name, "z_index": z_index, "class_id": class_id, "class_name": class_name,
                "is_sft": bool(diag.get("is_sft")), "gt_present": present[class_id - 1],
                "anchor_class_dice": anchor_values[class_id - 1],
                "anchor_overall_present_dice": anchor_overall, "ccd": diag.get("ccd"),
                "ccd_reliability": -float(diag.get("ccd")) if diag.get("ccd") is not None else np.nan,
                **values, "combined_A": np.nan,
                "adapted_class_dice": adapted_values[class_id - 1],
                "adapted_overall_dice": adapted_overall,
            })
        if (global_index + 1) % 200 == 0:
            print(f"EXP-3 seed={seed}: {global_index + 1}/{len(stream.items)}", flush=True)

    frame = add_combined_proxy(pd.DataFrame(records))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[FIELDS].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    if seed == 2026:
        verify_equivalence(
            records,
            exp2_reference,
            output.parent / "released_equivalence.txt",
        )
    metadata = {
        "seed": seed, "variant": "released", "batch_size": 1,
        "data_root": str(data_root), "checkpoint": str(checkpoint), "stream_order": "B -> C -> D",
        "num_slices": int(frame.global_index.nunique()), "num_query_classes": len(frame),
        "elapsed_seconds": time.time() - started, "device": str(device),
        "feature_normalization": "none; original bottleneck feature before SABE/SFF",
        "boundary_radius": 2, "prototype_topk": 5, "eps": 1e-8,
        "prediction_sha256": prediction_digest.hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--exp2-reference", type=Path, default=ROOT / "results/exp2/per_slice_diagnostics_seed2026.csv")
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.seed, args.exp2_reference)


if __name__ == "__main__":
    main()
