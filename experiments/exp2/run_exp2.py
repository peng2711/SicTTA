#!/usr/bin/env python3
"""Run Released SicTTA and collect class-reliability retrieval diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
EXP0_ROOT = ROOT / "experiments/exp0"
sys.path.insert(0, str(EXP0_ROOT))
from run_exp0 import ProcessedStream, dice, load_model


DOMAINS = ["B", "C", "D"]
CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
FIELDS = [
    "seed", "global_index", "domain", "volume_id", "slice_name", "z_index",
    "ccd", "ccd_threshold", "ccd_margin", "is_sft", "pool_size",
    "pool_size_before", "pool_size_after", "gt_present_lv", "gt_present_myo", "gt_present_rv",
    "anchor_dice_lv", "anchor_dice_myo", "anchor_dice_rv", "anchor_dice_overall_present",
    "class_dice_min", "class_dice_max", "class_dice_std", "class_dice_range", "weak_class_gap",
    "prototype_mass_lv", "prototype_mass_myo", "prototype_mass_rv",
    "global_topk_names", "global_topk_similarities",
    "lv_topk_names", "lv_topk_similarities", "myo_topk_names", "myo_topk_similarities",
    "rv_topk_names", "rv_topk_similarities",
    "global_lv_overlap", "global_myo_overlap", "global_rv_overlap",
    "lv_myo_overlap", "lv_rv_overlap", "myo_rv_overlap",
    "global_neighbor_presence_lv", "class_neighbor_presence_lv", "presence_gain_lv",
    "Q_global_lv", "Q_class_lv", "deltaQ_lv", "global_memory_mass_lv", "class_memory_mass_lv",
    "global_neighbor_presence_myo", "class_neighbor_presence_myo", "presence_gain_myo",
    "Q_global_myo", "Q_class_myo", "deltaQ_myo", "global_memory_mass_myo", "class_memory_mass_myo",
    "global_neighbor_presence_rv", "class_neighbor_presence_rv", "presence_gain_rv",
    "Q_global_rv", "Q_class_rv", "deltaQ_rv", "global_memory_mass_rv", "class_memory_mass_rv",
    "adapted_dice_average",
]


def conventional_dice(prediction, target, class_id):
    return dice(prediction, target, class_id)


def class_dice_present(prediction, target):
    values = []
    present = []
    for class_id, _ in CLASSES:
        has_class = bool(np.any(target == class_id))
        present.append(has_class)
        values.append(conventional_dice(prediction, target, class_id) if has_class else np.nan)
    valid = [value for value in values if np.isfinite(value)]
    overall = float(np.mean(valid)) if valid else np.nan
    return values, present, overall


def overlap(left, right):
    left, right = set(left), set(right)
    if not left or not right:
        return np.nan
    return float(len(left & right) / min(len(left), len(right)))


def jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return float(len(left & right) / len(union)) if union else np.nan


def topk_class(query, memory, memory_names, class_id):
    if memory is None or memory.shape[0] == 0:
        return [], [], []
    similarities = F.cosine_similarity(query[class_id - 1].unsqueeze(0), memory[:, class_id - 1, :], dim=1)
    count = min(5, memory.shape[0])
    indices = torch.argsort(similarities, descending=True)[:count].detach().cpu().tolist()
    names = [memory_names[index] for index in indices]
    values = [float(similarities[index].detach().cpu().item()) for index in indices]
    return names, values, indices


def neighbor_quality(names, class_id, memory_meta, memory_masses):
    entries = [memory_meta[name] for name in names if name in memory_meta]
    present = [entry["present"][class_id - 1] for entry in entries]
    present_values = [entry["dice"][class_id - 1] for entry in entries if entry["present"][class_id - 1]]
    masses = [mass for name, mass in zip(names, memory_masses) if name in memory_meta]
    return (
        float(np.mean(present)) if present else np.nan,
        float(np.mean(present_values)) if present_values else np.nan,
        float(np.mean(masses)) if masses else np.nan,
    )


def run(data_root: Path, checkpoint: Path, output: Path, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy="released")
    records = []
    memory_meta = {}
    volume_masks = defaultdict(dict)
    current_domain = None
    bank_checks = []
    started = time.time()

    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        current_domain = domain if current_domain is None else current_domain
        image_device = image.to(device)
        with torch.inference_mode():
            adapted_output = model(image_device, [name])
        diag = dict(model.last_diag)
        anchor_probability = model.last_anchor_probability
        anchor_prediction = anchor_probability.argmax(1).cpu().numpy()[0]
        adapted_prediction = adapted_output.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        anchor_values, present, anchor_overall = class_dice_present(anchor_prediction, target)
        adapted_values = [conventional_dice(adapted_prediction, target, class_id) for class_id, _ in CLASSES]
        present_values = [value for value in anchor_values if np.isfinite(value)]
        class_min = float(np.min(present_values)) if present_values else np.nan
        class_max = float(np.max(present_values)) if present_values else np.nan
        class_std = float(np.std(present_values)) if present_values else np.nan
        class_range = class_max - class_min if present_values else np.nan
        weak_gap = anchor_overall - class_min if present_values else np.nan

        query = model.last_class_prototypes
        masses = model.last_class_masses.detach().cpu().tolist()
        memory = model.last_class_memory_prototypes
        memory_masses = model.last_class_memory_masses
        memory_names = list(model.last_class_memory_names)
        if memory is not None:
            assert memory.shape[0] == len(memory_names)
            assert memory_masses.shape[0] == len(memory_names)
        bank_checks.append({
            "global_index": global_index,
            "pre_update_bank_size": 0 if memory is None else int(memory.shape[0]),
            "pre_update_name_size": len(memory_names),
        })

        global_names = list(diag.get("retrieved_names", []))
        global_similarities = [float(value) for value in diag.get("retrieved_similarities", [])]
        class_results = {}
        for class_id, suffix in CLASSES:
            class_names, class_similarities, class_indices = topk_class(query, memory, memory_names, class_id)
            class_masses = ([float(memory_masses[index, class_id - 1].detach().cpu().item()) for index in class_indices]
                            if memory_masses is not None else [])
            class_results[suffix] = {
                "names": class_names, "similarities": class_similarities, "masses": class_masses,
            }

        global_masses = []
        if memory_masses is not None:
            name_to_index = {value: index for index, value in enumerate(memory_names)}
            for class_id, _ in CLASSES:
                values = [float(memory_masses[name_to_index[name], class_id - 1].detach().cpu().item())
                          for name in global_names if name in name_to_index]
                global_masses.append(values)
        else:
            global_masses = [[], [], []]

        for class_id, suffix in CLASSES:
            global_presence, global_q, global_mass = neighbor_quality(
                global_names, class_id, memory_meta, global_masses[class_id - 1])
            class_presence, class_q, class_mass = neighbor_quality(
                class_results[suffix]["names"], class_id, memory_meta, class_results[suffix]["masses"])
            class_results[suffix].update({
                "global_presence": global_presence, "global_q": global_q, "global_mass": global_mass,
                "class_presence": class_presence, "class_q": class_q, "class_mass": class_mass,
            })

        volume_masks[(domain, volume)][z_index] = (adapted_prediction, target)
        if diag.get("is_sft"):
            memory_meta[name] = {"present": present, "dice": anchor_values}

        row = {
            "seed": seed, "global_index": global_index, "domain": domain, "volume_id": volume,
            "slice_name": name, "z_index": z_index, "ccd": diag.get("ccd"),
            "ccd_threshold": diag.get("ccd_threshold"), "ccd_margin": diag.get("ccd_margin"),
            "is_sft": diag.get("is_sft"), "pool_size": diag.get("pool_after", {}).get("name"),
            "pool_size_before": diag.get("pool_before", {}).get("name"),
            "pool_size_after": diag.get("pool_after", {}).get("name"),
            "gt_present_lv": present[0], "gt_present_myo": present[1], "gt_present_rv": present[2],
            "anchor_dice_lv": anchor_values[0], "anchor_dice_myo": anchor_values[1],
            "anchor_dice_rv": anchor_values[2], "anchor_dice_overall_present": anchor_overall,
            "class_dice_min": class_min, "class_dice_max": class_max, "class_dice_std": class_std,
            "class_dice_range": class_range, "weak_class_gap": weak_gap,
            "prototype_mass_lv": masses[0], "prototype_mass_myo": masses[1], "prototype_mass_rv": masses[2],
            "global_topk_names": json.dumps(global_names),
            "global_topk_similarities": json.dumps(global_similarities),
            "adapted_dice_average": float(np.mean(adapted_values)),
        }
        for class_id, suffix in CLASSES:
            result = class_results[suffix]
            row[f"{suffix}_topk_names"] = json.dumps(result["names"])
            row[f"{suffix}_topk_similarities"] = json.dumps(result["similarities"])
            row[f"global_{suffix}_overlap"] = overlap(global_names, result["names"])
        row["lv_myo_overlap"] = overlap(class_results["lv"]["names"], class_results["myo"]["names"])
        row["lv_rv_overlap"] = overlap(class_results["lv"]["names"], class_results["rv"]["names"])
        row["myo_rv_overlap"] = overlap(class_results["myo"]["names"], class_results["rv"]["names"])
        for _, suffix in CLASSES:
            result = class_results[suffix]
            row[f"global_neighbor_presence_{suffix}"] = result["global_presence"]
            row[f"class_neighbor_presence_{suffix}"] = result["class_presence"]
            row[f"presence_gain_{suffix}"] = result["class_presence"] - result["global_presence"]
            row[f"Q_global_{suffix}"] = result["global_q"]
            row[f"Q_class_{suffix}"] = result["class_q"]
            row[f"deltaQ_{suffix}"] = result["class_q"] - result["global_q"]
            row[f"global_memory_mass_{suffix}"] = result["global_mass"]
            row[f"class_memory_mass_{suffix}"] = result["class_mass"]
        records.append(row)
        if (global_index + 1) % 200 == 0:
            print(f"EXP-2 seed={seed}: {global_index + 1}/{len(stream.items)}", flush=True)

    for key, values in volume_masks.items():
        assert values
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    pool = model.pool
    bank_after_size = 0 if pool.class_prototype_bank is None else int(pool.class_prototype_bank.shape[0])
    assert bank_after_size == len(pool.name_list)
    metadata = {
        "seed": seed, "variant": "released", "batch_size": 1,
        "data_root": str(data_root), "checkpoint": str(checkpoint),
        "stream_order": "B -> C -> D", "num_slices": len(records),
        "elapsed_seconds": time.time() - started, "device": str(device),
        "diagnostic_bank_final_size": bank_after_size,
        "diagnostic_bank_final_names": list(pool.name_list),
        "bank_checks": bank_checks,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({key: value for key, value in metadata.items()
                      if key not in {"diagnostic_bank_final_names", "bank_checks"}}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.seed)


if __name__ == "__main__":
    main()
