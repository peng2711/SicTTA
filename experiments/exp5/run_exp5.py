#!/usr/bin/env python3
"""Collect inference-only signals for EXP-5 Adaptation Utility Diagnosis."""

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
from run_exp0 import ProcessedStream, dice, json_safe, load_model


CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
DOMAINS = ["B", "C", "D"]
EPS = 1e-8

FIELDS = [
    "seed", "global_index", "domain", "volume_id", "slice_name", "z_index", "class_id", "class_name",
    "is_sft", "gt_present", "anchor_dice", "released_dice", "gain", "beneficial_any", "harmful_any",
    "beneficial_1pp", "harmful_1pp", "beneficial_5pp", "harmful_5pp", "neutral_1pp",
    "anchor_class_confidence", "released_class_confidence", "delta_confidence",
    "anchor_class_entropy", "released_class_entropy", "delta_entropy", "prob_change",
    "soft_agreement", "hard_iou", "hard_change", "js_divergence", "anchor_soft_mass",
    "ccd", "ccd_reliability", "top1_similarity", "top5_mean_similarity", "top5_min_similarity",
    "global_proto_agreement", "correction_norm", "correction_ratio", "composite_A", "composite_B", "composite_C",
    "anchor_overall_dice", "released_overall_dice",
]


def class_confidence(probability):
    probability = probability[0].detach().float()
    masses = probability[1:].sum(dim=(1, 2))
    values = probability[1:].square().sum(dim=(1, 2)) / (masses + EPS)
    return values.cpu().numpy(), masses.cpu().numpy()


def class_entropy(probability):
    probability = probability[0].detach().float()
    entropy_map = -(probability * torch.log(probability + EPS)).sum(dim=0)
    masses = probability[1:].sum(dim=(1, 2))
    values = (probability[1:] * entropy_map).sum(dim=(1, 2)) / (masses + EPS)
    return values.cpu().numpy()


def class_probability_metrics(anchor_probability, released_probability):
    anchor = anchor_probability[0].detach().float()
    released = released_probability[0].detach().float()
    anchor_conf, anchor_mass = class_confidence(anchor_probability)
    released_conf, _ = class_confidence(released_probability)
    anchor_ent = class_entropy(anchor_probability)
    released_ent = class_entropy(released_probability)
    anchor_hard = anchor.argmax(dim=0).cpu().numpy()
    released_hard = released.argmax(dim=0).cpu().numpy()
    result = []
    for offset, class_id in enumerate((1, 2, 3)):
        pa, pr = anchor[class_id], released[class_id]
        weight = .5 * (pa + pr)
        prob_change = float((weight * (pr - pa).abs()).sum().item() / (weight.sum().item() + EPS))
        soft_agreement = float((2 * pa * pr).sum().item() /
                               ((pa.square() + pr.square()).sum().item() + EPS))
        mask_a, mask_r = anchor_hard == class_id, released_hard == class_id
        union = np.logical_or(mask_a, mask_r).sum()
        hard_iou = float(np.logical_and(mask_a, mask_r).sum() / (union + EPS)) if union else np.nan
        mean_probability = .5 * (anchor + released)
        js_map = .5 * (anchor * torch.log((anchor + EPS) / (mean_probability + EPS)) +
                        released * torch.log((released + EPS) / (mean_probability + EPS))).sum(dim=0)
        js_divergence = float((weight * js_map).sum().item() / (weight.sum().item() + EPS))
        result.append({
            "anchor_class_confidence": float(anchor_conf[offset]),
            "released_class_confidence": float(released_conf[offset]),
            "delta_confidence": float(released_conf[offset] - anchor_conf[offset]),
            "anchor_class_entropy": float(anchor_ent[offset]),
            "released_class_entropy": float(released_ent[offset]),
            "delta_entropy": float(released_ent[offset] - anchor_ent[offset]),
            "prob_change": prob_change,
            "soft_agreement": soft_agreement,
            "hard_iou": hard_iou,
            "hard_change": float(1.0 - hard_iou) if np.isfinite(hard_iou) else np.nan,
            "js_divergence": js_divergence,
            "anchor_soft_mass": float(anchor_mass[offset]),
        })
    return result, anchor_hard, released_hard


def present_dice(prediction, target):
    values, present = [], []
    for class_id, _ in CLASSES:
        exists = bool(np.any(target == class_id))
        present.append(exists)
        values.append(dice(prediction, target, class_id) if exists else np.nan)
    valid = [value for value in values if np.isfinite(value)]
    return values, present, float(np.mean(valid)) if valid else np.nan


def whole_prediction_dice(prediction, target):
    """Match EXP-4's fixed three-foreground-class average for slice/case selection."""
    values = [dice(prediction, target, class_id) for class_id, _ in CLASSES]
    return values, float(np.mean(values))


def prediction_digest_update(digest, domain, name, prediction):
    digest.update(f"{domain}\t{name}\n".encode())
    digest.update(np.asarray(prediction, dtype=np.int64).tobytes())


def retrieval_values(diag):
    similarities = [float(value) for value in diag.get("retrieved_similarities", [])]
    if not similarities:
        return np.nan, np.nan, np.nan
    return similarities[0], float(np.mean(similarities)), float(np.min(similarities))


def load_exp3_values(seed):
    path = ROOT / "results/exp3" / f"per_class_proxy_seed{seed}.csv"
    frame = pd.read_csv(path)
    return frame.set_index(["global_index", "class_name"])


def safe_ratio(numerator, denominator):
    return float(numerator / (denominator + EPS)) if np.isfinite(numerator) and np.isfinite(denominator) else np.nan


def build_case_summary(volume_values, seed):
    rows = []
    for (domain, volume), values in sorted(volume_values.items()):
        values = sorted(values, key=lambda item: item["z_index"])
        anchor_masks = np.stack([item["anchor_prediction"] for item in values])
        released_masks = np.stack([item["released_prediction"] for item in values])
        targets = np.stack([item["target"] for item in values])
        anchor_values, anchor_average = whole_prediction_dice(anchor_masks, targets)
        released_values, released_average = whole_prediction_dice(released_masks, targets)
        rows.append({
            "seed": seed, "domain": domain, "volume_id": volume, "num_slices": len(values),
            "anchor_dice_lv": anchor_values[0], "anchor_dice_myo": anchor_values[1], "anchor_dice_rv": anchor_values[2],
            "anchor_dice_average": anchor_average, "released_dice_lv": released_values[0],
            "released_dice_myo": released_values[1], "released_dice_rv": released_values[2],
            "released_dice_average": released_average,
        })
    return rows


def verify_equivalence(frame, metadata, exp4_csv, exp4_json, output_path):
    reference = pd.read_csv(exp4_csv).sort_values("global_index").drop_duplicates("global_index")
    current = frame.sort_values("global_index").drop_duplicates("global_index")
    same_order = len(reference) == len(current) and reference.slice_name.tolist() == current.slice_name.tolist()
    delta = (reference.adapted_dice_average.to_numpy() - current.released_overall_dice.to_numpy()) if same_order else np.asarray([np.inf])
    max_difference = float(np.nanmax(np.abs(delta)))
    reference_hash = json.loads(exp4_json.read_text()).get("prediction_sha256")
    hash_equal = metadata.get("prediction_sha256") == reference_hash
    passed = bool(same_order and max_difference <= 1e-8 and hash_equal)
    lines = [
        "EXP-5 Released seed=2026 vs EXP-4 Released equivalence",
        f"record_count_current={len(current)}", f"record_count_reference={len(reference)}",
        f"slice_names_and_order_equal={same_order}", f"max_released_overall_dice_difference={max_difference:.12f}",
        "tolerance=1e-8", f"prediction_sha256_equal={hash_equal}", f"PASS={passed}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("Released equivalence failed; stopping EXP-5")


def run(data_root: Path, checkpoint: Path, output: Path, seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy="released")
    exp3 = load_exp3_values(seed)
    records, volume_values = [], defaultdict(list)
    digest = hashlib.sha256(); started = time.time()

    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        with torch.inference_mode():
            released_probability = model(image.to(device), [name])
        diag = dict(model.last_diag)
        anchor_probability = model.last_anchor_probability
        released_prediction = released_probability.argmax(1).cpu().numpy()[0]
        anchor_prediction = anchor_probability.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        anchor_values, present, _ = present_dice(anchor_prediction, target)
        released_values, _, _ = present_dice(released_prediction, target)
        _, anchor_overall = whole_prediction_dice(anchor_prediction, target)
        _, released_overall = whole_prediction_dice(released_prediction, target)
        probability_values, _, _ = class_probability_metrics(anchor_probability, released_probability)
        top1, top5_mean, top5_min = retrieval_values(diag)
        gate = dict(diag.get("gate", {}))
        prediction_digest_update(digest, domain, name, released_prediction)
        for offset, (class_id, class_name) in enumerate(CLASSES):
            exp3_row = exp3.loc[(global_index, class_name)]
            values = probability_values[offset]
            global_agreement = float(exp3_row.global_proto_agreement) if np.isfinite(exp3_row.global_proto_agreement) else np.nan
            correction_norm = float(gate.get(f"corr_released_{class_name}", np.nan))
            row = {
                "seed": seed, "global_index": global_index, "domain": domain, "volume_id": volume,
                "slice_name": name, "z_index": z_index, "class_id": class_id, "class_name": class_name,
                "is_sft": bool(diag.get("is_sft", False)), "gt_present": present[offset],
                "anchor_dice": anchor_values[offset], "released_dice": released_values[offset],
                "gain": float(released_values[offset] - anchor_values[offset]) if present[offset] else np.nan,
                "beneficial_any": float(released_values[offset] > anchor_values[offset]) if present[offset] else np.nan,
                "harmful_any": float(released_values[offset] < anchor_values[offset]) if present[offset] else np.nan,
                "beneficial_1pp": float(released_values[offset] - anchor_values[offset] > .01) if present[offset] else np.nan,
                "harmful_1pp": float(released_values[offset] - anchor_values[offset] < -.01) if present[offset] else np.nan,
                "beneficial_5pp": float(released_values[offset] - anchor_values[offset] > .05) if present[offset] else np.nan,
                "harmful_5pp": float(released_values[offset] - anchor_values[offset] < -.05) if present[offset] else np.nan,
                "neutral_1pp": float(abs(released_values[offset] - anchor_values[offset]) <= .01) if present[offset] else np.nan,
                **values,
                "ccd": diag.get("ccd"), "ccd_reliability": -float(diag["ccd"]) if diag.get("ccd") is not None else np.nan,
                "top1_similarity": top1, "top5_mean_similarity": top5_mean, "top5_min_similarity": top5_min,
                "global_proto_agreement": global_agreement, "correction_norm": correction_norm,
                "correction_ratio": np.nan, "composite_A": np.nan, "composite_B": np.nan, "composite_C": np.nan,
                "anchor_overall_dice": anchor_overall, "released_overall_dice": released_overall,
            }
            records.append(row)
        volume_values[(domain, volume)].append({"z_index": z_index, "anchor_prediction": anchor_prediction,
                                                 "released_prediction": released_prediction, "target": target})
        if (global_index + 1) % 200 == 0:
            print(f"EXP-5 seed={seed}: {global_index + 1}/{len(stream.items)}", flush=True)

    frame = pd.DataFrame(records)
    exp4_path = ROOT / "results/exp4/raw" / f"released_seed{seed}.csv"
    exp4 = pd.read_csv(exp4_path).set_index("global_index")
    for class_name in [name for _, name in CLASSES]:
        mask = frame.class_name == class_name
        frame.loc[mask, "correction_norm"] = exp4.loc[frame.loc[mask, "global_index"], f"corr_released_{class_name}"].to_numpy()
        frame.loc[mask, "correction_ratio"] = exp4.loc[frame.loc[mask, "global_index"], "correction_norm_ratio"].to_numpy()
    frame["composite_A"] = frame.prob_change * (1.0 - frame.global_proto_agreement)
    frame["composite_B"] = frame.prob_change * np.maximum(-frame.delta_confidence, 0.0)
    frame["composite_C"] = frame.correction_ratio * (1.0 - frame.global_proto_agreement)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[FIELDS].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    case_rows = build_case_summary(volume_values, seed)
    metadata = {
        "seed": seed, "variant": "released", "batch_size": 1, "data_root": str(data_root),
        "checkpoint": str(checkpoint), "stream_order": "B -> C -> D", "num_slices": int(frame.global_index.nunique()),
        "num_query_classes": len(frame), "elapsed_seconds": time.time() - started, "device": str(device),
        "eps": EPS, "gt_used_only_for": ["dice", "gain", "oracle", "statistical_analysis"],
        "prediction_sha256": digest.hexdigest(), "case_summary": case_rows,
    }
    output.with_suffix(".json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False))
    if seed == 2026:
        verify_equivalence(frame, metadata,
                           ROOT / "results/exp4/raw/released_seed2026.csv",
                           ROOT / "results/exp4/raw/released_seed2026.json",
                           output.parent / "released_equivalence.txt")
    print(json.dumps({"seed": seed, "num_slices": int(frame.global_index.nunique()),
                      "num_query_classes": len(frame), "elapsed_seconds": metadata["elapsed_seconds"]}, indent=2), flush=True)


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
