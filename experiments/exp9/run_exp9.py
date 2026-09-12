#!/usr/bin/env python3
"""EXP-9: output-only post-adaptation confidence-delta selection.

The released SicTTA forward is executed exactly once.  The selector observes
the already-available anchor and adapted probabilities, then optionally mixes
their outputs.  It never changes model parameters, retrieval, admission, or
memory updates.
"""

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
from run_exp0 import ProcessedStream, assd, dice, json_safe, load_model  # noqa: E402


DOMAINS = ["B", "C", "D"]
CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
VARIANTS = ("released", "identity", "zero_hard", "mad_hard", "mad_soft", "mad_soft_no_rv")
EPS = 1e-8
FIELDS = [
    "variant", "seed", "global_index", "domain", "volume_id", "slice_name", "z_index",
    "is_sft", "ccd", "ccd_threshold", "ccd_margin", "pool_size_before", "pool_size_after",
    "history_len_before", "history_len_after",
]
for suffix in ("lv", "myo", "rv"):
    FIELDS.extend([
        f"anchor_confidence_{suffix}", f"adapted_confidence_{suffix}",
        f"delta_confidence_{suffix}", f"history_median_{suffix}", f"history_mad_{suffix}",
        f"robust_z_{suffix}", f"class_gate_{suffix}",
    ])
FIELDS.extend([
    "gate_map_mean", "gate_map_std", "gate_map_min", "gate_map_max", "fallback_pixel_mass",
    "anchor_dice_lv", "anchor_dice_myo", "anchor_dice_rv", "anchor_dice_average",
    "released_dice_lv", "released_dice_myo", "released_dice_rv", "released_dice_average",
    "selected_dice_lv", "selected_dice_myo", "selected_dice_rv", "selected_dice_average",
    "slice_elapsed_seconds",
])


def class_confidence(probability: torch.Tensor) -> torch.Tensor:
    """EXP-3 class confidence, for the three foreground classes."""
    probability = probability[0].detach().float()
    masses = probability[1:].sum(dim=(1, 2))
    return probability[1:].square().sum(dim=(1, 2)) / (masses + EPS)


class ConfidenceDeltaSelector:
    """Past-only class-wise output selector with optional robust calibration."""

    def __init__(self, variant: str, window: int, warmup: int,
                 robust_lambda: float, temperature: float):
        if variant not in VARIANTS:
            raise ValueError(variant)
        if window <= 0 or warmup < 0 or warmup > window:
            raise ValueError(f"invalid window/warmup: {window}/{warmup}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.variant = variant
        self.window = int(window)
        self.warmup = int(warmup)
        self.robust_lambda = float(robust_lambda)
        self.temperature = float(temperature)
        self.history = [[] for _ in CLASSES]

    def _statistics(self, class_index: int) -> tuple[float, float, float]:
        values = np.asarray(self.history[class_index], dtype=np.float64)
        if not len(values):
            return np.nan, np.nan, np.nan
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return median, mad, 1.4826 * mad

    def select(self, anchor: torch.Tensor, adapted: torch.Tensor):
        anchor_conf = class_confidence(anchor)
        adapted_conf = class_confidence(adapted)
        delta = adapted_conf - anchor_conf
        delta_values = delta.detach().cpu().numpy().astype(float)
        history_len_before = min(len(values) for values in self.history)
        medians, mads, robust_z = [], [], []
        gates = np.ones(len(CLASSES), dtype=np.float64)

        for index, value in enumerate(delta_values):
            median, mad, scale = self._statistics(index)
            medians.append(median); mads.append(mad)
            z_value = ((value - median) / max(scale, EPS)
                       if np.isfinite(median) and np.isfinite(scale) else np.nan)
            robust_z.append(z_value)
            if self.variant == "zero_hard":
                gates[index] = float(value >= 0.0)
            elif self.variant in {"mad_hard", "mad_soft", "mad_soft_no_rv"} and history_len_before >= self.warmup:
                if self.variant == "mad_hard":
                    gates[index] = float(z_value >= -self.robust_lambda)
                else:
                    # Leave the non-risk region untouched; attenuate smoothly
                    # only after crossing the robust lower-tail threshold.
                    argument = np.clip((z_value + self.robust_lambda) / self.temperature, -40.0, 0.0)
                    gates[index] = float(np.exp(argument))

        # Time-boxed EXP-9 diagnostic: preserve the released path for RV and
        # allow confidence-delta selection only for LV/MYO.
        if self.variant == "mad_soft_no_rv":
            gates[2] = 1.0

        # Background remains on the adapted path.  Anchor posterior mass turns
        # class decisions into one valid spatial convex mixture.
        if self.variant in {"released", "identity"}:
            gates[:] = 1.0
        if self.variant in {"released", "identity"}:
            # Preserve exact tensor identity for the regression-control paths.
            gate_map = torch.ones_like(anchor[:, :1])
            selected = adapted
        else:
            gate_vector = torch.tensor([1.0, *gates.tolist()], device=anchor.device,
                                       dtype=anchor.dtype).view(1, 4, 1, 1)
            gate_map = (anchor * gate_vector).sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            selected = gate_map * adapted + (1.0 - gate_map) * anchor
            selected = selected / selected.sum(dim=1, keepdim=True).clamp_min(EPS)

        for index, value in enumerate(delta_values):
            self.history[index].append(float(value))
            self.history[index] = self.history[index][-self.window:]
        diagnostics = {
            "anchor_confidence": anchor_conf.detach().cpu().tolist(),
            "adapted_confidence": adapted_conf.detach().cpu().tolist(),
            "delta_confidence": delta_values.tolist(), "history_median": medians,
            "history_mad": mads, "robust_z": robust_z, "class_gate": gates.tolist(),
            "history_len_before": history_len_before,
            "history_len_after": min(len(values) for values in self.history),
            "gate_map_mean": float(gate_map.mean().item()),
            "gate_map_std": float(gate_map.std(unbiased=False).item()),
            "gate_map_min": float(gate_map.min().item()),
            "gate_map_max": float(gate_map.max().item()),
            "fallback_pixel_mass": float((1.0 - gate_map).mean().item()),
        }
        return selected, diagnostics


def class_dice(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    return [float(dice(prediction, target, class_id)) for class_id, _ in CLASSES]


def prediction_digest_update(digest, domain, name, prediction):
    digest.update(f"{domain}\t{name}\n".encode())
    digest.update(np.asarray(prediction, dtype=np.int64).tobytes())


def summarize(records, volume_masks, variant, seed):
    rows, cases = [], []
    for domain in DOMAINS:
        subset = [row for row in records if row["domain"] == domain]
        if not subset:
            continue
        for (case_domain, volume), slices in volume_masks.items():
            if case_domain != domain:
                continue
            ordered = [slices[z] for z in sorted(slices)]
            prediction = np.stack([item[0] for item in ordered])
            target = np.stack([item[1] for item in ordered])
            values = class_dice(prediction, target)
            assd_values = [assd(prediction, target, class_id) for class_id, _ in CLASSES]
            cases.append({
                "variant": variant, "seed": seed, "domain": domain, "volume_id": volume,
                "dice_lv": values[0], "dice_myo": values[1], "dice_rv": values[2],
                "dice_average": float(np.mean(values)), "assd_lv": assd_values[0],
                "assd_myo": assd_values[1], "assd_rv": assd_values[2],
                "assd_average": float(np.nanmean(assd_values)), "num_slices": len(ordered),
            })
        rows.append(_summary_row(subset, variant, seed, domain))
    rows.append(_summary_row(records, variant, seed, "All"))
    return rows, cases


def _summary_row(records, variant, seed, domain):
    return {
        "variant": variant, "seed": seed, "domain": domain, "num_slices": len(records),
        "dice_lv": float(np.mean([row["selected_dice_lv"] for row in records])),
        "dice_myo": float(np.mean([row["selected_dice_myo"] for row in records])),
        "dice_rv": float(np.mean([row["selected_dice_rv"] for row in records])),
        "dice_average": float(np.mean([row["selected_dice_average"] for row in records])),
        "fallback_pixel_mass": float(np.mean([row["fallback_pixel_mass"] for row in records])),
        "fallback_slice_rate": float(np.mean([
            any(row[f"class_gate_{suffix}"] < 1.0 - EPS for suffix in ("lv", "myo", "rv"))
            for row in records
        ])),
    }


def verify_equivalence(frame, metadata, reference_csv, reference_json, output_path, label):
    reference_long = pd.read_csv(reference_csv).sort_values(["global_index", "class_name"])
    reference = reference_long.drop_duplicates("global_index")
    current = frame.sort_values("global_index").drop_duplicates("global_index")
    same_order = len(reference) == len(current) and reference.slice_name.tolist() == current.slice_name.tolist()
    deltas = []
    if same_order:
        for suffix in ("lv", "myo", "rv"):
            expected = reference_long[reference_long.class_name == suffix].sort_values("global_index")
            deltas.extend(np.abs(expected.released_dice.to_numpy()
                                 - current[f"selected_dice_{suffix}"].to_numpy()).tolist())
    max_difference = float(np.nanmax(deltas)) if deltas else float("inf")
    reference_hash = json.loads(reference_json.read_text())["prediction_sha256"]
    hash_equal = metadata["prediction_sha256"] == reference_hash
    passed = bool(same_order and max_difference <= 1e-8 and hash_equal)
    lines = [
        f"EXP-9 {label} seed=2026 vs EXP-5 Released equivalence",
        f"record_count_current={len(current)}", f"record_count_reference={len(reference)}",
        f"slice_names_and_order_equal={same_order}", f"max_class_dice_difference={max_difference:.12f}",
        "tolerance=1e-8", f"prediction_sha256_equal={hash_equal}", f"PASS={passed}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError(f"{label} equivalence failed; stopping EXP-9")


def run(data_root: Path, checkpoint: Path, output: Path, seed: int, variant: str,
        window: int, warmup: int, robust_lambda: float, temperature: float,
        limit: int | None = None):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy="released")
    selector = ConfidenceDeltaSelector(variant, window, warmup, robust_lambda, temperature)
    records, volume_masks = [], defaultdict(dict)
    digest = hashlib.sha256(); started = time.time()
    total = min(limit, len(stream.items)) if limit is not None else len(stream.items)
    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if global_index >= total:
            break
        slice_started = time.perf_counter()
        with torch.inference_mode():
            released_output = model(image.to(device), [name])
            anchor_output = model.last_anchor_probability
            selected_output, selector_diag = selector.select(anchor_output, released_output)
        target = label.numpy()[0]
        anchor_prediction = anchor_output.argmax(1).cpu().numpy()[0]
        released_prediction = released_output.argmax(1).cpu().numpy()[0]
        selected_prediction = selected_output.argmax(1).cpu().numpy()[0]
        anchor_values = class_dice(anchor_prediction, target)
        released_values = class_dice(released_prediction, target)
        selected_values = class_dice(selected_prediction, target)
        diag = dict(model.last_diag)
        row = {
            "variant": variant, "seed": seed, "global_index": global_index, "domain": domain,
            "volume_id": volume, "slice_name": name, "z_index": z_index,
            "is_sft": bool(diag.get("is_sft", False)), "ccd": diag.get("ccd"),
            "ccd_threshold": diag.get("ccd_threshold"), "ccd_margin": diag.get("ccd_margin"),
            "pool_size_before": diag.get("pool_before", {}).get("name", 0),
            "pool_size_after": diag.get("pool_after", {}).get("name", 0),
            "history_len_before": selector_diag["history_len_before"],
            "history_len_after": selector_diag["history_len_after"],
            "gate_map_mean": selector_diag["gate_map_mean"], "gate_map_std": selector_diag["gate_map_std"],
            "gate_map_min": selector_diag["gate_map_min"], "gate_map_max": selector_diag["gate_map_max"],
            "fallback_pixel_mass": selector_diag["fallback_pixel_mass"],
            "anchor_dice_lv": anchor_values[0], "anchor_dice_myo": anchor_values[1],
            "anchor_dice_rv": anchor_values[2], "anchor_dice_average": float(np.mean(anchor_values)),
            "released_dice_lv": released_values[0], "released_dice_myo": released_values[1],
            "released_dice_rv": released_values[2], "released_dice_average": float(np.mean(released_values)),
            "selected_dice_lv": selected_values[0], "selected_dice_myo": selected_values[1],
            "selected_dice_rv": selected_values[2], "selected_dice_average": float(np.mean(selected_values)),
            "slice_elapsed_seconds": time.perf_counter() - slice_started,
        }
        for index, (_, suffix) in enumerate(CLASSES):
            for key in ("anchor_confidence", "adapted_confidence", "delta_confidence",
                        "history_median", "history_mad", "robust_z", "class_gate"):
                row[f"{key}_{suffix}"] = selector_diag[key][index]
        records.append(row)
        volume_masks[(domain, volume)][z_index] = (selected_prediction, target)
        prediction_digest_update(digest, domain, name, selected_prediction)
        if (global_index + 1) % 200 == 0:
            print(f"EXP-9 {variant} seed={seed}: {global_index + 1}/{total}", flush=True)

    frame = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[FIELDS].to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    domain_summary, per_case = summarize(records, volume_masks, variant, seed)
    metadata = {
        "variant": variant, "seed": seed, "batch_size": 1, "data_root": str(data_root),
        "checkpoint": str(checkpoint), "stream_order": "B -> C -> D", "num_slices": len(records),
        "device": str(device), "elapsed_seconds": time.time() - started,
        "mean_slice_elapsed_seconds": float(frame.slice_elapsed_seconds.mean()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
        "prediction_sha256": digest.hexdigest(), "domain_summary": domain_summary, "per_case": per_case,
        "selector": {"scope": "output_only", "window": window, "warmup": warmup,
                     "robust_lambda": robust_lambda, "temperature": temperature,
                     "history": "past_only", "background_gate": 1.0},
        "gt_used_only_for": ["Dice", "ASSD"],
    }
    output.with_suffix(".json").write_text(json.dumps(json_safe(metadata), indent=2, allow_nan=False))
    if limit is None and seed == 2026 and variant in {"released", "identity"}:
        label = "Released" if variant == "released" else "Identity"
        verify_equivalence(
            frame, metadata, ROOT / "results/exp5/per_class_utility_seed2026.csv",
            ROOT / "results/exp5/per_class_utility_seed2026.json",
            output.parent.parent / ("released_equivalence.txt" if variant == "released" else "identity_equivalence.txt"),
            label,
        )
    print(json.dumps({"variant": variant, "seed": seed, "num_slices": len(records),
                      "elapsed_seconds": metadata["elapsed_seconds"],
                      "prediction_sha256": metadata["prediction_sha256"]}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--window", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--robust-lambda", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output, args.seed, args.variant, args.window,
        args.warmup, args.robust_lambda, args.temperature, args.limit)


if __name__ == "__main__":
    main()
