#!/usr/bin/env python3
"""EXP-8: read-only SABE BatchNorm input diagnostics.

The full runner observes the tensors entering the BatchNorm2d modules during
the already-released forward phases.  The pre-hook always returns ``None``
and never edits an input or output.  No diagnostic-only model forward is
performed here.
"""

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
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp0"))
from run_exp0 import ProcessedStream, dice, load_model  # noqa: E402


DOMAINS = ["B", "C", "D"]
PRIMARY_PHASES = ("sabe_encoder", "sabe_decoder")
EPS = 1e-8


class BNInputRecorder:
    """Read-only BN pre-hook; only the latest forward is retained."""

    def __init__(self, adapted_model, phase_owner):
        self.adapted_model = adapted_model
        self.phase_owner = phase_owner
        self.records: dict[str, dict[str, dict[str, object]]] = {}
        self.handles = []
        self.layer_names = []
        for name, module in adapted_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                self.layer_names.append(name)
                self.handles.append(module.register_forward_pre_hook(self._hook(name)))

    def _hook(self, layer_name):
        def hook(_module, inputs):
            phase = getattr(self.phase_owner, "exp8_bn_phase", None)
            if phase is None:
                return None
            if len(inputs) != 1 or not torch.is_tensor(inputs[0]):
                raise RuntimeError(f"EXP-8 BN hook received invalid input at {layer_name}")
            # Detach, calculate, and store diagnostics only.  Returning None
            # is essential: the forward input is neither replaced nor edited.
            x = inputs[0].detach().double()
            if x.ndim != 4:
                raise RuntimeError(f"EXP-8 expected NCHW at {layer_name}: {tuple(x.shape)}")
            mu = x.mean(dim=(2, 3))
            m2 = (x * x).mean(dim=(2, 3))
            var = torch.clamp(m2 - mu * mu, min=0.0)
            direct_mu = x.mean(dim=(0, 2, 3))
            direct_var = x.var(dim=(0, 2, 3), unbiased=False)
            self.records.setdefault(phase, {})[layer_name] = {
                "input_shape": tuple(x.shape),
                "mu": mu.cpu().numpy(),
                "m2": m2.cpu().numpy(),
                "var": var.cpu().numpy(),
                "sigma": torch.sqrt(var + EPS).cpu().numpy(),
                "direct_mu": direct_mu.cpu().numpy(),
                "direct_var": direct_var.cpu().numpy(),
            }
            return None
        return hook

    def reset(self):
        self.records = {}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def _metric(prediction, target, class_id=None):
    if class_id is None:
        prediction = prediction > 0
        target = target > 0
    else:
        prediction = prediction == class_id
        target = target == class_id
    return float((2.0 * np.logical_and(prediction, target).sum() + 1e-5) /
                 (prediction.sum() + target.sum() + 1e-5))


def _shift(mu, sigma, ref_mu, ref_sigma):
    mean_shift = float(np.mean(np.abs(mu - ref_mu) / (ref_sigma + EPS)))
    scale_shift = float(np.mean(np.abs(np.log((sigma + EPS) / (ref_sigma + EPS)))))
    return mean_shift, scale_shift, 0.5 * (mean_shift + scale_shift)


def _aggregate_stats(record):
    mu = np.asarray(record["mu"], dtype=np.float64)
    m2 = np.asarray(record["m2"], dtype=np.float64)
    mu_equal = mu.mean(axis=0)
    m2_equal = m2.mean(axis=0)
    var_equal = m2_equal - mu_equal * mu_equal
    sigma_equal = np.sqrt(np.maximum(var_equal, 0.0) + EPS)
    return mu, m2, mu_equal, m2_equal, var_equal, sigma_equal


def _weighted_stats(mu, m2, similarities, distances=None):
    count = len(similarities)
    if count == 0:
        raise RuntimeError("weighted statistics require at least one memory")
    values = np.asarray(similarities, dtype=np.float64)
    if distances is None:
        scores = values - values.max()
        weights = count * np.exp(scores) / np.exp(scores).sum()
    else:
        scores = -np.asarray(distances, dtype=np.float64)
        scores -= scores.max()
        weights = count * np.exp(scores) / np.exp(scores).sum()
    weighted_mu = (mu[0] + np.sum(weights[:, None] * mu[1:], axis=0)) / (count + 1.0)
    weighted_m2 = (m2[0] + np.sum(weights[:, None] * m2[1:], axis=0)) / (count + 1.0)
    weighted_var = weighted_m2 - weighted_mu * weighted_mu
    weighted_sigma = np.sqrt(np.maximum(weighted_var, 0.0) + EPS)
    return weights, weighted_mu, weighted_sigma


def _layer_rows(global_index, domain, query_name, is_sft, recorder, retrieval_names,
                similarities, sanity):
    rows, distance_rows, weighted_rows = [], [], []
    count = len(retrieval_names)
    if count == 0:
        return rows, distance_rows, weighted_rows
    if len(similarities) != count:
        raise RuntimeError("retrieval names/similarities length mismatch")
    for phase in PRIMARY_PHASES:
        phase_records = recorder.records.get(phase, {})
        expected_prefix = "enc." if phase == "sabe_encoder" else "dec1."
        phase_layers = [name for name in recorder.layer_names if name.startswith(expected_prefix)]
        for layer_name in phase_layers:
            if layer_name not in phase_records:
                raise RuntimeError(f"missing EXP-8 hook record: {phase}/{layer_name}")
            record = phase_records[layer_name]
            shape = tuple(record["input_shape"])
            if shape[0] != count + 1:
                raise RuntimeError(f"BN batch/retrieval mismatch at {phase}/{layer_name}: {shape}/{count}")
            mu, m2, mu_equal, _m2_equal, var_equal, sigma_equal = _aggregate_stats(record)
            direct_mu = np.asarray(record["direct_mu"], dtype=np.float64)
            direct_var = np.asarray(record["direct_var"], dtype=np.float64)
            sanity["max_equal_mu_error"] = max(
                sanity["max_equal_mu_error"], float(np.max(np.abs(mu_equal - direct_mu))))
            sanity["max_equal_var_error"] = max(
                sanity["max_equal_var_error"], float(np.max(np.abs(var_equal - direct_var))))
            if sanity["max_equal_mu_error"] > 1e-6 or sanity["max_equal_var_error"] > 1e-6:
                raise RuntimeError("EXP-8 equal-statistics reconstruction failed")

            # For encoder layers, the reference is the same layer's query-only
            # input from phase A.  A decoder query-only call does not exist in
            # the released path, so its reference is sample 0 of SABE itself.
            if phase == "sabe_encoder":
                query_record = recorder.records.get("query_encoder_initial", {}).get(layer_name)
                if query_record is None:
                    raise RuntimeError(f"missing query-only encoder record: {layer_name}")
                ref_mu = np.asarray(query_record["mu"], dtype=np.float64)[0]
                ref_sigma = np.asarray(query_record["sigma"], dtype=np.float64)[0]
            else:
                ref_mu = mu[0]
                ref_sigma = np.sqrt(np.maximum(np.asarray(record["var"])[0], 0.0) + EPS)
            mean_shift, scale_shift, equal_shift = _shift(
                mu_equal, sigma_equal, ref_mu, ref_sigma)

            query_mu, query_m2 = mu[0], m2[0]
            memory_mu, memory_m2 = mu[1:], m2[1:]
            query_sigma = np.sqrt(np.maximum(np.asarray(record["var"])[0], 0.0) + EPS)
            memory_sigma = np.sqrt(np.maximum(np.asarray(record["var"])[1:], 0.0) + EPS)
            mean_distances = np.mean(np.abs(memory_mu - query_mu) / (query_sigma + EPS), axis=1)
            scale_distances = np.mean(
                np.abs(np.log((memory_sigma + EPS) / (query_sigma[None, :] + EPS))), axis=1)
            distances = 0.5 * (mean_distances + scale_distances)

            sim_weights, sim_mu, sim_sigma = _weighted_stats(mu, m2, similarities)
            prox_weights, prox_mu, prox_sigma = _weighted_stats(mu, m2, similarities, distances)
            _, _, sim_shift = _shift(sim_mu, sim_sigma, ref_mu, ref_sigma)
            _, _, prox_shift = _shift(prox_mu, prox_sigma, ref_mu, ref_sigma)
            rel_sim = (equal_shift - sim_shift) / (equal_shift + EPS)
            rel_prox = (equal_shift - prox_shift) / (equal_shift + EPS)
            category = "encoder" if phase == "sabe_encoder" else "decoder"

            rows.append({
                "global_index": global_index, "domain": domain, "query_name": query_name,
                "is_sft": bool(is_sft), "phase": phase, "layer_name": layer_name,
                "batch_size": shape[0], "num_memories": count,
                "bn_shift_equal": equal_shift, "mean_shift_equal": mean_shift,
                "scale_shift_equal": scale_shift,
                "bn_shift_similarity_weighted": sim_shift,
                "bn_shift_proximity_weighted": prox_shift,
                "relative_reduction_similarity": rel_sim,
                "relative_reduction_proximity": rel_prox,
                "memory_distance_mean": float(np.mean(distances)),
                "memory_distance_std": float(np.std(distances)),
                "memory_distance_min": float(np.min(distances)),
                "memory_distance_max": float(np.max(distances)),
            })
            for rank in range(count):
                distance_rows.append({
                    "global_index": global_index, "domain": domain, "query_name": query_name,
                    "is_sft": bool(is_sft), "phase": phase, "layer_name": layer_name,
                    "memory_rank": rank + 1, "memory_name": retrieval_names[rank],
                    "retrieval_similarity": float(similarities[rank]),
                    "bn_mean_distance": float(mean_distances[rank]),
                    "bn_scale_distance": float(scale_distances[rank]),
                    "bn_distance": float(distances[rank]),
                })
            weighted_rows.append({
                "global_index": global_index, "domain": domain, "query_name": query_name,
                "is_sft": bool(is_sft), "phase": phase, "layer_name": layer_name,
                "num_memories": count, "query_weight": 1.0,
                "similarity_memory_weight_sum": float(sim_weights.sum()),
                "proximity_memory_weight_sum": float(prox_weights.sum()),
                "similarity_total_mass": float(1.0 + sim_weights.sum()),
                "proximity_total_mass": float(1.0 + prox_weights.sum()),
                "bn_shift_equal": equal_shift, "bn_shift_similarity_weighted": sim_shift,
                "bn_shift_proximity_weighted": prox_shift,
                "relative_reduction_similarity": rel_sim,
                "relative_reduction_proximity": rel_prox,
                "bn_distance_mean": float(np.mean(distances)),
            })
            sanity["num_layer_records"] += 1
            sanity["max_similarity_mass_error"] = max(
                sanity["max_similarity_mass_error"], abs(float(sim_weights.sum()) - count))
            sanity["max_proximity_mass_error"] = max(
                sanity["max_proximity_mass_error"], abs(float(prox_weights.sum()) - count))
            if abs(float(sim_weights.sum()) - count) > 1e-10:
                raise RuntimeError("similarity memory mass is not K")
    return rows, distance_rows, weighted_rows


def run(mode: str, data_root: Path, checkpoint: Path, output_dir: Path,
        seed: int = 2026, limit: int | None = None) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    adapted = load_model(mode, checkpoint, device, admission_policy="released", fusion_mode="released")
    recorder = BNInputRecorder(adapted.model, adapted) if mode == "full" else None
    metrics, layer_rows, distance_rows, weighted_rows = [], [], [], []
    prediction_digest = hashlib.sha256()
    started = time.time()
    sanity = {
        "hook_returned_none": True, "max_equal_mu_error": 0.0,
        "max_equal_var_error": 0.0, "max_similarity_mass_error": 0.0,
        "max_proximity_mass_error": 0.0, "num_layer_records": 0,
        "topk_order_checks": 0,
    }
    total = min(limit, len(stream.items)) if limit is not None else len(stream.items)
    try:
        for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
            if global_index >= total:
                break
            if recorder is not None:
                recorder.reset()
            with torch.inference_mode():
                output = adapted(image.to(device), [name])
            prediction = output.argmax(1).cpu().numpy()[0]
            prediction_digest.update(f"{domain}\t{name}\n".encode())
            prediction_digest.update(np.asarray(prediction, dtype=np.int64).tobytes())
            target = label.numpy()[0]
            row = {
                "global_index": global_index, "domain": domain, "query_name": name,
                "volume": volume, "z_index": z_index,
                "dice_foreground": _metric(prediction, target),
                "dice_lv": _metric(prediction, target, 1),
                "dice_myo": _metric(prediction, target, 2),
                "dice_rv": _metric(prediction, target, 3),
                "dice_average": float(np.mean([_metric(prediction, target, c) for c in (1, 2, 3)])),
            }
            if mode == "full":
                retrieval_names = list(adapted.last_localcorr_retrieved_names)
                similarities = list(adapted.last_localcorr_retrieved_similarities)
                retrieval_diag = getattr(adapted.pool, "last_retrieval_diag", {})
                if retrieval_names != list(adapted.last_diag.get("retrieved_names", [])):
                    raise RuntimeError("EXP-8 retrieval names disagree with released retrieval_diag")
                k = len(retrieval_names)
                if k != len(similarities) or k != len(retrieval_diag.get("indices", [])):
                    raise RuntimeError("EXP-8 actual Top-K disagrees with retrieval_diag")
                if k:
                    expected_shape = 1 + k
                    for phase in PRIMARY_PHASES:
                        for record in recorder.records.get(phase, {}).values():
                            if record["input_shape"][0] != expected_shape:
                                raise RuntimeError("EXP-8 SABE batch composition mismatch")
                    sanity["topk_order_checks"] += 1
                    layer_batch, memory_batch, weighted_batch = _layer_rows(
                        global_index, domain, name, adapted.last_diag.get("is_sft", False),
                        recorder, retrieval_names, similarities, sanity)
                    layer_rows.extend(layer_batch); distance_rows.extend(memory_batch)
                    weighted_rows.extend(weighted_batch)
                    by_category = {"encoder": [r for r in layer_batch if r["phase"] == "sabe_encoder"],
                                   "decoder": [r for r in layer_batch if r["phase"] == "sabe_decoder"]}
                    for category, values in by_category.items():
                        row[f"bn_shift_{category}"] = float(np.mean([r["bn_shift_equal"] for r in values]))
                        row[f"similarity_shift_reduction_{category}"] = float(np.mean([r["relative_reduction_similarity"] for r in values]))
                        row[f"proximity_shift_reduction_{category}"] = float(np.mean([r["relative_reduction_proximity"] for r in values]))
                    all_values = layer_batch
                    row["bn_shift_all"] = float(np.mean([r["bn_shift_equal"] for r in all_values]))
                    row["similarity_shift_reduction_all"] = float(np.mean([r["relative_reduction_similarity"] for r in all_values]))
                    row["proximity_shift_reduction_all"] = float(np.mean([r["relative_reduction_proximity"] for r in all_values]))
                    row["bn_shift_max_layer"] = max(all_values, key=lambda r: r["bn_shift_equal"])["layer_name"]
                    row["bn_shift_max"] = float(max(r["bn_shift_equal"] for r in all_values))
                    row["bn_shift_median"] = float(np.median([r["bn_shift_equal"] for r in all_values]))
                    row["bn_distance_min"] = float(np.min([r["bn_distance"] for r in memory_batch])) if memory_batch else np.nan
                    row["bn_distance_max"] = float(np.max([r["bn_distance"] for r in memory_batch])) if memory_batch else np.nan
                    row["bn_distance_mean"] = float(np.mean([r["bn_distance"] for r in memory_batch])) if memory_batch else np.nan
                    row["bn_distance_std"] = float(np.std([r["bn_distance"] for r in memory_batch])) if memory_batch else np.nan
                    row["bn_distance_range"] = row["bn_distance_max"] - row["bn_distance_min"]
                else:
                    for key in ("bn_shift_encoder", "bn_shift_decoder", "bn_shift_all",
                                "similarity_shift_reduction_encoder", "similarity_shift_reduction_decoder",
                                "similarity_shift_reduction_all", "proximity_shift_reduction_encoder",
                                "proximity_shift_reduction_decoder", "proximity_shift_reduction_all",
                                "bn_shift_max", "bn_shift_median", "bn_distance_min",
                                "bn_distance_max", "bn_distance_mean", "bn_distance_std", "bn_distance_range"):
                        row[key] = np.nan
                    row["bn_shift_max_layer"] = ""
                row["num_memories"] = k
                row["is_sft"] = bool(adapted.last_diag.get("is_sft", False))
            metrics.append(row)
            if (global_index + 1) % 200 == 0:
                print(f"EXP-8 {mode}: {global_index + 1}/{total}", flush=True)
    finally:
        if recorder is not None:
            recorder.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{mode}_slice_metrics_seed{seed}.csv"
    _write_rows(metrics_path, metrics)
    if mode == "full":
        _write_rows(output_dir / "bn_layer_query_statistics.csv", layer_rows)
        _write_rows(output_dir / "bn_memory_distance_statistics.csv", distance_rows)
        _write_rows(output_dir / "weighted_statistics_diagnostic.csv", weighted_rows)
    payload = {
        "mode": mode, "seed": seed, "num_slices": len(metrics),
        "prediction_sha256": prediction_digest.hexdigest(),
        "data_root": str(data_root), "checkpoint": str(checkpoint),
        "device": str(device), "elapsed_seconds": time.time() - started,
        "num_forward_calls": len(metrics),
        "diagnostic_forward_calls": 0,
        "bn_layer_count": len(recorder.layer_names) if recorder is not None else 0,
        "sanity": sanity,
    }
    (output_dir / f"{mode}_run_metadata_seed{seed}.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "sff"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.mode, args.data_root, args.checkpoint, args.output_dir, args.seed, args.limit)


if __name__ == "__main__":
    main()
