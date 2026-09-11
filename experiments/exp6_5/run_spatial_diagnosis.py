#!/usr/bin/env python3
"""EXP-6.5: read-only spatial misalignment diagnosis.

The only model call in the loop is the existing ``model(image)`` call.  All
geometry and alignment operations below are offline diagnostics and never
feed back into SicTTA.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp0"))
from run_exp0 import ProcessedStream, dice, json_safe, load_model


DOMAINS = ["B", "C", "D"]
CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
EPS = 1e-8
MASK_CLASSES = {"fg": None, "lv": 1, "myo": 2, "rv": 3}


def finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def safe_dice(query, memory, class_id=None):
    q = query > 0 if class_id is None else query == class_id
    m = memory > 0 if class_id is None else memory == class_id
    if not q.any() or not m.any():
        return np.nan
    return float(dice(q.astype(np.uint8), m.astype(np.uint8), 1))


def _geometry(probability):
    """Return normalized centroid/scale geometry for [C,H,W] probabilities."""
    probability = np.asarray(probability, dtype=np.float64)
    height, width = probability.shape[-2:]
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    xx, yy = np.meshgrid(x, y)

    def one(mass_map):
        mass = float(mass_map.sum())
        if mass < 1e-6:
            return {"mass": mass, "centroid_x": None, "centroid_y": None,
                    "sigma_x": None, "sigma_y": None, "valid": False}
        mux = float((mass_map * xx).sum() / (mass + EPS))
        muy = float((mass_map * yy).sum() / (mass + EPS))
        sigx = float(np.sqrt(max(0.0, (mass_map * (xx - mux) ** 2).sum() / (mass + EPS))))
        sigy = float(np.sqrt(max(0.0, (mass_map * (yy - muy) ** 2).sum() / (mass + EPS))))
        return {"mass": mass, "centroid_x": mux, "centroid_y": muy,
                "sigma_x": sigx, "sigma_y": sigy, "valid": True}

    result = {"fg": one(probability[1:].sum(axis=0))}
    for class_id, name in CLASSES:
        result[name] = one(probability[class_id])
    return result


def _distance(query_geometry, memory_geometry, key):
    q, m = query_geometry[key], memory_geometry[key]
    if not q["valid"] or not m["valid"]:
        return np.nan
    return float(np.hypot(q["centroid_x"] - m["centroid_x"],
                          q["centroid_y"] - m["centroid_y"]))


def _scale_errors(query_geometry, memory_geometry, key):
    q, m = query_geometry[key], memory_geometry[key]
    if not q["valid"] or not m["valid"]:
        return np.nan, np.nan, np.nan, np.nan
    ratio_x = q["sigma_x"] / (m["sigma_x"] + EPS)
    ratio_y = q["sigma_y"] / (m["sigma_y"] + EPS)
    return (float(ratio_x), float(ratio_y),
            float(abs(np.log(ratio_x + EPS))), float(abs(np.log(ratio_y + EPS))))


def _unit_grid(height, width, device):
    y = (torch.arange(height, device=device, dtype=torch.float32) + .5) / height
    x = (torch.arange(width, device=device, dtype=torch.float32) + .5) / width
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    return xx, yy


def warp_mask(mask, dx=0.0, dy=0.0, scale_x=1.0, scale_y=1.0,
              query_centroid=None, memory_centroid=None):
    """Deterministic nearest-neighbor warp with align_corners=False.

    Unit coordinates denote pixel centers in [0,1].  A positive translation
    moves the memory object toward increasing x/y.  For affine alignment the
    object is scaled around its memory centroid and then placed at the query
    centroid.
    """
    array = np.asarray(mask, dtype=np.float32)
    height, width = array.shape
    tensor = torch.from_numpy(array).view(1, 1, height, width)
    xx, yy = _unit_grid(height, width, tensor.device)
    if query_centroid is None or memory_centroid is None:
        source_x = xx - float(dx)
        source_y = yy - float(dy)
    else:
        source_x = float(memory_centroid[0]) + (xx - float(query_centroid[0])) / float(scale_x)
        source_y = float(memory_centroid[1]) + (yy - float(query_centroid[1])) / float(scale_y)
    grid = torch.stack((source_x * 2 - 1, source_y * 2 - 1), dim=-1).unsqueeze(0)
    result = F.grid_sample(tensor, grid, mode="nearest", padding_mode="zeros",
                           align_corners=False)[0, 0].numpy()
    return result > 0.5


def _unit_test_warp():
    mask = np.zeros((9, 11), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    assert np.array_equal(mask > 0, warp_mask(mask, dx=0.0, dy=0.0))
    shifted = warp_mask(mask, dx=1 / 11, dy=2 / 9)
    assert shifted[4:7, 4:7].all()


def _overlap_bundle(query_gt, memory_gt, query_geometry, memory_geometry):
    values = {}
    for key, class_id in MASK_CLASSES.items():
        if class_id is None:
            target_mask, source_mask = query_gt > 0, memory_gt > 0
        else:
            target_mask, source_mask = query_gt == class_id, memory_gt == class_id
        values[f"{key}_raw"] = safe_dice(query_gt, memory_gt, class_id)
        qgeom, mgeom = query_geometry[key], memory_geometry[key]
        distance = _distance(query_geometry, memory_geometry, key)
        values[f"{key}_distance"] = distance
        if qgeom["valid"] and mgeom["valid"]:
            delta = (qgeom["centroid_x"] - mgeom["centroid_x"],
                     qgeom["centroid_y"] - mgeom["centroid_y"])
            translated = warp_mask(source_mask, dx=delta[0], dy=delta[1])
            values[f"{key}_pred_translation"] = safe_dice(target_mask, translated, 1)
            ratio_x, ratio_y, error_x, error_y = _scale_errors(query_geometry, memory_geometry, key)
            extreme = not (.5 <= ratio_x <= 2.0 and .5 <= ratio_y <= 2.0)
            values[f"{key}_scale_ratio_x"] = ratio_x
            values[f"{key}_scale_ratio_y"] = ratio_y
            values[f"{key}_scale_error_x"] = error_x
            values[f"{key}_scale_error_y"] = error_y
            values[f"{key}_extreme_scale"] = extreme
            values[f"{key}_pred_affine"] = np.nan
            values[f"{key}_pred_affine_clipped"] = np.nan
            if not extreme:
                affine = warp_mask(source_mask, query_centroid=(qgeom["centroid_x"], qgeom["centroid_y"]),
                                   memory_centroid=(mgeom["centroid_x"], mgeom["centroid_y"]),
                                   scale_x=ratio_x, scale_y=ratio_y)
                values[f"{key}_pred_affine"] = safe_dice(target_mask, affine, 1)
            clipped = warp_mask(source_mask, query_centroid=(qgeom["centroid_x"], qgeom["centroid_y"]),
                                memory_centroid=(mgeom["centroid_x"], mgeom["centroid_y"]),
                                scale_x=np.clip(ratio_x, .5, 2.0), scale_y=np.clip(ratio_y, .5, 2.0))
            values[f"{key}_pred_affine_clipped"] = safe_dice(target_mask, clipped, 1)
            values[f"{key}_pred_translation_delta"] = values[f"{key}_pred_translation"] - values[f"{key}_raw"]
            values[f"{key}_pred_affine_delta"] = values[f"{key}_pred_affine"] - values[f"{key}_raw"]
            values[f"{key}_pred_affine_clipped_delta"] = values[f"{key}_pred_affine_clipped"] - values[f"{key}_raw"]
        else:
            for suffix in ["scale_ratio_x", "scale_ratio_y", "scale_error_x", "scale_error_y",
                           "pred_translation", "pred_affine", "pred_affine_clipped",
                           "pred_translation_delta", "pred_affine_delta", "pred_affine_clipped_delta"]:
                values[f"{key}_{suffix}"] = np.nan
            values[f"{key}_extreme_scale"] = False

        # GT-centroid oracle is foreground-only in the primary analysis.
        if key == "fg":
            qbinary, mbinary = query_gt > 0, memory_gt > 0
            if qbinary.any() and mbinary.any():
                qzero = np.zeros_like(qbinary, dtype=float)
                mzero = np.zeros_like(mbinary, dtype=float)
                qg = _geometry(np.stack([1.0 - qbinary, qbinary, qzero, qzero]))["fg"]
                mg = _geometry(np.stack([1.0 - mbinary, mbinary, mzero, mzero]))["fg"]
                oracle_delta = (qg["centroid_x"] - mg["centroid_x"], qg["centroid_y"] - mg["centroid_y"])
                oracle = warp_mask(mbinary, dx=oracle_delta[0], dy=oracle_delta[1])
                values["fg_gt_oracle"] = safe_dice(qbinary, oracle, 1)
            else:
                values["fg_gt_oracle"] = np.nan
        else:
            query_class, memory_class = query_gt == class_id, memory_gt == class_id
            if query_class.any() and memory_class.any():
                qzero = np.zeros_like(query_class, dtype=float)
                mzero = np.zeros_like(memory_class, dtype=float)
                qg = _geometry(np.stack([1.0 - query_class, query_class, qzero, qzero]))["fg"]
                mg = _geometry(np.stack([1.0 - memory_class, memory_class, mzero, mzero]))["fg"]
                oracle_delta = (qg["centroid_x"] - mg["centroid_x"], qg["centroid_y"] - mg["centroid_y"])
                oracle = warp_mask(memory_class, dx=oracle_delta[0], dy=oracle_delta[1])
                values[f"{key}_gt_oracle"] = safe_dice(query_class, oracle, 1)
            else:
                values[f"{key}_gt_oracle"] = np.nan

    for key in MASK_CLASSES:
        values[f"{key}_gt_oracle_delta"] = values[f"{key}_gt_oracle"] - values[f"{key}_raw"]
    return values


def pair_record(global_index, domain, query_name, rank, similarity, query_prob,
                memory_prob, query_gt, memory_gt, is_sft, pool_size):
    qgeo, mgeo = _geometry(query_prob), _geometry(memory_prob)
    values = _overlap_bundle(query_gt, memory_gt[0], qgeo, mgeo)
    row = {
        "global_index": global_index, "query_domain": domain, "query_name": query_name,
        "memory_name": memory_gt[1], "memory_rank": rank, "cosine_similarity": float(similarity),
        "is_sft": bool(is_sft), "pool_size": pool_size,
    }
    # memory_gt is a (mask, name) tuple at call sites; keep the API explicit.
    row["memory_name"] = memory_gt[1]
    memory_mask = memory_gt[0]
    row.update({
        "query_fg_mass": qgeo["fg"]["mass"], "memory_fg_mass": mgeo["fg"]["mass"],
        "query_fg_centroid_x": qgeo["fg"]["centroid_x"], "query_fg_centroid_y": qgeo["fg"]["centroid_y"],
        "memory_fg_centroid_x": mgeo["fg"]["centroid_x"], "memory_fg_centroid_y": mgeo["fg"]["centroid_y"],
        "fg_centroid_distance": values["fg_distance"],
        "fg_sigma_q_x": qgeo["fg"]["sigma_x"], "fg_sigma_q_y": qgeo["fg"]["sigma_y"],
        "fg_sigma_m_x": mgeo["fg"]["sigma_x"], "fg_sigma_m_y": mgeo["fg"]["sigma_y"],
        "fg_scale_ratio_x": values["fg_scale_ratio_x"], "fg_scale_ratio_y": values["fg_scale_ratio_y"],
        "fg_scale_error_x": values["fg_scale_error_x"], "fg_scale_error_y": values["fg_scale_error_y"],
        "fg_extreme_scale": values["fg_extreme_scale"],
        "raw_gt_overlap_fg": values["fg_raw"],
        "pred_translation_gt_overlap_fg": values["fg_pred_translation"],
        "pred_affine_gt_overlap_fg": values["fg_pred_affine"],
        "pred_affine_clipped_gt_overlap_fg": values["fg_pred_affine_clipped"],
        "gt_oracle_translation_overlap_fg": values["fg_gt_oracle"],
        "delta_pred_translation_fg": values["fg_pred_translation_delta"],
        "delta_pred_affine_fg": values["fg_pred_affine_delta"],
        "delta_pred_affine_clipped_fg": values["fg_pred_affine_clipped_delta"],
        "delta_gt_oracle_translation_fg": values["fg_gt_oracle_delta"],
    })
    for _, key in CLASSES:
        q, m = qgeo[key], mgeo[key]
        row.update({
            f"query_{key}_mass": q["mass"], f"memory_{key}_mass": m["mass"],
            f"query_{key}_centroid_x": q["centroid_x"], f"query_{key}_centroid_y": q["centroid_y"],
            f"memory_{key}_centroid_x": m["centroid_x"], f"memory_{key}_centroid_y": m["centroid_y"],
            f"{key}_centroid_distance": values[f"{key}_distance"],
            f"{key}_sigma_q_x": q["sigma_x"], f"{key}_sigma_q_y": q["sigma_y"],
            f"{key}_sigma_m_x": m["sigma_x"], f"{key}_sigma_m_y": m["sigma_y"],
            f"{key}_scale_error_x": values[f"{key}_scale_error_x"], f"{key}_scale_error_y": values[f"{key}_scale_error_y"],
            f"{key}_extreme_scale": values[f"{key}_extreme_scale"],
            f"raw_gt_overlap_{key}": values[f"{key}_raw"],
            f"pred_translation_gt_overlap_{key}": values[f"{key}_pred_translation"],
            f"pred_affine_gt_overlap_{key}": values[f"{key}_pred_affine"],
            f"pred_affine_clipped_gt_overlap_{key}": values[f"{key}_pred_affine_clipped"],
            f"delta_pred_translation_{key}": values[f"{key}_pred_translation_delta"],
            f"delta_pred_affine_{key}": values[f"{key}_pred_affine_delta"],
            f"delta_pred_affine_clipped_{key}": values[f"{key}_pred_affine_clipped_delta"],
            f"gt_oracle_translation_overlap_{key}": values[f"{key}_gt_oracle"],
            f"delta_gt_oracle_translation_{key}": values[f"{key}_gt_oracle_delta"],
        })
    return row


def query_record(global_index, domain, name, is_sft, pool_size, qgeo, pool_geometries,
                 pair_rows, topk_names):
    row = {"global_index": global_index, "domain": domain, "query_name": name,
           "is_sft": bool(is_sft), "pool_size": pool_size, "topk_count": len(pair_rows)}
    for key in ["fg", "lv", "myo", "rv"]:
        all_dist = [_distance(qgeo, g["geometry"], key) for g in pool_geometries]
        all_dist = [x for x in all_dist if np.isfinite(x)]
        top_dist = [r[f"{key}_centroid_distance"] for r in pair_rows
                    if np.isfinite(r[f"{key}_centroid_distance"])]
        nearest = sorted(all_dist)[:len(topk_names)]
        row[f"topk_mean_{key}_centroid_distance"] = finite_mean(top_dist)
        row[f"pool_mean_{key}_centroid_distance"] = finite_mean(all_dist)
        row[f"pool_median_{key}_centroid_distance"] = float(np.median(all_dist)) if all_dist else np.nan
        row[f"spatial_nearest_k_mean_{key}_distance"] = finite_mean(nearest)
    # Backward-compatible aliases for the required foreground field names.
    row["topk_mean_centroid_distance"] = row["topk_mean_fg_centroid_distance"]
    row["pool_mean_centroid_distance"] = row["pool_mean_fg_centroid_distance"]
    row["pool_median_centroid_distance"] = row["pool_median_fg_centroid_distance"]
    row["spatial_nearest_k_mean_distance"] = row["spatial_nearest_k_mean_fg_distance"]
    for prefix, field in [("raw", "raw_gt_overlap"), ("pred_translation", "pred_translation_gt_overlap"),
                          ("pred_affine", "pred_affine_gt_overlap"), ("pred_affine_clipped", "pred_affine_clipped_gt_overlap"),
                          ("gt_oracle", "gt_oracle_translation_overlap")]:
        values = [r[f"{field}_fg"] for r in pair_rows]
        row[f"topk_mean_{prefix}_overlap"] = finite_mean(values)
        row[f"topk_best_{prefix}_overlap"] = float(np.nanmax(values)) if np.isfinite(values).any() else np.nan
        if prefix != "raw":
            row[f"topk_{prefix}_gain"] = row[f"topk_mean_{prefix}_overlap"] - row["topk_mean_raw_overlap"]
    row["topk_translation_gain"] = row["topk_mean_pred_translation_overlap"] - row["topk_mean_raw_overlap"]
    row["topk_affine_gain"] = row["topk_mean_pred_affine_overlap"] - row["topk_mean_raw_overlap"]
    row["topk_affine_clipped_gain"] = row["topk_mean_pred_affine_clipped_overlap"] - row["topk_mean_raw_overlap"]
    row["topk_gt_oracle_gain"] = row["topk_mean_gt_oracle_overlap"] - row["topk_mean_raw_overlap"]
    for _, key in CLASSES:
        raw = finite_mean([r[f"raw_gt_overlap_{key}"] for r in pair_rows])
        trans = finite_mean([r[f"pred_translation_gt_overlap_{key}"] for r in pair_rows])
        oracle = finite_mean([r[f"gt_oracle_translation_overlap_{key}"] for r in pair_rows])
        row[f"topk_mean_raw_overlap_{key}"] = raw
        row[f"topk_mean_pred_translation_overlap_{key}"] = trans
        row[f"topk_translation_gain_{key}"] = trans - raw
        row[f"topk_mean_gt_oracle_overlap_{key}"] = oracle
        row[f"topk_gt_oracle_gain_{key}"] = oracle - raw
    return row


def run(data_root, checkpoint, output_dir, seed=2026, limit=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _unit_test_warp()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy="released", fusion_mode="released")
    gt_cache = {}
    pair_rows, query_rows = [], []
    digest = hashlib.sha256()
    started = time.time()
    total = min(limit, len(stream.items)) if limit is not None else len(stream.items)
    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if global_index >= total:
            break
        with torch.inference_mode():
            output = model(image.to(device), [name])
        prediction = output.argmax(1).cpu().numpy()[0]
        digest.update(f"{domain}\t{name}\n".encode())
        digest.update(np.asarray(prediction, dtype=np.int64).tobytes())
        query_prob = model.last_anchor_probability[0].detach().cpu().numpy()
        query_gt = label.numpy()[0]
        diag = model.last_diag
        indices = list(diag.get("retrieved_indices", []))
        similarities = list(diag.get("retrieved_similarities", []))
        names = list(model.last_spatial_memory_names)
        bank = model.last_spatial_anchor_probability_bank
        pair_batch = []
        if bank is not None:
            bank_np = bank.detach().cpu().numpy()
            for rank, (index, similarity) in enumerate(zip(indices, similarities), start=1):
                if index >= len(names) or index >= len(bank_np) or names[index] not in gt_cache:
                    continue
                memory_name = names[index]
                memory_gt = (gt_cache[memory_name], memory_name)
                row = pair_record(global_index, domain, name, rank, similarity, query_prob,
                                  bank_np[index], query_gt, memory_gt, diag.get("is_sft", False), len(names))
                pair_batch.append(row)
                pair_rows.append(row)

            pool_geometries = []
            for index in range(min(len(names), len(bank_np))):
                geometry = _geometry(bank_np[index])
                pool_geometries.append({"fg_distance": _distance(_geometry(query_prob), geometry, "fg"),
                                        "geometry": geometry})
        else:
            pool_geometries = []
        query_rows.append(query_record(global_index, domain, name, diag.get("is_sft", False),
                                       len(names), _geometry(query_prob), pool_geometries,
                                       pair_batch, [r["memory_name"] for r in pair_batch]))
        gt_cache[name] = query_gt.copy()
        if (global_index + 1) % 200 == 0:
            print(f"EXP-6.5: {global_index + 1}/{total}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / f"pair_spatial_diagnostics_seed{seed}.csv"
    query_path = output_dir / f"query_spatial_summary_seed{seed}.csv"
    pd.DataFrame(pair_rows).to_csv(pair_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(query_rows).to_csv(query_path, index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {"seed": seed, "num_slices": total, "num_pairs": len(pair_rows),
                "prediction_sha256": digest.hexdigest(), "elapsed_seconds": time.time() - started,
                "device": str(device), "data_root": str(data_root), "checkpoint": str(checkpoint),
                "stream_order": "B -> C -> D", "topk": 5, "gt_used_only_for": "offline diagnosis",
                "one_model_call_per_query": True, "warp_align_corners": False,
                "warp_interpolation": "nearest", "warp_padding": "zeros"}
    (output_dir / f"run_metadata_seed{seed}.json").write_text(json.dumps(json_safe(metadata), indent=2))
    reference = ROOT / "results/exp6/raw/released_seed2026.json"
    if limit is None and seed == 2026 and reference.exists():
        ref_hash = json.loads(reference.read_text())["prediction_sha256"]
        passed = bool(metadata["prediction_sha256"] == ref_hash)
        (output_dir / "released_equivalence.txt").write_text(
            f"EXP-6.5 Released vs EXP-6 Released\nrecord_count={total}\n"
            f"prediction_sha256_equal={metadata['prediction_sha256'] == ref_hash}\nPASS={passed}\n")
        if not passed:
            raise RuntimeError("EXP-6.5 Released equivalence failed")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.output_dir, args.seed, args.limit)


if __name__ == "__main__":
    main()
