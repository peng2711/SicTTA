#!/usr/bin/env python3
"""EXP-7: read-only local correspondence diagnosis.

Feature matching is performed after the existing SicTTA forward returns.  It
never changes the model output, memory, retrieval, or adaptation state.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/exp0"))
from run_exp0 import ProcessedStream, json_safe, load_model


DOMAINS = ["B", "C", "D"]
CLASSES = [(1, "lv"), (2, "myo"), (3, "rv")]
RADII = [1, 2, 3]
EPS = 1e-8


def resize_label(label, size):
    tensor = torch.from_numpy(np.asarray(label, dtype=np.float32)).view(1, 1, *label.shape)
    result = F.interpolate(tensor, size=size, mode="nearest")[0, 0].long().numpy()
    unique = set(np.unique(result).tolist())
    if not unique.issubset({0, 1, 2, 3}):
        raise RuntimeError(f"invalid resized class ids: {unique}")
    return result


def boundary_mask(labels):
    foreground = labels > 0
    boundary = np.zeros_like(foreground, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.full_like(labels, -1)
            y0, y1 = max(0, dy), min(labels.shape[0], labels.shape[0] + dy)
            x0, x1 = max(0, dx), min(labels.shape[1], labels.shape[1] + dx)
            shifted[y0:y1, x0:x1] = labels[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            boundary |= foreground & (shifted != labels)
    return boundary


def _metric(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()) if len(values) else np.nan


def _location_matches(query_feature, memory_feature, memory_labels, radius):
    """Return same/local/global matches for every feature-grid location."""
    query = F.normalize(query_feature.float(), dim=0)
    memory = F.normalize(memory_feature.float(), dim=0)
    height, width = query.shape[-2:]
    similarity = torch.einsum("cyx,cij->yxij", query, memory).cpu().numpy()
    rows = []
    for y in range(height):
        for x in range(width):
            local_y0, local_y1 = max(0, y - radius), min(height, y + radius + 1)
            local_x0, local_x1 = max(0, x - radius), min(width, x + radius + 1)
            local = similarity[y, x, local_y0:local_y1, local_x0:local_x1]
            local_index = int(np.argmax(local))
            local_y, local_x = np.unravel_index(local_index, local.shape)
            local_y += local_y0
            local_x += local_x0
            global_index = int(np.argmax(similarity[y, x]))
            global_y, global_x = np.unravel_index(global_index, (height, width))
            rows.append({
                "y": y, "x": x, "same_similarity": float(similarity[y, x, y, x]),
                "local_similarity": float(similarity[y, x, local_y, local_x]),
                "local_x": int(local_x), "local_y": int(local_y),
                "global_similarity": float(similarity[y, x, global_y, global_x]),
                "global_x": int(global_x), "global_y": int(global_y),
                "memory_same_label": int(memory_labels[y, x]),
                "memory_local_label": int(memory_labels[local_y, local_x]),
                "memory_global_label": int(memory_labels[global_y, global_x]),
            })
    return rows


def _agreement(rows, query_labels, boundary, class_id=None, radius=None):
    positions = query_labels > 0 if class_id is None else query_labels == class_id
    selected = [r for r in rows if positions[r["y"], r["x"]]]
    if not selected:
        return {"same": np.nan, "local": np.nan, "global": np.nan, "oracle": np.nan,
                "corrected_count": 0, "harmed_count": 0,
                "corrected_delta_similarity": np.nan, "harmed_delta_similarity": np.nan,
                "corrected_displacement": np.nan, "harmed_displacement": np.nan,
                "same_similarity": np.nan, "local_similarity": np.nan, "delta_similarity": np.nan,
                "mean_displacement": np.nan, "median_displacement": np.nan,
                "zero_displacement_fraction": np.nan, "distance_le1_fraction": np.nan,
                "distance_gt1_fraction": np.nan, "boundary_same": np.nan, "boundary_local": np.nan,
                "interior_same": np.nan, "interior_local": np.nan}
    local_correct = [r["memory_local_label"] == int(query_labels[r["y"], r["x"]]) for r in selected]
    same_correct = [r["memory_same_label"] == int(query_labels[r["y"], r["x"]]) for r in selected]
    global_correct = [r["memory_global_label"] == int(query_labels[r["y"], r["x"]]) for r in selected]
    oracle = []
    for r in selected:
        y, x = r["y"], r["x"]
        y0, y1 = max(0, y - radius), min(query_labels.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(query_labels.shape[1], x + radius + 1)
        # memory labels have the same feature resolution, so the query window
        # and memory window share the exact integer coordinate support.
        oracle.append(bool(np.any(r["memory_window"][y0:y1, x0:x1] == int(query_labels[y, x]))) if "memory_window" in r else False)
    dx = np.asarray([r["local_x"] - r["x"] for r in selected], dtype=float)
    dy = np.asarray([r["local_y"] - r["y"] for r in selected], dtype=float)
    distance = np.hypot(dx, dy)
    boundary_rows = [r for r in selected if boundary[r["y"], r["x"]]]
    interior_rows = [r for r in selected if not boundary[r["y"], r["x"]]]

    def row_agreement(items, field):
        return _metric([r[field] == int(query_labels[r["y"], r["x"]]) for r in items]) if items else np.nan

    return {"same": _metric(same_correct), "local": _metric(local_correct), "global": _metric(global_correct),
            "corrected_count": int(sum(not s and l for s, l in zip(same_correct, local_correct))),
            "harmed_count": int(sum(s and not l for s, l in zip(same_correct, local_correct))),
            "corrected_delta_similarity": _metric([r["local_similarity"] - r["same_similarity"] for r, s, l in zip(selected, same_correct, local_correct) if not s and l]),
            "harmed_delta_similarity": _metric([r["local_similarity"] - r["same_similarity"] for r, s, l in zip(selected, same_correct, local_correct) if s and not l]),
            "corrected_displacement": _metric([np.hypot(r["local_x"] - r["x"], r["local_y"] - r["y"]) for r, s, l in zip(selected, same_correct, local_correct) if not s and l]),
            "harmed_displacement": _metric([np.hypot(r["local_x"] - r["x"], r["local_y"] - r["y"]) for r, s, l in zip(selected, same_correct, local_correct) if s and not l]),
            "oracle": _metric(oracle), "same_similarity": _metric([r["same_similarity"] for r in selected]),
            "local_similarity": _metric([r["local_similarity"] for r in selected]),
            "delta_similarity": _metric([r["local_similarity"] - r["same_similarity"] for r in selected]),
            "mean_displacement": _metric(distance), "median_displacement": float(np.median(distance)),
            "zero_displacement_fraction": _metric(distance == 0),
            "distance_le1_fraction": _metric(distance <= 1), "distance_gt1_fraction": _metric(distance > 1),
            "boundary_same": row_agreement(boundary_rows, "memory_same_label"),
            "boundary_local": row_agreement(boundary_rows, "memory_local_label"),
            "interior_same": row_agreement(interior_rows, "memory_same_label"),
            "interior_local": row_agreement(interior_rows, "memory_local_label")}


def pair_metrics(query_feature, memory_feature, query_labels, memory_labels, boundary):
    query_fg = query_labels > 0
    base = []
    radius_rows = {}
    for radius in [0, *RADII]:
        rows = _location_matches(query_feature, memory_feature, memory_labels, radius)
        for row in rows:
            # The full memory label map is attached only to diagnostic rows.
            row["memory_window"] = memory_labels
        radius_rows[radius] = rows
    for radius in RADII:
        metrics = {key: _agreement(radius_rows[radius], query_labels, boundary, None, radius)
                   for key in ["fg"]}
        for _, key in CLASSES:
            metrics[key] = _agreement(radius_rows[radius], query_labels, boundary,
                                      {"lv": 1, "myo": 2, "rv": 3}[key], radius)
        base.append((radius, metrics))

    same_rows = radius_rows[0]
    same_fg = _agreement(same_rows, query_labels, boundary, None, 0)
    global_metrics = same_fg
    primary = dict([item for item in base if item[0] == 2][0][1]["fg"])
    result = {"num_query_fg_locations": int(query_fg.sum()),
              "same_agree_fg": same_fg["same"], "global_nn_agree_fg": global_metrics["global"],
              "same_similarity_fg": same_fg["same_similarity"],
              "local_r2_similarity_fg": primary["local_similarity"],
              "delta_similarity_r2_fg": primary["delta_similarity"],
              "mean_displacement_r2": primary["mean_displacement"],
              "median_displacement_r2": primary["median_displacement"],
              "zero_displacement_fraction_r2": primary["zero_displacement_fraction"],
              "distance_le1_fraction_r2": primary["distance_le1_fraction"],
              "distance_gt1_fraction_r2": primary["distance_gt1_fraction"],
              "corrected_count_r2_fg": primary["corrected_count"], "harmed_count_r2_fg": primary["harmed_count"],
              "corrected_delta_similarity_r2_fg": primary["corrected_delta_similarity"],
              "harmed_delta_similarity_r2_fg": primary["harmed_delta_similarity"],
              "corrected_displacement_r2_fg": primary["corrected_displacement"],
              "harmed_displacement_r2_fg": primary["harmed_displacement"],
              "boundary_same_agree": primary["boundary_same"], "boundary_local_agree": primary["boundary_local"],
              "interior_same_agree": primary["interior_same"], "interior_local_agree": primary["interior_local"]}
    for radius, metrics in base:
        result[f"local_r{radius}_agree_fg"] = metrics["fg"]["local"]
        result[f"oracle_local_r{radius}_agree_fg"] = metrics["fg"]["oracle"]
        result[f"local_r{radius}_delta_fg"] = metrics["fg"]["local"] - metrics["fg"]["same"]
        result[f"available_gain_r{radius}_fg"] = metrics["fg"]["oracle"] - metrics["fg"]["same"]
        result[f"recovery_ratio_r{radius}_fg"] = result[f"local_r{radius}_delta_fg"] / (result[f"available_gain_r{radius}_fg"] + EPS)
        for _, key in CLASSES:
            m = metrics[key]
            result[f"same_agree_{key}"] = m["same"]
            result[f"global_nn_agree_{key}"] = m["global"]
            result[f"local_r{radius}_agree_{key}"] = m["local"]
            result[f"local_r{radius}_delta_{key}"] = m["local"] - m["same"]
            result[f"oracle_local_r{radius}_agree_{key}"] = m["oracle"]
            result[f"available_gain_r{radius}_{key}"] = m["oracle"] - m["same"]
            result[f"recovery_ratio_r{radius}_{key}"] = result[f"local_r{radius}_delta_{key}"] / (result[f"available_gain_r{radius}_{key}"] + EPS)
            if radius == 2:
                result[f"mean_displacement_r2_{key}"] = m["mean_displacement"]
                result[f"median_displacement_r2_{key}"] = m["median_displacement"]
                result[f"zero_displacement_fraction_r2_{key}"] = m["zero_displacement_fraction"]
                result[f"corrected_count_r2_{key}"] = m["corrected_count"]
                result[f"harmed_count_r2_{key}"] = m["harmed_count"]
                result[f"corrected_delta_similarity_r2_{key}"] = m["corrected_delta_similarity"]
                result[f"harmed_delta_similarity_r2_{key}"] = m["harmed_delta_similarity"]
                result[f"corrected_displacement_r2_{key}"] = m["corrected_displacement"]
                result[f"harmed_displacement_r2_{key}"] = m["harmed_displacement"]
    # Same-position is exactly radius zero; this is retained for a formal sanity check.
    result["radius0_same_agree_fg"] = same_fg["same"]
    result["radius0_same_similarity_fg"] = same_fg["same_similarity"]
    return result


def make_pair_row(global_index, domain, query_name, memory_name, rank, similarity,
                  query_feature, memory_feature, query_labels, memory_labels, is_sft):
    boundary = boundary_mask(query_labels)
    values = pair_metrics(query_feature, memory_feature, query_labels, memory_labels, boundary)
    row = {"global_index": global_index, "domain": domain, "query_name": query_name,
           "memory_name": memory_name, "memory_rank": rank, "global_similarity": float(similarity),
           "is_sft": bool(is_sft)}
    row.update(values)
    return row


def query_row(global_index, domain, query_name, is_sft, pair_rows):
    row = {"global_index": global_index, "domain": domain, "query_name": query_name,
           "is_sft": bool(is_sft), "num_memories": len(pair_rows)}
    fields = ["same_agree_fg", "local_r1_agree_fg", "local_r2_agree_fg", "local_r3_agree_fg",
              "global_nn_agree_fg", "oracle_local_r2_agree_fg", "mean_displacement_r2",
              "zero_displacement_fraction_r2", "boundary_same_agree", "boundary_local_agree",
              "interior_same_agree", "interior_local_agree"]
    for _, key in CLASSES:
        fields += [f"same_agree_{key}", f"local_r2_agree_{key}", f"local_r2_delta_{key}"]
    for field in fields:
        row[field] = _metric([r[field] for r in pair_rows if np.isfinite(r[field])])
    row["delta_r2_fg"] = row["local_r2_agree_fg"] - row["same_agree_fg"]
    row["oracle_local_r2_agree_fg"] = _metric([r["oracle_local_r2_agree_fg"] for r in pair_rows if np.isfinite(r["oracle_local_r2_agree_fg"])])
    row["global_nn_agreement"] = row["global_nn_agree_fg"]
    row["boundary_delta"] = row["boundary_local_agree"] - row["boundary_same_agree"]
    row["interior_delta"] = row["interior_local_agree"] - row["interior_same_agree"]
    row["mean_delta_similarity_r2_fg"] = _metric([r["delta_similarity_r2_fg"] for r in pair_rows if np.isfinite(r["delta_similarity_r2_fg"])])
    return row


def run(data_root, checkpoint, output_dir, seed=2026, limit=None):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(data_root, DOMAINS)
    model = load_model("full", checkpoint, device, admission_policy="released", fusion_mode="released")
    gt_cache = {}
    pair_rows, query_rows = [], []
    digest = hashlib.sha256(); started = time.time()
    total = min(limit, len(stream.items)) if limit is not None else len(stream.items)
    for global_index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if global_index >= total: break
        with torch.inference_mode():
            output = model(image.to(device), [name])
        prediction = output.argmax(1).cpu().numpy()[0]
        digest.update(f"{domain}\t{name}\n".encode()); digest.update(np.asarray(prediction, dtype=np.int64).tobytes())
        query_labels = resize_label(label.numpy()[0], model.last_localcorr_query_feature.shape[-2:])
        qfeature = model.last_localcorr_query_feature.detach().float().cpu()
        mfeatures = model.last_localcorr_retrieved_features
        names = list(model.last_localcorr_retrieved_names)
        similarities = list(model.last_localcorr_retrieved_similarities)
        pair_batch = []
        if mfeatures is not None:
            mfeatures = mfeatures.detach().float().cpu()
            for rank, (memory_name, similarity) in enumerate(zip(names, similarities), start=1):
                if memory_name not in gt_cache: continue
                memory_labels = resize_label(gt_cache[memory_name], qfeature.shape[-2:])
                row = make_pair_row(global_index, domain, name, memory_name, rank, similarity,
                                    qfeature, mfeatures[rank - 1], query_labels, memory_labels,
                                    model.last_diag.get("is_sft", False))
                pair_batch.append(row); pair_rows.append(row)
        query_rows.append(query_row(global_index, domain, name, model.last_diag.get("is_sft", False), pair_batch))
        gt_cache[name] = label.numpy()[0].copy()
        if (global_index + 1) % 200 == 0: print(f"EXP-7: {global_index + 1}/{total}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(output_dir / f"pair_local_correspondence_seed{seed}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(query_rows).to_csv(output_dir / f"query_local_correspondence_seed{seed}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {"seed": seed, "num_slices": total, "num_pairs": len(pair_rows),
                "prediction_sha256": digest.hexdigest(), "device": str(device),
                "one_model_call_per_query": True, "feature_source": "pre-SABE/pre-SFF retrieval tensors",
                "primary_radius": 2, "sensitivity_radii": [1, 3], "gt_used_only_for": "offline correspondence evaluation"}
    (output_dir / f"run_metadata_seed{seed}.json").write_text(json.dumps(json_safe(metadata), indent=2))
    reference = ROOT / "results/exp6_5/formal/run_metadata_seed2026.json"
    if limit is None and seed == 2026 and reference.exists():
        reference_hash = json.loads(reference.read_text())["prediction_sha256"]
        passed = metadata["prediction_sha256"] == reference_hash
        (output_dir / "released_equivalence.txt").write_text(
            f"EXP-7 Released vs EXP-6.5 Released\nrecord_count={total}\n"
            f"prediction_sha256_equal={passed}\nPASS={passed}\n")
        if not passed: raise RuntimeError("EXP-7 Released equivalence failed")
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
