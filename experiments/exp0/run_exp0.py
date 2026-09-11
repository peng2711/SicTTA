#!/usr/bin/env python3
"""EXP-0 reproduction runner for the released SicTTA implementation.

The runner reads the author's processed 2-D slices directly.  It deliberately
keeps the official stream order and starts a fresh process/model per method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
NAME_RE = re.compile(r"(.+)_([0-9]+)_z([0-9]+)\.nii\.gz$")
PAPER = {
    "BN(S)": (71.72, 64.37, 72.57, 69.73),
    "BN(T)": (72.30, 69.59, 71.03, 71.24),
    "SFF": (73.90, 71.45, 72.64, 72.92),
    "SABE": (74.89, 71.32, 73.84, 73.62),
    "SABE+SFF": (79.13, 76.15, 77.30, 77.88),
}
MODES = {
    "source": "BN(S)",
    "bnt": "BN(T)",
    "sff": "SFF",
    "sabe": "SABE",
    "full": "SABE+SFF",
}


def digest_stream(root: Path, domains: list[str]) -> str:
    digest = hashlib.sha256()
    for domain in domains:
        frame = pd.read_csv(root / domain / "all.csv")
        for value in frame["image"].astype(str):
            digest.update(f"{domain}\t{Path(value).name}\n".encode())
    return digest.hexdigest()


class ProcessedStream:
    def __init__(self, root: Path, domains: list[str], max_volumes: int | None = None):
        self.root = root
        self.domains = domains
        self.items: list[tuple[str, str, int, str]] = []
        seen: dict[str, set[str]] = defaultdict(set)
        for domain in domains:
            frame = pd.read_csv(root / domain / "all.csv")
            for value in frame["image"].astype(str):
                name = Path(value).name
                match = NAME_RE.fullmatch(name)
                if match is None:
                    raise RuntimeError(f"Unrecognized processed slice name: {name}")
                volume = f"{match.group(1)}_F{match.group(2)}"
                if max_volumes is not None and volume not in seen[domain]:
                    if len(seen[domain]) >= max_volumes:
                        continue
                    seen[domain].add(volume)
                self.items.append((domain, volume, int(match.group(3)), name))

    def __iter__(self):
        for domain, volume, z_index, name in self.items:
            image_path = self.root / domain / "image" / name
            label_path = self.root / domain / "label" / name
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
            label = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).astype(np.int64)
            if image.shape != (1, 320, 320) or label.shape != (1, 320, 320):
                raise RuntimeError(f"Unexpected shape for {domain}/{name}: {image.shape}/{label.shape}")
            yield torch.from_numpy(image).unsqueeze(0), torch.from_numpy(label).long(), domain, volume, z_index, name


def dice(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred, truth = prediction == class_id, target == class_id
    return float((2.0 * np.logical_and(pred, truth).sum() + 1e-5) /
                 (pred.sum() + truth.sum() + 1e-5))


def assd(prediction: np.ndarray, target: np.ndarray, class_id: int) -> float:
    pred = prediction == class_id
    truth = target == class_id
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        return float("nan")
    structure = ndimage.generate_binary_structure(pred.ndim, 1)
    pred_surface = np.logical_and(pred, ~ndimage.binary_erosion(pred, structure))
    truth_surface = np.logical_and(truth, ~ndimage.binary_erosion(truth, structure))
    pred_distance = ndimage.distance_transform_edt(~pred_surface)
    truth_distance = ndimage.distance_transform_edt(~truth_surface)
    return float((truth_distance[pred_surface].mean() + pred_distance[truth_surface].mean()) / 2.0)


def json_safe(value):
    """Convert undefined metric values to JSON null."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_model(mode: str, checkpoint: Path, device: torch.device,
               admission_policy: str = "released"):
    sys.path.insert(0, str(ROOT))
    from robustbench.seg_net.unet import UNet
    from robustbench.tta import setup_sictta
    from robustbench.utils import get_params
    from sotas import norm

    base = UNet(get_params("unet", "mms2d"))
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    base.load_state_dict(state, strict=True)
    base = base.to(device).eval()
    if mode == "source":
        return base
    if mode == "bnt":
        # EXP0_REPRO: Official Norm behavior: only BN layers use
        # EXP0_REPRO: current-image stats; dropout and the rest remain eval-mode.
        return norm.Norm(base).to(device)
    if mode == "sff":
        # EXP0_REPRO: SFF uses source BN and no enhanced batch.
        return setup_sictta(base, use_test_bn=False, use_sabe=False, use_sff=True,
                            admission_policy=admission_policy)
    if mode == "sabe":
        return setup_sictta(base, use_test_bn=True, use_sabe=True, use_sff=False,
                            admission_policy=admission_policy)
    if mode == "full":
        return setup_sictta(base, use_test_bn=True, use_sabe=True, use_sff=True,
                            admission_policy=admission_policy)
    raise ValueError(mode)


def dataset_check(root: Path, output: Path) -> None:
    lines = ["EXP-0 dataset sanity check", f"root={root}", ""]
    for domain in "ABCD":
        rows = pd.read_csv(root / domain / "all.csv")
        patients, volumes = set(), set()
        shapes, classes, ranges, empty = set(), set(), [], 0
        failures = []
        for row in rows.itertuples(index=False):
            name = Path(row.image).name
            match = NAME_RE.fullmatch(name)
            if match is None:
                failures.append(f"bad_name:{name}")
                continue
            patients.add(match.group(1))
            volumes.add(f"{match.group(1)}_F{match.group(2)}")
            try:
                image = sitk.GetArrayFromImage(sitk.ReadImage(str(root / domain / "image" / name)))
                label = sitk.GetArrayFromImage(sitk.ReadImage(str(root / domain / "label" / name)))
                shapes.add((tuple(image.shape), tuple(label.shape)))
                classes.update(np.unique(label).astype(int).tolist())
                ranges.append((float(image.min()), float(image.max())))
                empty += int(not np.any(label > 0))
            except Exception as error:  # EXP0_REPRO: retain diagnostic failures
                failures.append(f"{name}: {type(error).__name__}: {error}")
        lines.extend([
            f"Domain {domain}",
            f"  patients={len(patients)} volumes={len(volumes)} slices={len(rows)}",
            f"  image_label_shapes={sorted(shapes)}",
            f"  label_class_ids={sorted(classes)}",
            f"  intensity_range=[{min(x[0] for x in ranges):.6g}, {max(x[1] for x in ranges):.6g}]",
            f"  empty_mask_slices={empty}",
            f"  read_failures={len(failures)}",
        ])
        if failures:
            lines.extend(f"    {failure}" for failure in failures[:20])
        lines.append("")
    lines.extend([
        "Expected semantic mapping from official loader/paper: background=0, LV=1, MYO=2, RV=3.",
        "The released CSVs contain absolute paths from the author's machine; this runner remaps by domain/split basename.",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def checkpoint_check(checkpoint: Path, output: Path, device: torch.device) -> None:
    sys.path.insert(0, str(ROOT))
    from robustbench.seg_net.unet import UNet
    from robustbench.utils import get_params
    state = torch.load(checkpoint, map_location="cpu")
    model = UNet(get_params("unet", "mms2d"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    with torch.inference_mode():
        result = model(torch.zeros(1, 1, 320, 320, device=device))
    lines = [
        f"checkpoint={checkpoint}", f"state_type={type(state).__name__}",
        f"state_keys={len(state)}", f"missing_keys={missing}",
        f"unexpected_keys={unexpected}", f"output_shape={tuple(result.shape)}",
        f"finite={bool(torch.isfinite(result).all())}",
        f"softmax_max_sum_error={float((result.softmax(1).sum(1)-1).abs().max())}",
        f"predicted_class_ids={torch.unique(result.argmax(1)).tolist()}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def build_environment(root: Path, checkpoint: Path, output: Path, seed: int, mode: str) -> None:
    lines = [
        f"git commit: {os.popen(f'git -C {ROOT} rev-parse HEAD').read().strip()}",
        f"python: {sys.version.replace(chr(10), ' ')}",
        f"torch: {torch.__version__}", f"CUDA version: {torch.version.cuda}",
        f"cuDNN version: {torch.backends.cudnn.version()}",
        f"numpy: {np.__version__}", f"pandas: {pd.__version__}",
        f"SimpleITK: {sitk.Version_VersionString()}",
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"dataset root: {root}", f"checkpoint: {checkpoint}",
        "model architecture: UNet (ft_chns=[16,32,64,128,256], 4 classes)",
        "input resolution: 320x320", "batch size: 1", f"random seed: {seed}",
        f"current mode: {MODES[mode]}", f"stream order: B -> C -> D",
        f"stream sha256: {digest_stream(root, ['B','C','D'])}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def run(mode: str, root: Path, checkpoint: Path, output: Path, seed: int,
        max_volumes: int | None = None) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stream = ProcessedStream(root, ["B", "C", "D"], max_volumes=max_volumes)
    model = load_model(mode, checkpoint, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    volume_masks: dict[tuple[str, str], dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    domain_started: dict[str, float] = {}
    domain_ended: dict[str, float] = {}
    pool_boundary = {}
    started = time.time()
    current_domain = None
    for index, (image, label, domain, volume, z_index, name) in enumerate(stream):
        if domain != current_domain:
            if current_domain is not None:
                domain_ended[current_domain] = time.time()
            current_domain = domain
            domain_started[domain] = time.time()
            pool_boundary[domain] = {"start": len(getattr(getattr(model, "pool", None), "name_list", []))}
        before = len(getattr(getattr(model, "pool", None), "name_list", []))
        with torch.inference_mode():
            if mode in {"sff", "sabe", "full"}:
                output_tensor = model(image.to(device), [name])
            else:
                output_tensor = model(image.to(device))
        after = len(getattr(getattr(model, "pool", None), "name_list", []))
        prediction = output_tensor.argmax(1).cpu().numpy()[0]
        target = label.numpy()[0]
        values = [dice(prediction, target, class_id) for class_id in (1, 2, 3)]
        records.append({
            "domain": domain, "name": name, "volume": volume, "z_index": z_index,
            "dice_lv": values[0], "dice_myo": values[1], "dice_rv": values[2],
            "dice_average": float(np.mean(values)), "memory_size": after,
            "memory_written": bool(after > before),
        })
        volume_masks[(domain, volume)][z_index] = (prediction, target)
    if current_domain is not None:
        domain_ended[current_domain] = time.time()
    for domain in ["B", "C", "D"]:
        pool_boundary.setdefault(domain, {"start": 0})
        pool_boundary[domain]["end"] = max((r["memory_size"] for r in records if r["domain"] == domain), default=pool_boundary[domain]["start"])

    summary = {}
    per_case = []
    for domain in ["B", "C", "D"]:
        subset = [r for r in records if r["domain"] == domain]
        if not subset:
            continue
        domain_cases = [key for key in volume_masks if key[0] == domain]
        case_scores = []
        for _, volume in domain_cases:
            slices = volume_masks[(domain, volume)]
            ordered = [slices[z] for z in sorted(slices)]
            prediction = np.stack([item[0] for item in ordered])
            target = np.stack([item[1] for item in ordered])
            class_dice = [dice(prediction, target, class_id) for class_id in (1, 2, 3)]
            class_assd = [assd(prediction, target, class_id) for class_id in (1, 2, 3)]
            per_case.append({"method": MODES[mode], "domain": domain, "patient_id": volume,
                             "dice_lv": class_dice[0], "dice_myo": class_dice[1],
                             "dice_rv": class_dice[2], "dice_average": float(np.mean(class_dice)),
                             "assd_lv": class_assd[0], "assd_myo": class_assd[1],
                             "assd_rv": class_assd[2], "assd_average": float(np.nanmean(class_assd)),
                             "num_slices": len(ordered)})
            case_scores.append(per_case[-1])
        summary[domain] = {
            "num_volumes": len(case_scores), "num_slices": len(subset),
            "dice_lv": float(np.mean([r["dice_lv"] for r in subset])),
            "dice_myo": float(np.mean([r["dice_myo"] for r in subset])),
            "dice_rv": float(np.mean([r["dice_rv"] for r in subset])),
            "dice_average": float(np.mean([r["dice_average"] for r in subset])),
            "assd_lv": float(np.nanmean([r["assd_lv"] for r in case_scores])),
            "assd_myo": float(np.nanmean([r["assd_myo"] for r in case_scores])),
            "assd_rv": float(np.nanmean([r["assd_rv"] for r in case_scores])),
            "assd_average": float(np.nanmean([r["assd_average"] for r in case_scores])),
            "runtime_seconds": domain_ended[domain] - domain_started[domain],
            "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2
                                    if torch.cuda.is_available() else 0.0),
            "memory_writes": int(sum(r["memory_written"] for r in subset)),
            "pool_start": pool_boundary[domain]["start"], "pool_end": pool_boundary[domain]["end"],
        }
    average = {}
    for key in ("dice_lv", "dice_myo", "dice_rv", "dice_average", "assd_lv", "assd_myo", "assd_rv", "assd_average"):
        average[key] = float(np.nanmean([summary[d][key] for d in ["B", "C", "D"]]))
    average.update({
        "num_volumes": sum(summary[d]["num_volumes"] for d in ["B", "C", "D"]),
        "num_slices": sum(summary[d]["num_slices"] for d in ["B", "C", "D"]),
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_mb": max(summary[d]["peak_gpu_memory_mb"] for d in ["B", "C", "D"]),
        "memory_writes": sum(summary[d]["memory_writes"] for d in ["B", "C", "D"]),
    })
    payload = {"method": MODES[mode], "mode": mode, "seed": seed,
               "data_root": str(root), "checkpoint": str(checkpoint),
               "domains": ["B", "C", "D"], "batch_size": 1,
               "summary": summary, "average": average, "pool_boundary": pool_boundary,
               "records": records, "per_case": per_case,
               "elapsed_seconds": time.time() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-volumes", type=int)
    parser.add_argument("--dataset-check", type=Path)
    parser.add_argument("--checkpoint-check", type=Path)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()
    if args.dataset_check:
        dataset_check(args.data_root, args.dataset_check)
    if args.checkpoint_check:
        checkpoint_check(args.checkpoint, args.checkpoint_check,
                         torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if args.environment:
        build_environment(args.data_root, args.checkpoint, args.environment, args.seed, args.mode)
    payload = run(args.mode, args.data_root, args.checkpoint, args.output, args.seed, args.max_volumes)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
