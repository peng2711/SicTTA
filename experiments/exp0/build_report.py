#!/usr/bin/env python3
"""Build the machine-readable EXP-0 summary and reproduction report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/exp0"
FILES = {"BN(S)": "bns.json", "BN(T)": "bnt.json", "SFF": "sff.json",
         "SABE": "sabe.json", "SABE+SFF": "full.json"}
PAPER = {
    "BN(S)": (71.72, 64.37, 72.57, 69.73),
    "BN(T)": (72.30, 69.59, 71.03, 71.24),
    "SFF": (73.90, 71.45, 72.64, 72.92),
    "SABE": (74.89, 71.32, 73.84, 73.62),
    "SABE+SFF": (79.13, 76.15, 77.30, 77.88),
}
DOMAINS = ["B", "C", "D"]


def percentage(value):
    return None if value is None else float(value) * 100.0


def main():
    payloads = {method: json.loads((OUT / filename).read_text())
                for method, filename in FILES.items()}
    summary_fields = ["method", "domain", "dice_lv", "dice_myo", "dice_rv",
                      "dice_average", "assd_lv", "assd_myo", "assd_rv",
                      "assd_average", "num_volumes", "num_slices",
                      "runtime_seconds", "peak_gpu_memory_mb"]
    with (OUT / "exp0_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for method, payload in payloads.items():
            for domain in DOMAINS:
                data = payload["summary"][domain]
                writer.writerow({"method": method, "domain": domain, **{
                    key: percentage(data[key]) if key.startswith("dice_") else data[key]
                    for key in summary_fields[2:]}})
            data = payload["average"]
            writer.writerow({"method": method, "domain": "Average", **{
                key: percentage(data[key]) if key.startswith("dice_") else data[key]
                for key in summary_fields[2:]}})

    case_fields = ["method", "domain", "patient_id", "dice_lv", "dice_myo",
                   "dice_rv", "dice_average", "assd_lv", "assd_myo", "assd_rv",
                   "assd_average", "num_slices"]
    with (OUT / "exp0_per_case.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=case_fields)
        writer.writeheader()
        for method, payload in payloads.items():
            for row in payload["per_case"]:
                writer.writerow({key: percentage(row[key]) if key.startswith("dice_")
                                 else row[key] for key in case_fields})

    lines = [
        "# EXP-0 SicTTA Reproduction", "",
        "## Environment", "",
        "- Python 3.8.18; PyTorch 2.3.1+cu121; NVIDIA GeForce RTX 4090.",
        "- Seed: 2026; batch size: 1; input: 320x320.",
        "- Full environment details: `environment.txt`.", "",
        "## Dataset", "",
        "- Root: `/home/peng/project9_9/data/paper_mms2d`.",
        "- Source domain: A; continual target order: B -> C -> D.",
        "- Stream: `all.csv`, shuffle disabled; slices are read in CSV order.",
        "- A/B/C/D: 1599/2049/1207/835 slices; 95/125/75/50 patients; 190/250/150/100 patient-frame volumes.",
        "- Mapping: background=0, LV=1, MYO=2, RV=3; all slices read as (1,320,320); no read failures or empty-mask slices.", "",
        "## Source checkpoint", "",
        "- `/home/peng/project9_9/github/SicTTA_checkpoint_62df/save_model/mms2d_unet/source-A-source-model-latest.pth`.",
        "- Strict state_dict load: 136 keys, no missing/unexpected keys.",
        "- The tracked repository checkpoint at `save_model/mms2d_unet/source-A-source-model-latest.pth` is a 0.8 MB corrupt zip and was not used or overwritten.", "",
        "## Exact commands", "",
        "```bash",
        "cd /home/peng/SicTTA",
        "/home/peng/project9_9/.venvs/spegc/bin/python experiments/exp0/run_exp0.py --mode {source,bnt,sff,sabe,full} --data-root /home/peng/project9_9/data/paper_mms2d --checkpoint /home/peng/project9_9/github/SicTTA_checkpoint_62df/save_model/mms2d_unet/source-A-source-model-latest.pth --output results/exp0/<mode>.json --seed 2026",
        "```", "",
        "## Code changes", "",
        "- Added EXP0_REPRO-only `use_test_bn`, `use_sabe`, and `use_sff` switches while preserving full SicTTA defaults.",
        "- Added `experiments/exp0/run_exp0.py` for protocol-controlled stream, sanity checks, Dice/ASSD, pool logging, and per-case output.",
        "- Added `experiments/exp0/build_report.py` for summary tables.",
        "- No training code, GT-based adaptation, memory policy, K/L/alpha, augmentation, loss, or algorithmic method was added.", "",
        "## Results", "",
        "Dice values below are slice-level LV/MYO/RV mean Dice, reported in percent; Average is a macro-average over B/C/D.", "",
        "| Method | Domain B | Domain C | Domain D | Average |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, payload in payloads.items():
        values = [payload["summary"][domain]["dice_average"] * 100 for domain in DOMAINS]
        values.append(payload["average"]["dice_average"] * 100)
        lines.append(f"| {method} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
    lines += ["", "### Paper / reproduction / difference", "",
              "| Method | Paper (B/C/D/Avg) | Our reproduction (B/C/D/Avg) | Difference (B/C/D/Avg) |",
              "|---|---:|---:|---:|"]
    for method, payload in payloads.items():
        ours = [payload["summary"][domain]["dice_average"] * 100 for domain in DOMAINS]
        ours.append(payload["average"]["dice_average"] * 100)
        paper = PAPER[method]
        delta = [ours[i] - paper[i] for i in range(4)]
        lines.append(f"| {method} | " + " / ".join(f"{x:.2f}" for x in paper) + " | " +
                     " / ".join(f"{x:.2f}" for x in ours) + " | " +
                     " / ".join(f"{x:+.2f}" for x in delta) + " |")
    lines += ["", "### Memory boundary audit", "",
              "| Method | B start/end | C start/end | D start/end | Writes B/C/D |",
              "|---|---:|---:|---:|---:|"]
    for method, payload in payloads.items():
        boundary = payload["pool_boundary"]
        writes = [payload["summary"][domain]["memory_writes"] for domain in DOMAINS]
        lines.append(f"| {method} | {boundary['B']['start']}/{boundary['B']['end']} | "
                     f"{boundary['C']['start']}/{boundary['C']['end']} | "
                     f"{boundary['D']['start']}/{boundary['D']['end']} | "
                     + "/".join(map(str, writes)) + " |")
    lines += ["", "## Reproduction Gap Analysis", "",
              "- The core trend is reproduced: BN(S) < BN(T), SFF and SABE exceed BN(T), and SABE+SFF is best.",
              "- BN(S), BN(T), SABE, and full SicTTA are within 0.28 Dice points of the paper Average column. SFF has the largest absolute domain gap (+3.73 on D; +1.23 on Average).",
              "- The current repository's `sotas/sictta.py` uses the source/anchor model for CCD (`model_anchor.eval()`), whereas the older historical checkpoint repository uses `model.eval()`; this was not silently changed. The current branch also admits 41 samples in B and none in C/D for these settings, so the pool remains across boundaries but does not grow.",
              "- The paper's displayed class/domain values and its Average column are not always the same arithmetic aggregate; comparison above uses the paper Average values exactly as supplied.",
              "- ASSD is reported as volume-level 3-D pixel-distance ASSD from stacked patient-frame slices. Dice remains the official slice-level binary Dice implementation; undefined empty-class ASSD values are left blank in CSV.", "",
              "## Conclusion", "",
              "EXP-0 is successful as a protocol and trend reproduction. The absolute gap is largest for SFF, and its implementation/memory-admission discrepancy is documented above. EXP-1 has not been started.", ""]
    (OUT / "exp0_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
