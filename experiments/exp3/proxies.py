"""Pure test-time class reliability proxy functions for EXP-3."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion


EPS = 1e-8
CLASS_IDS = (1, 2, 3)


def _numpy(value):
    return value.detach().float().cpu().numpy()


def _prototype_agreement(query, memory, indices, class_offset):
    if memory is None or memory.shape[0] == 0:
        return np.nan, np.nan, np.nan
    similarities = F.cosine_similarity(
        query[class_offset].unsqueeze(0), memory[:, class_offset, :], dim=1)
    top_count = min(5, memory.shape[0])
    top_indices = torch.argsort(similarities, descending=True)[:top_count]
    top_values = similarities[top_indices].detach().cpu().numpy()
    if indices is None:
        return float(top_values[0]), float(top_values.mean()), []
    valid = [index for index in indices if 0 <= index < memory.shape[0]]
    global_values = similarities[torch.as_tensor(valid, device=similarities.device)] if valid else None
    global_agreement = float(global_values.mean().detach().cpu().item()) if global_values is not None else np.nan
    return float(top_values[0]), float(top_values.mean()), global_agreement


def compute_class_proxies(anchor_probability, feature_map, query_prototypes, memory, global_indices):
    """Compute all proxies from anchor probability and current Released memory only."""
    probability = anchor_probability[0].detach().float()
    feature = feature_map[0].detach().float()
    class_prediction = probability.argmax(dim=0)
    entropy_map = -(probability * torch.log(probability + EPS)).sum(dim=0)
    feature_probability = F.interpolate(
        probability.unsqueeze(0), size=feature.shape[-2:], mode="bilinear", align_corners=False
    )[0]
    feature_flat = feature.reshape(feature.shape[0], -1).transpose(0, 1)
    prediction_np = _numpy(class_prediction)
    entropy_np = _numpy(entropy_map)
    results = {}
    for class_id in CLASS_IDS:
        suffix = {1: "lv", 2: "myo", 3: "rv"}[class_id]
        class_probability = probability[class_id]
        soft_mass = float(class_probability.sum().item())
        entropy = float((class_probability * entropy_map).sum().item() / (soft_mass + EPS))
        confidence = float((class_probability.square()).sum().item() / (soft_mass + EPS))
        other_probability = torch.cat((probability[:class_id], probability[class_id + 1:]), dim=0)
        class_margin_map = class_probability - other_probability.max(dim=0).values
        margin = float((class_probability * class_margin_map).sum().item() / (soft_mass + EPS))

        feature_weights = feature_probability[class_id].reshape(-1)
        feature_mass = feature_weights.sum()
        prototype = (feature_weights.unsqueeze(1) * feature_flat).sum(dim=0) / (feature_mass + EPS)
        distances = (feature_flat - prototype.unsqueeze(0)).square().sum(dim=1)
        compactness = float((feature_weights * distances).sum().item() / (feature_mass.item() + EPS))

        predicted_mask = prediction_np == class_id
        boundary_entropy = np.nan
        boundary_margin = np.nan
        interior_entropy = np.nan
        contrast = np.nan
        if predicted_mask.any():
            structure = np.ones((3, 3), dtype=bool)
            eroded = binary_erosion(predicted_mask, structure=structure, border_value=0)
            boundary = np.logical_xor(predicted_mask, eroded)
            band = binary_dilation(boundary, structure=structure, iterations=2)
            if band.any():
                boundary_entropy = float(entropy_np[band].mean())
                margin_np = _numpy(class_margin_map)
                boundary_margin = float(margin_np[band].mean())
            if eroded.any():
                interior_entropy = float(entropy_np[eroded].mean())
            if np.isfinite(boundary_entropy) and np.isfinite(interior_entropy):
                contrast = boundary_entropy - interior_entropy

        proto_top1, proto_top5, global_agreement = _prototype_agreement(
            query_prototypes, memory, global_indices, class_id - 1
        )
        predicted_area = float(predicted_mask.mean())
        soft_area = soft_mass / float(probability.shape[-2] * probability.shape[-1])
        results[suffix] = {
            "soft_mass": soft_mass,
            "class_entropy": entropy,
            "class_entropy_reliability": -entropy,
            "class_confidence": confidence,
            "class_margin": margin,
            "feature_compactness": compactness,
            "feature_compactness_reliability": -compactness,
            "proto_top1": proto_top1,
            "proto_top5": proto_top5,
            "global_proto_agreement": global_agreement,
            "boundary_entropy": boundary_entropy,
            "boundary_entropy_reliability": -boundary_entropy if np.isfinite(boundary_entropy) else np.nan,
            "boundary_margin": boundary_margin,
            "interior_entropy": interior_entropy,
            "boundary_interior_contrast": contrast,
            "pred_area": predicted_area,
            "soft_area": soft_area,
        }
    return results
