"""Downstream readouts and three discrepancy blocks for Figure 6."""
from __future__ import annotations
from typing import Any
import numpy as np

def weighted_category_proportions(evidence: Any, codes: np.ndarray) -> np.ndarray:
    """Return the four global bars on the fixed empirical FDR support."""

    labels = np.atleast_2d(np.asarray(codes, dtype=np.int8))
    selected = np.asarray(evidence.mask, dtype=bool).ravel()
    raw_weights = np.asarray(evidence.raw_weights, dtype=float).ravel()
    if labels.shape[1] != selected.size or raw_weights.shape != selected.shape:
        raise ValueError("global labels, mask and raw weights do not align")
    active = labels[:, selected]
    weights = raw_weights[selected]
    counts = np.stack(
        [
            np.sum((active == category) * weights[None, :], axis=1)
            for category in range(4)
        ],
        axis=1,
    )
    totals = np.sum(counts, axis=1)
    if np.any(totals <= 0.0):
        raise ValueError("global category bars require positive total raw weight")
    return counts / totals[:, None]


def figure5_visual_rfi(evidence: Any, codes: np.ndarray) -> np.ndarray:
    """Return all six displayed visual RFI bars with full-bin denominators."""

    labels = np.atleast_2d(np.asarray(codes, dtype=np.int8))
    detected = np.asarray(evidence.mask, dtype=bool).ravel()
    raw_weights = np.asarray(evidence.raw_weights, dtype=float).ravel()
    pairs = tuple(evidence.figure5_pairs)
    denominators = np.asarray(evidence.figure5_denominators, dtype=float).ravel()
    if (
        labels.shape[1] != detected.size
        or raw_weights.shape != detected.shape
        or len(pairs) != 6
        or denominators.shape != (6,)
        or np.any(denominators <= 0.0)
    ):
        raise ValueError("Figure-5 visual inputs do not satisfy the six-bin contract")
    result = np.empty((labels.shape[0], 6), dtype=float)
    for index, (left, right) in enumerate(pairs):
        selected = detected & (
            np.asarray(left, dtype=bool).ravel() | np.asarray(right, dtype=bool).ravel()
        )
        active = labels[:, selected]
        weights = raw_weights[selected]
        numerator = np.sum((active == 0) * weights[None, :], axis=1)
        numerator -= np.sum((active == 1) * weights[None, :], axis=1)
        result[:, index] = numerator / denominators[index]
    if np.any(~np.isfinite(result)) or np.any(np.abs(result) > 1.0 + 1e-12):
        raise ValueError("visual RFI bars must be finite in [-1, 1]")
    return result


def semantic_status_rfi(
    evidence: Any, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the two displayed status means using original full-ROI denominators."""

    labels = np.atleast_2d(np.asarray(codes, dtype=np.int8)).copy()
    detected = np.asarray(evidence.mask, dtype=bool).ravel()
    if labels.shape[1] != detected.size:
        raise ValueError("semantic labels and the fixed FDR mask do not align")
    labels[:, ~detected] = -1
    reported = np.asarray(evidence.semantic_reported, dtype=bool).ravel()
    masks = tuple(
        np.asarray(mask, dtype=bool).ravel() for mask in evidence.semantic_full_masks
    )
    if len(masks) != reported.size or not np.any(reported) or np.all(reported):
        raise ValueError("semantic status requires reported and unreported ROIs")
    roi_rfi = np.empty((labels.shape[0], len(masks)), dtype=float)
    for index, selected in enumerate(masks):
        denominator = float(np.sum(selected))
        if denominator <= 0.0:
            raise ValueError("semantic full-ROI denominator must be positive")
        active = labels[:, selected]
        roi_rfi[:, index] = (
            np.sum(active == 0, axis=1) - np.sum(active == 1, axis=1)
        ) / denominator
    return (
        np.mean(roi_rfi[:, ~reported], axis=1),
        np.mean(roi_rfi[:, reported], axis=1),
    )


def semantic_block_error(
    semantic_unreported_rfi: np.ndarray,
    semantic_reported_rfi: np.ndarray,
    *,
    target_semantic_unreported_rfi: float,
    target_semantic_reported_rfi: float,
) -> np.ndarray:
    """Return normalized Euclidean distance across semantic-status RFI values."""

    model_u = np.asarray(semantic_unreported_rfi, dtype=float).ravel()
    model_r = np.asarray(semantic_reported_rfi, dtype=float).ravel()
    target_u = float(target_semantic_unreported_rfi)
    target_r = float(target_semantic_reported_rfi)
    if (
        model_u.shape != model_r.shape
        or np.any(~np.isfinite(model_u))
        or np.any(~np.isfinite(model_r))
        or not np.isfinite(target_u)
        or not np.isfinite(target_r)
    ):
        raise ValueError("semantic-status RFI values must be finite and aligned")
    result = np.sqrt(
        (np.square(0.5 * (model_u - target_u)) + np.square(0.5 * (model_r - target_r)))
        / 2.0
    )
    if np.any(result < 0.0) or np.any(result > 1.0 + 1e-12):
        raise ValueError("semantic block error must be finite in [0, 1]")
    return result


def three_block_errors(
    global_proportions: np.ndarray,
    visual_rfi: np.ndarray,
    semantic_unreported_rfi: np.ndarray,
    semantic_reported_rfi: np.ndarray,
    *,
    target_global_proportions: np.ndarray,
    target_visual_rfi: np.ndarray,
    target_semantic_unreported_rfi: float,
    target_semantic_reported_rfi: float,
) -> np.ndarray:
    """Compress the 4+6+2 Panel-B gaps into three equally weighted blocks."""

    global_values = np.asarray(global_proportions, dtype=float)
    visual_values = np.asarray(visual_rfi, dtype=float)
    model_u = np.asarray(semantic_unreported_rfi, dtype=float).ravel()
    model_r = np.asarray(semantic_reported_rfi, dtype=float).ravel()
    target_global = np.asarray(target_global_proportions, dtype=float).ravel()
    target_visual = np.asarray(target_visual_rfi, dtype=float).ravel()
    count = len(global_values)
    if (
        global_values.shape != (count, 4)
        or visual_values.shape != (count, 6)
        or model_u.shape != (count,)
        or model_r.shape != (count,)
        or target_global.shape != (4,)
        or target_visual.shape != (6,)
    ):
        raise ValueError("the twelve Panel-B bars have incompatible shapes")
    semantic = semantic_block_error(
        model_u,
        model_r,
        target_semantic_unreported_rfi=target_semantic_unreported_rfi,
        target_semantic_reported_rfi=target_semantic_reported_rfi,
    )
    result = np.column_stack(
        (
            np.sqrt(
                np.mean(
                    np.square(global_values - target_global[None, :]),
                    axis=1,
                )
            )
            / np.sqrt(0.5),
            np.sqrt(
                np.mean(
                    np.square(0.5 * np.abs(visual_values - target_visual[None, :])),
                    axis=1,
                )
            ),
            semantic,
        )
    )
    if (
        result.shape != (count, 3)
        or np.any(~np.isfinite(result))
        or np.any(result < 0.0)
        or np.any(result > 1.0 + 1e-12)
    ):
        raise ValueError("three Panel-B block errors must be finite in [0, 1]")
    return result


def global_block_error(
    global_proportions: np.ndarray,
    target_global_proportions: np.ndarray,
) -> np.ndarray:
    """Return the normalized four-global-bar RMSE used by the audited v7-v10 display contract."""

    values = np.asarray(global_proportions, dtype=float)
    target = np.asarray(target_global_proportions, dtype=float).ravel()
    if values.ndim != 2 or values.shape[1] != 4 or target.shape != (4,):
        raise ValueError("global Panel-B bars must have four categories")
    result = np.sqrt(np.mean(np.square(values - target[None, :]), axis=1))
    result /= np.sqrt(0.5)
    if np.any(~np.isfinite(result)) or np.any(result > 1.0 + 1e-12):
        raise ValueError("global four-bar error must be finite in [0, 1]")
    return result
