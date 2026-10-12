"""Deterministic 5-region depth geometry + statistics.

前/左/右 = horizontal thirds of the VERT_FRAC-trimmed middle band;
上/下    = full-width top/bottom strips (incl. corners).

Consolidates region_depths/region_stats (depth_annotation), score_frame
(mine_obstacle_frames), and close_frac_from_hist (calibrate_depth).
"""
import numpy as np

from .schema import DIRS, HORIZ_DIRS, VERT_DIRS
from .thresholds import (
    DIST_THRESH, FRAC_THRESH, VERT_FRAC, DEPTH_MIN, DEPTH_MAX, RANK_THRESH,
)


def region_depths(depth, vert_frac=VERT_FRAC):
    """Slice a depth map into 5 non-overlapping regions -> {dir: array}."""
    h, w = depth.shape
    w3 = w // 3
    h_top = int(h * vert_frac)
    mid0, mid1 = h_top, h - h_top
    return {
        "前": depth[mid0:mid1, w3:2 * w3],
        "左": depth[mid0:mid1, :w3],
        "右": depth[mid0:mid1, 2 * w3:],
        "上": depth[:h_top, :],
        "下": depth[h - h_top:, :],
    }


def valid_depth_mask(arr, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX):
    return np.isfinite(arr) & (arr >= depth_min) & (arr <= depth_max)


def region_stats(depth, dist_thresh=None, frac_thresh=FRAC_THRESH,
                 vert_frac=VERT_FRAC, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX):
    """Per-direction decision metadata:
    {dir: {threshold, near_ratio, valid_pixels, median_depth, obstacle}}."""
    dist_thresh = dist_thresh or DIST_THRESH
    regions = region_depths(depth, vert_frac=vert_frac)
    out = {}
    for d in DIRS:
        arr = regions[d]
        valid = valid_depth_mask(arr, depth_min, depth_max)
        n = int(valid.sum())
        thresh = dist_thresh[d]
        if n == 0:
            out[d] = {"threshold": thresh, "near_ratio": 0.0,
                      "valid_pixels": 0, "median_depth": None, "obstacle": False}
            continue
        near = float(((arr < thresh) & valid).sum()) / n
        out[d] = {
            "threshold": thresh,
            "near_ratio": near,
            "valid_pixels": n,
            "median_depth": float(np.median(arr[valid])),
            "obstacle": near > frac_thresh,
        }
    return out


def close_frac(arr, dist_thresh, depth_min=DEPTH_MIN, depth_max=DEPTH_MAX):
    """Fraction of valid region pixels closer than dist_thresh."""
    valid = valid_depth_mask(arr, depth_min, depth_max)
    n = int(valid.sum())
    return float(((arr < dist_thresh) & valid).sum()) / n if n else 0.0


def score_frame(depth, dist_thresh=None, frac_thresh=FRAC_THRESH,
                rank_thresh=RANK_THRESH, vert_frac=VERT_FRAC,
                depth_min=DEPTH_MIN, depth_max=DEPTH_MAX):
    """Mining score: strict near_ratio (at dist_thresh, for labeling) + loose
    near_ratio (at rank_thresh, for candidate ranking) + median, per direction."""
    dist_thresh = dist_thresh or DIST_THRESH
    regions = region_depths(depth, vert_frac=vert_frac)
    strict, loose, median = {}, {}, {}
    for d in DIRS:
        arr = regions[d]
        valid = valid_depth_mask(arr, depth_min, depth_max)
        n = int(valid.sum())
        strict[d] = float(((arr < dist_thresh[d]) & valid).sum()) / n if n else 0.0
        loose[d] = float(((arr < rank_thresh) & valid).sum()) / n if n else 0.0
        median[d] = float(np.median(arr[valid])) if n else None
    return {"strict": strict, "loose": loose, "median": median}


def close_frac_from_hist(hist, bin_edges, dist_thresh):
    """Estimate close-fraction from a cached histogram (calibration fast-path)."""
    total = hist.sum()
    if total == 0:
        return 0.0
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return float(hist[centers < dist_thresh].sum()) / float(total)


HORIZ_DIRS = HORIZ_DIRS  # re-export for callers importing from geometry
VERT_DIRS = VERT_DIRS
