"""Numeric thresholds + config signature (single source of truth).

Defaults reproduce the pre-refactor behavior exactly. Override via config.yaml.
"""
import json

# Per-direction collision distance (metres). 前 1.0, 左/右/上/下 0.5.
DIST_THRESH = {"前": 1.0, "左": 0.5, "右": 0.5, "上": 0.5, "下": 0.5}
# Back-compat aliases (old code used a single horiz / vert value).
DIST_THRESH_HORIZ = DIST_THRESH["前"]
DIST_THRESH_VERT = DIST_THRESH["上"]

FRAC_THRESH = 0.05        # near-pixel fraction within a region above which obstacle=True
VERT_FRAC = 0.25          # top/bottom strip height fraction; trimmed before horizontal split
DEPTH_MIN, DEPTH_MAX = 0.05, 80.0   # valid-depth mask bounds (m)
RANK_THRESH = 2.0         # loose distance used by mining score_frame (candidate ranking only)
TARGET_RATE = 0.35        # balance: each direction must reach this positive fraction
CONFIG_VERSION = 2        # bump to invalidate prior depth outputs


def build_config_sig(dist_thresh=None, frac_thresh=None, vert_frac=None, version=None):
    """Stable signature of the depth-decision config, written into the meta file
    so stale outputs (changed thresholds) can be detected. Matches the historical
    scheme (sort_keys) so existing meta files keep validating."""
    return json.dumps(
        {
            "v": version if version is not None else CONFIG_VERSION,
            "thresh": dist_thresh if dist_thresh is not None else DIST_THRESH,
            "frac": frac_thresh if frac_thresh is not None else FRAC_THRESH,
            "scheme": "per_env_model",
            "vert_frac": vert_frac if vert_frac is not None else VERT_FRAC,
        },
        sort_keys=True,
    )


# Canonical signature for the current defaults — compared against on-disk meta.
CONFIG_SIG = build_config_sig()
