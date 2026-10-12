"""5-region depth geometry on a synthetic depth map."""
import numpy as np

from vlm_intention.common.geometry import region_depths, region_stats, score_frame
from vlm_intention.common.schema import DIRS


def _synthetic(h=400, w=600, near=0.2, far=10.0):
    d = np.full((h, w), far, dtype=np.float32)
    h_top = int(h * 0.25); w3 = w // 3
    d[:h_top, :] = near          # 上 close
    d[h_top:h - h_top, 2 * w3:] = near   # 右 close (middle band)
    return d


def test_region_depths_shapes():
    r = region_depths(_synthetic())
    assert set(r) == set(DIRS)
    assert r["上"].shape[0] == 100         # 25% of 400
    assert r["前"].shape == (200, 200)     # middle band, middle third


def test_region_stats_detects_close():
    stats = region_stats(_synthetic())
    assert stats["上"]["obstacle"] is True     # near strip
    assert stats["右"]["obstacle"] is True
    assert stats["前"]["obstacle"] is False
    assert stats["下"]["obstacle"] is False
    assert 0.0 <= stats["左"]["near_ratio"] <= 1.0


def test_score_frame_keys():
    s = score_frame(_synthetic())
    assert set(s["strict"]) == set(DIRS)
    assert set(s["loose"]) == set(DIRS)
    assert s["strict"]["上"] > 0.5
