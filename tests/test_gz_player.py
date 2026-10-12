"""path_player pure interpolation tests (no rclpy / gazebo_msgs — macOS safe)."""

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "ros2_ws" / "cinema_traj_gz_demo"))

from cinema_traj_gz_demo.path_player import _nlerp, interp_pose  # noqa: E402


def _poses():
    return [
        {"stamp_s": 0.0, "position": (0.0, 0.0, 1.0), "orientation": (1.0, 0.0, 0.0, 0.0)},
        {"stamp_s": 2.0, "position": (2.0, 0.0, 1.0), "orientation": (0.0, 0.0, 0.0, 1.0)},
        {"stamp_s": 4.0, "position": (2.0, 2.0, 2.0), "orientation": (0.0, 0.0, 1.0, 0.0)},
    ]


class TestInterpPose:
    def test_clamped_at_both_ends(self):
        p = _poses()
        assert interp_pose(p, -1.0) is p[0]
        assert interp_pose(p, 99.0) is p[-1]

    def test_midpoint_interpolation(self):
        out = interp_pose(_poses(), 1.0)
        assert out["position"] == pytest.approx((1.0, 0.0, 1.0))
        # halfway between identity and 180°-about-z ⇒ 90° about z
        w, x, y, z = out["orientation"]
        assert w == pytest.approx(math.cos(math.pi / 4), abs=1e-9)
        assert z == pytest.approx(math.sin(math.pi / 4), abs=1e-9)
        assert x == pytest.approx(0.0, abs=1e-9) and y == pytest.approx(0.0, abs=1e-9)

    def test_quaternions_stay_normalized(self):
        for t in (0.3, 1.7, 3.9):
            q = interp_pose(_poses(), t)["orientation"]
            assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0, abs=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            interp_pose([], 0.0)


class TestNlerp:
    def test_shortest_arc_negates_opposite_quaternion(self):
        qa, qb = (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)   # same rotation
        out = _nlerp(qa, qb, 0.5)
        assert out == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)
