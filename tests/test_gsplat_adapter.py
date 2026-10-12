"""gsplat_render adapter tests — pure-math coverage, NO GPU needed.

The point of these tests: the gsplat viewmat/PLY conventions are a silent-fail
surface — a flipped axis yields a perfectly-valid but mirrored rendering. These
checks pin the physical camera semantics against OpenCV/gsplat conventions.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tools.gsplat_render import intrinsics, load_3dgs_ply, viewmat_from_pos_look


def _project(eye, look, pt, fx=250.0, w=640, h=360):
    vm = viewmat_from_pos_look(eye, look)
    p = vm @ np.append(np.asarray(pt, float), 1.0)
    return fx * p[0] / p[2] + w / 2, fx * p[1] / p[2] + h / 2


def test_viewmat_matches_measured_kit_truth():
    """viewmat pinned to measurements against real Kit renders
    (chiral probe stages, this session): facing -Z ⇒ +X goes right,
    +Y goes up; facing +Z ⇒ +X goes left (Kit's own mirror convention).
    Simview renders use the identical physical convention — this pins
    pixel-parity between gsplat and the Isaac viewer."""
    # facing -Z (south-like)
    assert _project([0, 0, 0], [0, 0, -4], [1.2, 0, -4])[0] > 320      # +X → right
    assert _project([0, 0, 0], [0, 0, -4], [0, 1, -4])[1] < 180        # +Y → above
    # facing +Z (north-like) — measured left, mirroring Kit's image convention
    assert _project([0, 0, -4], [0, 0, 0], [1.2, 0, 0])[0] < 320       # +X → left
    assert _project([0, 0, -4], [0, 0, 0], [0, 1, 0])[1] < 180
    # aim lands dead center regardless
    u, v = _project([1.3, 2.1, -1.4], [0.6, 1.5, -2.1], [0.6, 1.5, -2.1])
    assert (u, v) == pytest.approx((320, 180), abs=1e-6)


def test_viewmat_is_a_proper_rotation():
    vm = viewmat_from_pos_look([1.2, 1.9, -0.8], [0.5, 1.5, -2.4])
    R = np.linalg.inv(vm)[:3, :3]                      # c2w rotation block
    assert np.linalg.det(R) == pytest.approx(1.0)      # recourse: no reflection


def test_intrinsics_square_pixels():
    k = intrinsics(360, 640, 60.0)
    f = (360 / 2) / np.tan(np.radians(30))
    assert k[0, 0] == pytest.approx(f) and k[1, 1] == pytest.approx(f)
    assert k[0, 2] == 320 and k[1, 2] == 180


def _mini_3dgs_ply(path):
    """Write a 2-vertex valid 3DGS-a-like PLY with known values."""
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
            + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    v0 = [-1.0, 2.0, 3.0,  +0.35355496, 0, -0.35355496, 0.0,  # f_dc → 0.6/0.5/0.4
          0.5, 0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][:14]
    v1 = [4.0, -5.0, 6.0,  0, 0, 0,
          1.0, -0.1, -0.2, -0.3, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0][:14]
    with open(path, "wb") as f:
        f.write(head.encode("latin1"))
        f.write(struct.pack("<14f", *v0))
        f.write(struct.pack("<14f", *v1))


def test_load_3dgs_ply_conversions(tmp_path):
    p = tmp_path / "s.ply"
    _mini_3dgs_ply(p)
    g = load_3dgs_ply(p, up="y")
    assert g["means"].shape == (2, 3)
    assert g["means"][0, 1] == pytest.approx(2.0)
    # f_dc=0.35355496 → color 0.5 + 0.2820548·0.35355496
    assert g["colors"][0, 0] == pytest.approx(0.59972185, abs=1e-6)
    # scales stay in LOG space here (gsplat.exp in the runner); rot raw
    assert g["quats"][1, 1] == pytest.approx(1.0)

    # Z-up input: (x,y,z) → (x,z,-y) then shift
    g = load_3dgs_ply(p, up="z", shift=[0, 1.5, 0])
    assert g["means"][0][0] == pytest.approx(-1.0)
    assert g["means"][0][1] == pytest.approx(3.0 + 1.5)
    assert g["means"][0][2] == pytest.approx(-2.0)
