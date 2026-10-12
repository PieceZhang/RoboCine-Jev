"""Video labels follow physical time and real inter-shot reconfiguration."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_dynamic_dual_view.py"
SPEC = importlib.util.spec_from_file_location("dynamic_dual_view", SCRIPT)
video = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(video)


def run(reference=None):
    # The physical clock slows relative to the path clock during avoidance.
    rows = [dict(phase="preparing", time_s="-1", shot_type="orbit",
                 transition_active="0", motion_active="0", position_error=".4", orientation_error=".5"),
            dict(phase="tracking", time_s="0", shot_type="orbit",
                 transition_active="0", motion_active="1", position_error=".0123", orientation_error=".01"),
            dict(phase="tracking", time_s="2", shot_type="orbit",
                 transition_active="1", motion_active="0", position_error=".5", orientation_error=".8"),
            dict(phase="tracking", time_s="4", shot_type="pan",
                 transition_active="0", motion_active="1", position_error=".0234", orientation_error=".02"),
            dict(phase="tracking", time_s="6", shot_type="pan",
                 transition_active="0", motion_active="0", position_error=".6", orientation_error=".9"),
            dict(phase="stopping", time_s="8", shot_type="pan",
                 transition_active="0", motion_active="0", position_error=".7", orientation_error="1.0")]
    return SimpleNamespace(rows=rows, elapsed=np.array([0., 1., 5., 9., 10., 11.]),
                           reference=reference or {})


def intershot_reference():
    return {"execution_shots": [dict(type="orbit", start_s=0., end_s=3.,
                                      motion_start_s=0., motion_end_s=2.),
                                 dict(type="pan", start_s=3., end_s=8.,
                                      motion_start_s=4., motion_end_s=6.)]}


def test_avoidance_wall_clock_cannot_anticipate_a_transition_or_next_shot():
    source = run(intershot_reference())
    before = video.frame_annotation(source, 4.999)
    assert before["phase"] == "orbit"
    assert before["sample_wall_elapsed_s"] == 1.
    assert before["position_error_cm"] == pytest.approx(1.23)
    assert before["orientation_error_deg"] == pytest.approx(np.degrees(.01))
    during = video.frame_annotation(source, 5.)
    assert during["phase"] == "arc"
    assert during["position_error_cm"] is None
    assert during["orientation_error_deg"] is None
    assert video.frame_annotation(source, 8.999)["phase"] == "arc"
    assert video.frame_annotation(source, 9.)["phase"] == "pan"


def test_missing_reference_cannot_invent_arc_and_holds_have_no_errors():
    source = run()
    for t in (-.1, 0., 5., 10., 11., 50.):
        state = video.frame_annotation(source, t)
        assert state["phase"] == "idle"
        assert not state["recording"]
        assert state["position_error_cm"] is None
        assert state["orientation_error_deg"] is None


def test_phase_and_motion_flags_guard_connector_metadata():
    source = run(intershot_reference())
    assert video.frame_annotation(source, 0.)["phase"] == "idle"
    assert video.frame_annotation(source, 9.)["phase"] == "pan"
    assert video.frame_annotation(source, 11.)["phase"] == "idle"


def test_stationary_camera_intershot_base_reconfiguration_is_arc_not_hold():
    source = run(intershot_reference())
    # The controller holds the camera pose while it repositions the base.
    # The saved inter-shot window plus transition_active identifies that
    # execution stage; no camera-motion inference or fixed timer is needed.
    source.rows[2].update(command_v=".04", command_omega=".2",
                          camera_x=".1", camera_y=".2", camera_z="1.")
    during = video.frame_annotation(source, 5.)
    assert during["phase"] == "arc"
    assert not during["recording"]
    assert during["position_error_cm"] is None
    assert during["orientation_error_deg"] is None
    assert video.frame_annotation(source, 0.)["phase"] == "idle"
    assert video.frame_annotation(source, 10.)["phase"] == "idle"
    source.rows[2]["transition_active"] = "0"
    assert video.frame_annotation(source, 5.)["phase"] == "idle"


def test_rendered_red_and_green_indicators_and_shooting_only_error_row():
    # Use the same canvas and font as production so the indicator is actually
    # beside the left camera title and the error disappears from arc frames.
    source = run(intershot_reference())
    pixels = np.zeros((540, 960, 3), dtype=np.uint8)
    shot = video.caption_frame(pixels, pixels, "crossing", source, 1.)
    arc = video.caption_frame(pixels, pixels, "crossing", source, 5.)
    stopping = video.caption_frame(pixels, pixels, "crossing", source, 11.)
    assert shot.shape == arc.shape == (664, 1920, 3)
    title_width = video.caption_font(22).getbbox("camera 第一人称视角")[2]
    center = (video.HEADER_HEIGHT // 2, 20 + title_width + 24)
    np.testing.assert_array_equal(shot[center], video.RECORDING_RED)
    np.testing.assert_array_equal(arc[center], video.IDLE_GREEN)
    np.testing.assert_array_equal(stopping[center], video.IDLE_GREEN)
    footer_error_region = np.s_[540 + video.HEADER_HEIGHT + 39:664, :960]
    assert np.any(shot[footer_error_region] != (19, 26, 35))
    assert np.all(arc[footer_error_region] == (19, 26, 35))
    assert np.all(stopping[footer_error_region] == (19, 26, 35))
