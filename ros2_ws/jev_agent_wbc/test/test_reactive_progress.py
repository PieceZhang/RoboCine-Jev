from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np

from jev_agent_wbc.engine import configure_reactive_progress
from jev_agent_wbc.observed_clock import update


def test_installing_a_verified_plan_disables_unplanned_retiming():
    original = NS(is_combo=False)
    controller = NS()
    configure_reactive_progress(controller, original, original, True)
    assert controller.body_aware and controller.observed_clock_enabled
    approved = NS(original=original)
    configure_reactive_progress(controller, original, approved, True)
    assert not controller.body_aware and not controller.observed_clock_enabled


def test_combinations_and_disabled_backend_settings_are_preserved():
    original = NS(is_combo=True)
    controller = NS()
    configure_reactive_progress(controller, original, original, True)
    assert not controller.body_aware and not controller.observed_clock_enabled
    controller.body_aware = controller.observed_clock_enabled = True
    configure_reactive_progress(controller, original, original, False)
    assert controller.body_aware and controller.observed_clock_enabled


def cached_clock(last):
    reference = object()
    controller = NS(clock_guide_reference=reference, clock_guide_future=None,
                    clock_guide=None, reconfiguration=NS(pool=None),
                    clock_observation=last)
    return controller, reference


def test_servo_advances_clock_between_sensor_updates_without_rewriting_observation():
    last = dict(at=2., active=True, progress=3., rate=.5, phase_end=8.)
    before = deepcopy(last)
    controller, reference = cached_clock(last)
    positions = []
    for at in (2.02, 2.04, 2.06, 2.08):
        info = update(controller, reference, np.zeros(9), np.zeros(8), at, 0.)
        positions.append(info["progress"])
    np.testing.assert_allclose(positions, [3.01, 3.02, 3.03, 3.04])
    assert last == before and controller.clock_observation is last


def test_cached_clock_never_crosses_phase_deadline_or_creates_motion_when_inactive():
    last = dict(at=2., active=True, progress=7.99, rate=2.5, phase_end=8.)
    controller, reference = cached_clock(last)
    info = update(controller, reference, np.zeros(9), np.zeros(8), 2.08, 0.)
    assert info["progress"] == 8.
    inactive = dict(at=2., active=False, rate=1.)
    controller, reference = cached_clock(inactive)
    assert update(controller, reference, np.zeros(9), np.zeros(8), 2.08, 0.) is inactive
