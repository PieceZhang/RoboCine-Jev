"""Continuous targets, startup-clock isolation, dropout and physical bounds."""
import numpy as np
import pytest
from jvs_mock.joint_command_filter import JointCommandFilter


def servo(q=0.):
    return JointCommandFilter([q], [-1.], [1.], [1.], 4.5)


def test_zero_frequency_forwards_native_targets_without_filter_delay():
    s = JointCommandFilter([0.], [-1.], [1.], [1.], 0.)
    assert s.push(.1, [.2], [.4], [1., -1.], .101)
    q, v, w = s.step(.102, .001)
    np.testing.assert_array_equal(q, [.2])
    np.testing.assert_array_equal(v, [.4])
    np.testing.assert_array_equal(w, [1., -1.])
    assert s.push(.103, [.199], [-.4], [-1., 1.], .103)
    q, v, w = s.step(.104, .001)
    np.testing.assert_array_equal(q, [.199])
    np.testing.assert_array_equal(v, [-.4])
    np.testing.assert_array_equal(w, [-1., 1.])


def test_unfiltered_dropout_holds_position_and_stops_all_velocity_targets():
    s = JointCommandFilter([0.], [-1.], [1.], [1.], 0.)
    assert s.push(.1, [.2], [.4], [1., -1.], .1)
    q, v, w = s.step(.251, .001)
    np.testing.assert_array_equal(q, [.2])
    np.testing.assert_array_equal(v, [0.])
    np.testing.assert_array_equal(w, [0., 0.])


def test_new_command_changes_reference_without_teleporting_the_drive_target():
    s=servo()
    assert s.push(0., [.3], [0.], [1., -1.], 0.)
    assert s.position[0] == s.velocity[0] == 0.
    q,v,w=s.step(.001,.001)
    assert 0. < q[0] < .001 and 0. < v[0] <= 1.
    assert 0. < w[0] < 1.


def test_constant_target_settles_without_overshoot():
    s=servo();s.push(0., [.2], [0.], [0.,0.], 0.)
    values=[s.step(i*.001,.001)[0][0] for i in range(1,2501)]
    assert min(values)>0. and max(values)<=.2
    assert values[-1] == pytest.approx(.2,abs=1e-4)


def test_startup_wall_time_cannot_poison_simulation_clock_commands():
    s=servo()
    assert not s.push(1791453509., [0.], [0.], [0.,0.], .1)
    assert s.push(.1,[0.],[0.],[0.,0.],.117)
    assert s.unsynchronized == 1 and s.rejected == 0


def test_sparse_input_stays_continuous_and_stale_wheels_stop():
    s=servo();s.push(0., [.2], [.4], [1.,-1.], 0.)
    for i in range(1,1500):
        before=s.position.copy();q,v,w=s.step(i*.001,.001)
        assert np.max(np.abs(q-before))<=.001+1e-12
        assert np.max(np.abs(v))<=1.+1e-12
    assert np.max(np.abs(w))<1e-6
    assert q[0] == pytest.approx(.2,abs=.001)


def test_reversal_near_joint_stop_respects_bounds():
    s=servo(.98);s.push(0.,[.99],[1.],[0.,0.],0.)
    for i in range(1,200):
        if i==70:s.push(.07,[.96],[-1.],[0.,0.],.07)
        q,v,_=s.step(i*.001,.001)
        assert -1. <= q[0] <= 1. and abs(v[0])<=1.+1e-12


def test_invalid_commands_do_not_replace_last_valid_reference():
    s=servo();assert s.push(.1,[.1],[.1],[0.,0.],.1)
    for q,v,w in [([1.1],[0.],[0.,0.]),([0.],[1.1],[0.,0.]),([np.nan],[0.],[0.,0.]),
                  ([0.],[0.],[16.,0.])]:
        assert not s.push(.2,q,v,w,.2)
    assert not s.push(.09,[0.],[0.],[0.,0.],.1)
    assert s.rejected==5 and s.command[0] == .1


def test_filter_tail_cannot_carry_a_joint_past_its_reserved_stop():
    s=servo(.8)
    for i in range(1,1000):
        time=i*.001
        if i%17==1:s.push(time,[.99],[1.],[0.,0.],time)
        q,v,_=s.step(time,.001)
        endpoint=q+2*v/s.omega+s.acceleration[2:]/s.omega**2
        assert endpoint[0] <= .995+1e-6
        assert q[0]<=.995+1e-6


def test_nonfinite_initial_state_limits_or_clock_are_rejected():
    for q, lower, upper, speed in [([np.nan], [-1.], [1.], [1.]),
            ([2.], [-1.], [1.], [1.]), ([0.], [-np.inf], [1.], [1.]),
            ([0.], [-1.], [1.], [np.nan])]:
        with pytest.raises(ValueError):
            JointCommandFilter(q, lower, upper, speed, 2.)
    s = servo()
    assert not s.push(.1, [0.], [0.], [0., 0.], np.nan)
    with pytest.raises(ValueError):
        s.step(np.nan, .001)
