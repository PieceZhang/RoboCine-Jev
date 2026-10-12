"""Obstacle-free kinematic tracking for fixture audits and trajectory previews.

This is an offline counterfactual, not the safety controller or a PhysX result.
No obstacle geometry or future obstacle trajectory enters this rollout.
"""
import numpy as np

from .actuation import advance_command
from .whole_body_qp import Limits, WholeBodyQP


def nominal_rollout(reference, initial_state, base_z):
    state = np.array(initial_state, dtype=float)
    q_command = state[3:].copy()
    limits = Limits(command_ramp=True, position_margin=.03)
    limits.velocity *= .95
    limits.wheel_speed *= .95
    limits.acceleration *= .4
    limits.acceleration[1] = Limits().acceleration[1]*.15
    limits.wheel_acceleration *= .4
    controller = WholeBodyQP(limits)
    times = np.linspace(0., reference.duration, round(reference.duration/.02)+1)
    states = [state.copy()]
    failure = ''
    for t, end in zip(times[:-1], times[1:]):
        dt = end-t
        previous = controller.previous.copy()
        controller.time_s = t
        try:
            command, _ = controller.solve(state, *reference.sample(t), dt,
                base_z=base_z, commanded_q=q_command)
        except RuntimeError as error:
            failure = str(error)
            break
        state = advance_command(state, previous, command, dt, True)
        q_command += .5*dt*(previous[2:]+command[2:])
        states.append(state.copy())
    return times[:len(states)], np.array(states), failure
