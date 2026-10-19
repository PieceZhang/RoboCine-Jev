"""Weighted resolved-rate control for the native 3-DOF planar base + UR5.

The Jacobian and feedback poses are measured from PhysX. This module never
teleports a link; its output is a bounded generalized joint velocity.
"""
import numpy as np


class WholeBodyController:
    def __init__(self, home):
        self.home = np.asarray(home, dtype=float)
        self.mobility = np.diag([0.35, 0.35, 0.5, 1., 1., 1., 1., 1., 1.])
        self.limits = np.array([0.25, 0.25, 0.35, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9])

    def velocity(self, q, jacobian, position_error, rotation_error, feedforward):
        twist = np.asarray(feedforward) + np.r_[4.0 * position_error, 3.0 * rotation_error]
        inverse = self.mobility @ jacobian.T @ np.linalg.inv(
            jacobian @ self.mobility @ jacobian.T + 0.025**2 * np.eye(6))
        secondary = -0.15 * (q - self.home)
        velocity = inverse @ twist + (np.eye(9) - inverse @ jacobian) @ secondary
        return velocity / max(1.0, float(np.max(np.abs(velocity) / self.limits)))
