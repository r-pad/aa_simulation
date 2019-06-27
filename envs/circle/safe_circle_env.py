#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow circles at a target velocity,
with CPO enforcing a distance constraint.
"""

import numpy as np

from rllab.envs.base import Step

from aa_simulation.envs.circle.circle_env import CircleEnv


class SafeCircleEnv(CircleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory at a target velocity with CPO enforcing the distance
    constraint (reward function doesn't).
    """

    def __init__(self, target_velocity, radius, dt, model_type, robot_type):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(SafeCircleEnv, self).__init__(
            target_velocity=target_velocity,
            radius=radius,
            dt=dt,
            model_type=model_type,
            robot_type=robot_type
        )


    def get_reward(self, state, action):
        """
        Reward function definition.
        """
        observation = self.state_to_observation(state)
        r = self.radius
        x, y, _, x_dot, y_dot, _ = state
        _, theta, _, _ = observation
        velocity = np.sqrt(x_dot**2 + y_dot**2)
        distance = r - np.sqrt(x**2 + y**2)

        reward = -(velocity - self.target_velocity)**2
        reward -= max(0, abs(theta) - np.pi/2)**2

        info = {}
        info['dist'] = distance
        info['vel'] = velocity
        return reward, info

