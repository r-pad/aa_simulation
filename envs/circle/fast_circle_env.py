#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow circles as fast as possible.
"""

import numpy as np

from rllab.envs.base import Step

from aa_simulation.envs.circle.circle_env import CircleEnv


class FastCircleEnv(CircleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates as fast as possible.
    """

    def __init__(self, target_velocity, radius, dt, model_type, robot_type):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(FastCircleEnv, self).__init__(
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
        velocity = np.sqrt(x_dot**2 + y_dot**2)
        distance = r - np.sqrt(x**2 + y**2)

        reward = velocity**2

        info = {}
        info['observation'] = observation
        info['dist'] = distance
        info['vel'] = velocity
        return reward, info

