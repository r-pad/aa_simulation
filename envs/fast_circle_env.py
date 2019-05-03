#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow circles as fast as possible.
"""

import csv

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from aa_simulation.envs.circle_env import CircleEnv
from aa_simulation.misc.utils import normalize_angle


class FastCircleEnv(CircleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates as fast as possible.
    """

    def __init__(self, target_velocity, radius, dt, model_type):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(FastCircleEnv, self).__init__(
            target_velocity=target_velocity,
            radius=radius,
            dt=dt,
            model_type=model_type
        )


    def step(self, action):
        """
        Move one iteration forward in simulation.
        """
        # Get next state from dynamics equations
        if action[0] < 0:   # Only allow forward direction
            action[0] = 0
        self._action = action
        nextstate = self._model.state_transition(self._state, action,
                self._dt)
        next_observation = self._state_to_relative(nextstate)

        # Assign reward to transition
        self._state = nextstate
        done = False
        r = self.radius
        x, y, _, x_dot, y_dot, _ = nextstate
        velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
        distance = r - np.sqrt(x**2 + y**2)
        reward = velocity**2

        return Step(observation=next_observation, reward=reward,
                done=done, dist=distance, vel=velocity,
                kappa=self._model.kappa)

