#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow circles of
arbitrary curvature.
"""

import csv

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv
from aa_simulation.misc.utils import normalize_angle


class CircleEnv(VehicleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates.
    """

    def __init__(self, target_velocity, radius, dt):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(CircleEnv, self).__init__(target_velocity, dt)

        # Radius of trajectory to follow
        self.radius = radius

        # Reward function parameters
        self._lambda1 = 0.25
        self._lambda2 = 0.25


    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(4,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        # Compute domain randomized variables
        x = np.random.uniform(-0.25, 0.25) - self.radius
        yaw = np.random.uniform(-np.pi/3, np.pi/3) + np.deg2rad(270)
        x_dot = np.random.uniform(0, 1.3)
        y_dot = np.random.uniform(-0.6, 0.6)
        yaw_dot = np.random.uniform(-2.0, 2.0)

        state = np.zeros(6)
        state[0] = x
        state[2] = yaw
        state[3] = x_dot
        state[4] = y_dot
        state[5] = yaw_dot
        return state


    def step(self, action):
        """
        Move one iteration forward in simulation.
        """
        # Get next state from dynamics equations
        self._action = action
        if self._action[0] < 0:     # Only allow forward direction
            self._action[0] = 0
        nextstate = self._model.state_transition(self._state, action,
                self._dt)
        next_observation = self._state_to_relative(nextstate)

        # Assign reward to transition
        self._state = nextstate
        done = False
        r = self.radius
        x, y, _, x_dot, y_dot, _ = nextstate
        dx, dth, dx_dot, dth_dot = next_observation
        velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
        vel_diff = velocity - self.target_velocity
        distance = r - np.sqrt(np.square(x)+np.square(y))
        reward = -np.abs(distance)
        reward -= self._lambda1 * vel_diff**2
        reward -= self._lambda2 * max(0, abs(dth) - np.pi/2)**2

        return Step(observation=next_observation, reward=reward,
                done=done, dist=distance, vel=vel_diff,
                kappa=self._model.kappa)


    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = None
        self._state = self.get_initial_state
        observation = self._state_to_relative(self._state)

        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation


    def _state_to_relative(self, state):
        """
        Convert state [x, y, yaw, x_dot, y_dot, yaw_dot] to
        [dx, theta, ddx, dtheta]
        """
        r = self.radius
        x, y, yaw, x_dot, y_dot, yaw_dot = state

        dx = np.sqrt(np.square(x) + np.square(y)) - r
        theta = normalize_angle(np.arctan2(-x, y) + np.pi - yaw)
        ddx = x/(x**2 + y**2)**0.5*x_dot + y/(x**2 + y**2)**0.5*y_dot
        dtheta = x/(x**2 + y**2)*x_dot - y/(x**2 + y**2)*y_dot - yaw_dot

        # May want to rescale/normalize values to each other.
        return np.array([dx, theta, ddx, dtheta])

