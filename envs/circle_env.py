#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow circles of
arbitrary curvature.
"""

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

    def __init__(self, target_velocity, radius, dt, model_type):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(CircleEnv, self).__init__(
            target_velocity=target_velocity,
            dt=dt,
            model_type=model_type
        )

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
        x_dot = np.random.uniform(0, 2*self.target_velocity)
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
        if action[0] < 0:   # Only allow forward direction
            action[0] = 0
        self._action = action
        nextstate = self._model.state_transition(self._state, action,
                self._dt)
        self._state = nextstate
        reward, info = self.get_reward(nextstate, action)
        return Step(observation=info['observation'], reward=reward, done=False,
                dist=info['dist'], vel=info['vel'], kappa=self._model.kappa)


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


    def get_reward(self, state, action):
        """
        Reward function definition.
        """
        observation = self._state_to_relative(state)
        r = self.radius
        x, y, _, x_dot, y_dot, _ = state
        dx, dth, dx_dot, dth_dot = observation
        velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
        vel_diff = velocity - self.target_velocity
        distance = r - np.sqrt(x**2 + y**2)

        reward = -np.abs(distance)
        reward -= self._lambda1 * vel_diff**2
        reward -= self._lambda2 * max(0, abs(dth) - np.pi/2)**2

        info = {}
        info['observation'] = observation
        info['dist'] = distance
        info['vel'] = vel_diff
        return reward, info


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

