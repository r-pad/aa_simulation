#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow arbitrary
curvatures.
"""

import csv

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv


class LocalPlannerEnv(VehicleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates.
    """

    def __init__(self, radius, target_velocity):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(LocalPlannerEnv, self).__init__(target_velocity)

        # Radius of trajectory to follow
        self.radius = radius


    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(4,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        state = np.zeros(6)
        state[0] = -self.radius
        state[2] = np.deg2rad(270)
        return state


    def step(self, action):
        """
        Move one iteration forward in simulation.
        """
        # Get next state from dynamics equations
        self._action = action
        nextstate = self._model.state_transition(self._state, action,
                self._dt)

        # Check collision and assign reward to transition
        collision = self._check_collision(nextstate)
        if collision:
            reward = -100
            done = True
            distance = np.inf
            vel_diff = np.inf
        else:
            self._state = nextstate
            done = False

            # Trajectory following
            r = self.radius
            x, y, _, x_dot, y_dot, _ = nextstate
            lambda1 = 0.25
            velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
            vel_diff = velocity - self.target_velocity
            distance = r-np.sqrt(np.square(x)+np.square(y))
            reward = -np.abs(distance)
            reward -= lambda1 * np.square(vel_diff)

        next_observation = self._state_to_relative(nextstate)
        return Step(observation=next_observation, reward=reward,
                done=done, dist=distance, vel=vel_diff)


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
        theta = self._normalize_angle(np.arctan2(-x, y) + np.pi - yaw)
        ddx = x/(x**2 + y**2)**0.5*x_dot + y/(x**2 + y**2)**0.5*y_dot
        dtheta = x/(x**2 + y**2)*x_dot - y/(x**2 + y**2)*y_dot - yaw_dot

        #return np.array([dx/0.012, theta/0.067, ddx/-0.4139, dtheta/-0.6505])
        return np.array([dx, theta, ddx, dtheta])


    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi).
        """
        angle = angle % (2*np.pi)
        if (angle > np.pi):
            angle -= 2*np.pi
        return angle

