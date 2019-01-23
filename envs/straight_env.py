#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Edward Ahn

Environment for training a local planner to move in a straight line.
"""

import csv

import numpy as np
import math

from rllab.envs.base import Step
from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv
from aa_simulation.misc.utils import normalize_angle


class StraightEnv(VehicleEnv):
    """
    Simulation environment for an RC car following a straight
    line trajectory. The reward function encourages the agent to
    move right on the line y=0 for all time.
    """

    def __init__(self, target_velocity):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(StraightEnv, self).__init__(target_velocity)


    @property
    def observation_space(self):
        """
        Define the shape of input vector to the neural network.
        """
        return Box(low=-np.inf, high=np.inf, shape=(5,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        # Total margin of domain randomization for each variable
        angle_margin = np.deg2rad(60)
        position_margin = 0.5
        velocity_margin = 1.5 * self.target_velocity

        # Compute domain randomized variables
        y = position_margin * np.random.random() - position_margin/2
        yaw = angle_margin * np.random.random() - angle_margin/2
        x_dot = velocity_margin * np.random.random()
        y_dot = velocity_margin * np.random.random() - velocity_margin/2

        state = np.zeros(6)
        state[1] = y
        state[2] = yaw
        state[3] = x_dot
        state[4] = y_dot
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
            x, y, _, x_dot, y_dot, _ = nextstate
            lambda1 = 0.25
            velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
            vel_diff = velocity - self.target_velocity
            distance = y
            reward = -np.absolute(distance)
            reward -= lambda1 * np.square(vel_diff)

        next_observation = self._state_to_observation(nextstate)
        return Step(observation=next_observation, reward=reward,
                done=done, dist=distance, vel=vel_diff)


    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = None
        self._state = self.get_initial_state
        observation = self._state_to_observation(self._state)

        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation


    @staticmethod
    def project_line(state, x0, y0, angle):
        """
        Note that this policy is trained to follow a straight line
        to the right (y = 0). To follow an arbitrary line, use this
        function to transform the current absolute state to a form
        that makes the policy believe the car is moving to the right.

        :param state: Current absolute state of robot
        :param x0: x-value of start of line to follow
        :param y0: y-value of start of line to follow
        :param angle: Angle of line to follow
        """
        x, y, yaw, x_dot, y_dot, yaw_dot = state
        angle = normalize_angle(angle)

        current_angle = np.arctan2((y - y0), (x - x0))
        projected_angle = normalize_angle(current_angle - angle)
        dist = np.sqrt(np.square(x - x0) + np.square(y - y0))

        new_y = dist * np.sin(projected_angle)
        new_yaw = normalize_angle(yaw - angle)
        new_x_dot = x_dot * np.cos(angle) + y_dot * np.sin(angle)
        new_y_dot = -x_dot * np.sin(angle) + y_dot * np.cos(angle)

        return np.array([new_y, new_yaw, new_x_dot, new_y_dot, yaw_dot])


    def _state_to_observation(self, state):
        """
        Prepare state to be read as input to neural network.
        """
        _, y, yaw, x_dot, y_dot, yaw_dot = state
        yaw = normalize_angle(yaw)
        return np.array([y, yaw, x_dot, y_dot, yaw_dot])

