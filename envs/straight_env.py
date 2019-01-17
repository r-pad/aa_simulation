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
        '''
        Define the shape of input vector to the neural network.
        '''
        return Box(low=-np.inf, high=np.inf, shape=(5,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        y = 0.3*np.random.random() - 0.15
        yaw = np.deg2rad(30 * (np.random.random()) - 15)
        x_dot = np.random.random() * self.target_velocity
        y_dot = (2*np.random.random() - 1) * self.target_velocity
        yaw_dot = 2*np.random.random() - 1

        state = np.zeros(6)
        state[1] = y
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
            lambda1 = 0.2
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


    def _state_to_observation(self, state):
        """
        Prepare state to be read as input to neural network.
        """
        _, y, yaw, x_dot, y_dot, yaw_dot = state
        yaw = self._normalize_angle(yaw)
        return np.array([y, yaw, x_dot, y_dot, yaw_dot])


    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi).
        """
        angle = angle % (2*np.pi)
        if (angle >= np.pi):
            angle -= 2*np.pi
        return angle
