#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for driving car in circular arc trajectory using relative
coordinates.
"""

import csv

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from car_simulation.envs.base import VehicleEnv


class ArcRelativeEnv2(VehicleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates.
    """

    def __init__(self):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(ArcRelativeEnv2, self).__init__()

        # Get obstacles from CSV file
        #   Convention: (x, y, r) for each obstacle, which is a circle
        filename = 'car_simulation/envs/arc/obstacles.csv'
        with open(filename, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            values = list(reader)
            obstacle_list = [[float(x) for x in row] for row in values]
            self._obstacles = np.array(obstacle_list)

        # Radius of trajectory to follow
        self.radius = 1.5


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
        else:
            self._state = nextstate
            done = False

            # Trajectory following
            r = self.radius
            x, y, _, x_dot, y_dot, _ = nextstate
            target_velocity = 0.7
            lambda1 = 0.25
            velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
            distance = np.abs(r-np.sqrt(np.square(x)+np.square(y)))
            reward = -distance
            reward -= lambda1 * np.square(velocity - target_velocity)

        next_observation = self._state_to_relative(nextstate)
        return Step(observation=next_observation, reward=reward,
                done=done)


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
        [dx, dy, yaw, dx_dot, dy_dot, yaw_dot]
        """
        r = self.radius
        x, y, yaw, x_dot, y_dot, yaw_dot = state

        dx = np.sqrt(np.square(x) + np.square(y)) - r
        dx_dot = np.sqrt(np.square(x_dot) + np.square(y_dot))
        theta = self._normalize_angle(np.arctan2(-x, y) - yaw)
        theta_dot = self._normalize_angle(np.arctan2(-x_dot, y_dot) - yaw_dot)
        print('theta = %f', theta)
        print('theta_dot = %f', theta_dot)

        return np.array([dx, dx_dot, theta, theta_dot])


    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi).
        """
        angle = angle % (2*np.pi)
        if (angle > np.pi):
            angle -= 2*np.pi
        return angle

