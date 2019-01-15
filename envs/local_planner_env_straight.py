#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Jiyuan Zhou

Environment for training local planner to move on a straight way.
"""

import csv

import numpy as np
import math

from rllab.envs.base import Step
from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv


class LocalPlannerEnvStraight(VehicleEnv):
    """
    Simulation environment for an RC car following a straight
    line trajectory using relative coordinates.
    The straight line trajectory starts from point [0, 0], and its
    direction is always right, namely yaw is 0.
    """

    def __init__(self, target_velocity):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(LocalPlannerEnvStraight, self).__init__(target_velocity)

        # Parameters of the line to follow
        # Always strating from (0, 0)
        self.init_x = 0
        self.init_y = 0

        # Fixed target position, only used for reward calculation.
        self.target_x = 1
        self.target_y = 0
        
    @property
    def observation_space(self):
        '''
        Redefine the shape of input vector of NN.
        '''
        return Box(low=-np.inf, high=np.inf, shape=(5,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        # Random initial position and yaw (from -90 to 90)
        y = np.random.random() * 0.3 - 0.15
        # For training, the model is always moving right, with direction 0.
        # Random initial direction, +- 90 degrees initial direction error
        self.init_dir = self._normalize_angle(np.deg2rad(180 * (np.random.random()) - 90))

        # Random initial velocity:
        # Able to sustain 0 - 3 times target velocity on x direction.
        # Able to -2 - 2 times target velocity on y direction.
        self.init_dx = 2 * np.random.random() * self.target_velocity
        self.init_dy = (3 * np.random.random() - 1.5) * self.target_velocity

        # Random initial dyaw. (small values only)
        # dyaw = np.random.random() * 0.01 - 0.005
        dyaw = 0
        

        state = np.zeros(6)
        state[1] = y
        state[2] = self.init_dir
        state[3] = self.init_dx
        state[4] = self.init_dy
        state[5] = dyaw
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
            x, y, _, dx, dy, _ = nextstate

            # Relative weights.
            lambda_vel = 0.2

            # Velocity difference
            velocity = np.sqrt(np.square(dx) + np.square(dy))
            vel_diff = velocity - self.target_velocity

            # Position difference
            distance = self._cal_distance(x, y)

            reward = -np.absolute(distance)
            reward -= lambda_vel * np.square(vel_diff)

        next_observation = self._modify_state(nextstate)        
        return Step(observation=next_observation, reward=reward,
                done=done, dist=distance, vel=vel_diff)

    def _cal_distance(self, x, y):
        '''
        Calculate the distance between current position
        and target straight trajectory.
        '''
        # print(y)
        return y

    def _modify_state(self, state):
        """
        Add target direction and target velocity to state, to feed
        in the NN.
        """
        x, y, yaw, dx, dy, dyaw = state
        yaw = self._normalize_angle(yaw)
        dyaw = self._normalize_angle(dyaw)
        return np.array([y, yaw, dx, dy, dyaw])


    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = None
        self._state = self.get_initial_state
        observation = self._modify_state(self._state)

        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation

    def _normalize_angle(self, angle):
        """
        Normalize angle to [-pi, pi).
        """
        angle = angle % (2*np.pi)
        if (angle >= np.pi):
            angle -= 2*np.pi
        return angle
