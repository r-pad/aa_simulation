#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for driving car in circular arc trajectory.
"""

import csv

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from car_simulation.envs.base import VehicleEnv


class ArcEnv(VehicleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory.
    """

    def __init__(self):
        """
        Initialize super class parameters, obstacles and goal.
        """
        super(ArcEnv, self).__init__()

        # Get obstacles from CSV file
        #   Convention: (x, y, r) for each obstacle, which is a circle
        filename = 'car_simulation/envs/arc/obstacles.csv'
        with open(filename, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            values = list(reader)
            obstacle_list = [[float(x) for x in row] for row in values]
            self._obstacles = np.array(obstacle_list)

        # Goal location
        #   Convention: (x, y, r)
        filename = 'car_simulation/envs/arc/goal.csv'
        with open(filename, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            values = list(reader)[0]
            self._goal = [float(x) for x in values]


    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(6,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        state = np.zeros(6)
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
            if self._reached_goal(nextstate):
                reward = 0
                done = True
            else:
                location = nextstate[0:2]
                goal = self._goal[0:2]
                done = False

                # Trajectory following
                r = goal[0] / 2
                x, y, _, x_dot, y_dot, _ = nextstate
                target_velocity = 0.7
                lambda1 = 0.25
                velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
                distance = np.abs(y)
                reward = -distance
                reward -= lambda1 * np.square(velocity - target_velocity)

        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward,
                done=done)


    def _reached_goal(self, state):
        """
        Check if state is at goal position, within an epsilon.
        """
        point = state[0:2]
        goal = self._goal[0:2]
        if np.linalg.norm(goal-point) < self._goal[2]:
            return True
        return False

