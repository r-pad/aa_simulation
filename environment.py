#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Simulation environment using vehicle model defined in Model.py.
"""

import csv
import yaml

import numpy as np

from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step

from car_simulation.model import VehicleModel
from car_simulation.renderer import _Renderer


class VehicleEnv(Env):
    """
    Simulation environment for an RC car traversing through
    obstacles specified by the user.
    """

    _MIN_VELOCITY = 0.0
    _MAX_VELOCITY = 4.0
    _MAX_STEER_ANGLE = np.deg2rad(45)
    _HORIZON_LENGTH = 200


    def __init__(self):
        """
        Initialize environment parameters.
        """
        # Instantiate vehicle model and interpret parameters
        stream = open('car_simulation/params.yaml', 'r')
        self._params = yaml.load(stream)
        self._model = VehicleModel(self._params)
        self._action = None

        # Time between each simulation iteration
        self._dt = 0.02

        # Get obstacles from CSV file
        #   Convention: (x, y, r) for each obstacle, which is a circle
        with open('car_simulation/obstacles.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            values = list(reader)
            obstacle_list = [[float(x) for x in row] for row in values]
            self._obstacles = np.array(obstacle_list)

        # Goal location
        #   Convention: (x, y, r)
        with open('car_simulation/goal.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            values = list(reader)[0]
            self._goal = [float(x) for x in values]

        # Instantiates object handling simulation renderings
        self._renderer = None


    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(6,))


    @property
    def action_space(self):
        low = np.array([VehicleEnv._MIN_VELOCITY,
            -VehicleEnv._MAX_STEER_ANGLE])
        high = np.array([VehicleEnv._MAX_VELOCITY,
            VehicleEnv._MAX_STEER_ANGLE])
        return Box(low=low, high=high)


    @property
    def horizon(self):
        return VehicleEnv._HORIZON_LENGTH


    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = None
        self._state = np.zeros(6)
        observation = np.copy(self._state)

        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation


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
            reward = -20
            done = True
        else:
            self._state = nextstate
            if self._reached_goal(nextstate):
                reward = 0
                done = True
            else:
                location = nextstate[0:2]
                goal = self._goal[0:2]
                reward = -np.linalg.norm(location-goal)
                done = False

        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward,
                done=done)


    def render(self):
        """
        Render simulation environment.
        """
        print('current state:', self._state)
        if self._renderer == None:
            self._renderer = _Renderer(self._params, self._obstacles,
                    self._goal)
        self._renderer.update(self._state, self._action)


    def _check_collision(self, state):
        """
        Check if state collides with any of the obstacles.
        """
        point = state[0:2]

        if len(self._obstacles) == 0:
            return False

        for i in range(len(self._obstacles)):
            obstacle = self._obstacles[i]
            center = obstacle[0:2]
            radius = obstacle[2]
            if np.linalg.norm(center-point) < radius:
                return True

        return False


    def _reached_goal(self, state):
        """
        Check if state is at goal position, within an epsilon.
        """
        point = state[0:2]
        goal = self._goal[0:2]
        if np.linalg.norm(goal-point) < self._goal[2]:
            return True
        return False

