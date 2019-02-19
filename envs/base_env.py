#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Simulation environment using vehicle model defined in Model.py.
"""

import yaml

import numpy as np

from rllab.envs.base import Env
from rllab.misc import logger
from rllab.spaces import Box

from aa_simulation.envs.model import VehicleModel
from aa_simulation.envs.renderer import _Renderer


class VehicleEnv(Env):
    """
    Simulation environment for an RC car traversing through
    obstacles specified by the user.
    """

    _MIN_VELOCITY = 0.0
    _MAX_VELOCITY = 1.3
    _MAX_STEER_ANGLE = np.pi / 6
    _HORIZON_LENGTH = 200


    def __init__(self, target_velocity):
        """
        Initialize environment parameters.
        """
        # Instantiate vehicle model and interpret parameters
        stream = open('aa_simulation/envs/model_params.yaml', 'r')
        self._params = yaml.load(stream)
        self._model = VehicleModel(self._params)
        self._action = None
        self._obstacles = []
        self._goal = None
        self.target_velocity = target_velocity

        # Time between each simulation iteration
        self._dt = 0.035

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
        self._state = self.get_initial_state
        observation = np.copy(self._state)

        # Reset renderer if available
        if self._renderer is not None:
            self._renderer.reset()

        return observation


    def step(self, action):
        """
        Move one iteration forward in simulation.
        """
        raise NotImplementedError


    def render(self):
        """
        Render simulation environment.
        """
        if self._renderer == None:
            self._renderer = _Renderer(self._params, self._obstacles,
                    self._goal, self.__class__.__name__)
        self._renderer.update(self._state, self._action)


    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on collected paths.
        """
        dists = []
        vels = []
        for path in paths:
            dists.append(path['env_infos']['dist'])
            vels.append(path['env_infos']['vel'])
        dists = np.abs(dists)
        vels = np.abs(vels)

        logger.record_tabular('AverageAbsDistanceError', np.mean(dists))
        logger.record_tabular('AverageAbsVelocityError', np.mean(vels))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        raise NotImplementedError


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

