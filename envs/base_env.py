#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Simulation environment using vehicle model defined in model.py.
"""

import yaml

import numpy as np

from rllab.envs.base import Env
from rllab.misc import logger
from rllab.spaces import Box

from aa_simulation.envs.model import BrushTireModel
from aa_simulation.envs.renderer import _Renderer


class VehicleEnv(Env):
    """
    Simulation environment for a FFAST RC car.
    """

    _MIN_VELOCITY = 0.0
    _MAX_VELOCITY = 10.0
    _MAX_STEER_ANGLE = np.pi / 6
    _HORIZON_LENGTH = 50


    def __init__(self, target_velocity, dt):
        """
        Initialize environment parameters.
        """
        # Instantiate vehicle model and interpret parameters
        stream = open('aa_simulation/envs/model_params.yaml', 'r')
        self._params = yaml.load(stream)
        self._model = BrushTireModel(self._params)
        self._action = None
        self.target_velocity = target_velocity

        # Time between each simulation iteration
        # Note: dt is measured to be 0.035, but we train with longer dt
        #       for more stability in commanded actions.
        self._dt = dt

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
            self._renderer = _Renderer(self._params,
                    self.__class__.__name__)
        self._renderer.update(self._state, self._action)


    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on collected paths.
        """
        log_kappa = False
        if self.__class__.__name__ == 'CircleEnv':
            log_kappa = True

        dists = []
        vels = []
        kappas = []
        for path in paths:
            dists.append(path['env_infos']['dist'])
            vels.append(path['env_infos']['vel'])
            if log_kappa:
                kappas.append(path['env_infos']['kappa'])
        dists = np.abs(dists)
        vels = np.abs(vels)
        if log_kappa:
            kappas = np.abs(kappas)

        logger.record_tabular('AverageAbsDistanceError', np.mean(dists))
        logger.record_tabular('AverageAbsVelocityError', np.mean(vels))
        logger.record_tabular('MaxAbsDistanceError', np.max(dists))
        logger.record_tabular('MaxAbsVelocityError', np.max(vels))
        if log_kappa:
            logger.record_tabular('AverageKappa', np.mean(kappas))
            logger.record_tabular('MaxKappa', np.max(kappas))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        raise NotImplementedError
