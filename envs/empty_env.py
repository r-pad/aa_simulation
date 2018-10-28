#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Empty environment for test purposes.
"""

import numpy as np

from rllab.envs.base import Step
from rllab.spaces import Box

from car_simulation.envs.base import VehicleEnv


class EmptyEnv(VehicleEnv):
    """
    Simulation environment for an RC car roaming around in an empty
    environment for test purposes.
    """

    def __init__(self):
        """
        Initialize super class parameters.
        """
        super(EmptyEnv, self).__init__()


    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(6,))


    @property
    def get_initial_state(self):
        return np.zeros(6)


    def step(self, action):
        """
        Move one iteration forward in simulation.
        """
        self._action = action
        self._state = self._model.state_transition(self._state, action,
                self._dt)
        next_observation = np.copy(self._state)
        reward = 0
        done = False
        return Step(observation=next_observation, reward=reward,
                done=done)
