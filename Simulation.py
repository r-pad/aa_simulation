#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn
"""

import yaml

import numpy as np
import matplotlib.pyplot as plt

import Model
from Model import VehicleModel


class CarSimulation(object):
    """
    Object that controls visualization of simulation.
    """

    def __init__(self):
        """
        Initialize simulation parameters.
        """
        stream = file('params.yaml', 'r')
        params = yaml.load(stream)
        self.model = VehicleModel(params)


    def update(self, X):
        """
        Update visualization using new state.
        """
        print self.model.state_transition(np.zeros(6), np.zeros(2))


if __name__ == '__main__':
    simulation = CarSimulation()
    simulation.update(1)
