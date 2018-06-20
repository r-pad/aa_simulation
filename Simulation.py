#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn
"""

import yaml

import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt

import Model
from Model import VehicleModel


class CarSimulation(object):
    """
    Object that controls visualization of simulation.
    """

    # Simulation modes
    MODE_CONSTANT_INPUT = 1      # Constant control input to system


    def __init__(self, simulation_mode):
        """
        Initialize simulation parameters.
        """
        # Instantiate vehicle model
        stream = file('params.yaml', 'r')
        params = yaml.load(stream)
        self.model = VehicleModel(params)

        # Initialize state and controls
        if simulation_mode == CarSimulation.MODE_CONSTANT_INPUT:
            state = np.zeros(6)
            state[0] = 0
            state[1] = 0
            state[2] = 0
            state[3] = 1
            state[4] = 0.216854
            state[5] = -0.772949
            self.X = state
            cmd_vel = 4.253134
            steer = np.deg2rad(15)
            control = np.zeros(2)
            control[0] = cmd_vel
            control[1] = steer
            self.U = control
            self.dt = 0.02
        else:
            raise ValueError('Invalid simulation mode')


    def run(self):
        """
        Update visualization using new states computed from model.
        """
        x = [self.X[0]]
        y = [self.X[1]]

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        line1, = ax.plot(x, y, 'b-')

        while True:
            self.X = self.model.state_transition(self.X, self.U, self.dt)
            x.append(self.X[0])
            y.append(self.X[1])
            line1.set_xdata(x)
            line1.set_ydata(y)
            fig.canvas.draw()


if __name__ == '__main__':
    simulation = CarSimulation(CarSimulation.MODE_CONSTANT_INPUT)
    simulation.run()

