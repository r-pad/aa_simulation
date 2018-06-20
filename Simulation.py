#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Issues to fix:
    - dt calculation
    - Joystick input
"""

import yaml

import numpy as np
import matplotlib.animation as anim
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as tf

import Model
from Model import VehicleModel


class CarSimulation(object):
    """
    Object that controls visualization of simulation.
    """

    # Simulation modes
    CONSTANT_INPUT = 1          # Constant control input to system
    JOYSTICK_INPUT = 2          # Joystick control input to system


    def __init__(self, simulation_mode):
        """
        Initialize simulation parameters.
        """
        # Instantiate vehicle model and interpret parameters
        stream = file('params.yaml', 'r')
        params = yaml.load(stream)
        self.model = VehicleModel(params)
        self.L_f = params['L_f']
        self.L_r = params['L_r']
        self.tw = params['tw']
        self.wheel_dia = params['wheel_dia']
        self.wheel_w = params['wheel_w']

        # Graphics specific variables
        self.car = None

        # Initialize state and controls
        if simulation_mode == CarSimulation.CONSTANT_INPUT:
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
        elif simulation_mode == CarSimulation.JOYSTICK_INPUT:
            raise ValueError('Joystick mode not implemented yet')
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
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        trajectory, = ax.plot(x, y, 'b-')

        while True:
            self.X = self.model.state_transition(self.X, self.U, self.dt)
            x.append(self.X[0])
            y.append(self.X[1])
            trajectory.set_xdata(x)
            trajectory.set_ydata(y)
            self._draw_car(ax, self.X[0], self.X[1], self.X[2])
            fig.canvas.draw()


    def _draw_car(self, ax, x, y, yaw):
        """
        Draw car on simulation.
        """
        if self.car is None:
            tw2 = self.tw / 2
            self.car_coords = np.array([
                [-self.L_r, -self.L_r, self.L_f, self.L_f, -self.L_r],
                [-tw2, tw2, tw2, -tw2, -tw2],
                [1, 1, 1, 1, 1]])
        else:
            self.car.remove()

        # Apply coordinate transform to car
        pos_tf = np.array([
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]])
        pos_body = np.dot(pos_tf, self.car_coords)

        # Draw car onto screen using polygon patches
        self.car = patches.Polygon(pos_body[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        ax.add_patch(self.car)


if __name__ == '__main__':
    simulation = CarSimulation(CarSimulation.CONSTANT_INPUT)
    simulation.run()

