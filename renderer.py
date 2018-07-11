#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Simulation environment renderer.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as tf


class _Renderer(object):
    """
    Renders the RC car and obstacles contained in the simulation.
    """

    def __init__(self, params, obstacles):
        """
        Initialize simulation.
        """
        # Simulation parameters
        self._params = params
        self._obstacles = obstacles

        # Saved car trajectory
        self._x = []
        self._y = []

        # Create visualization
        plt.ion()
        self._fig = plt.figure()
        window_size = 10
        self._ax = self._fig.add_subplot(111)
        self._ax.set_aspect('equal')
        self._ax.set_xlim(-window_size, window_size)
        self._ax.set_ylim(-window_size, window_size)
        self._trajectory, = self._ax.plot(self._x, self._y, 'b-')
        for i in range(len(self._obstacles)):
            obstacle = self._obstacles[i]
            obstacle_x = obstacle[0]
            obstacle_y = obstacle[1]
            obstacle_r = obstacle[2]
            circle = plt.Circle((obstacle_x, obstacle_y), obstacle_r,
                    fill=False)
            self._ax.add_artist(circle)


    def update(self, state):
        """
        Update visualization to show new state.
        """
        self._x.append(state[0])
        self._y.append(state[1])
        self._trajectory.set_xdata(self._x)
        self._trajectory.set_ydata(self._y)
        #self._draw_car(state[0], state[1], state[2])
        self._fig.canvas.draw()
        plt.pause(0.001)


    #def _draw_car(self, x, y, yaw)
