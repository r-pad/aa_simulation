#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Simulation environment renderer.
"""

import functools as ft

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as tf
import numpy as np


class _Renderer(object):
    """
    Renders the RC car and obstacles contained in the simulation.
    """

    def __init__(self, params, obstacles, goal, env_type):
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
        self._ax = self._fig.add_subplot(111)
        self._ax.set_aspect('equal')
        if env_type == 'EmptyEnv':
            self._ax.set_xlim(-1, 4)
            self._ax.set_ylim(-2, 2)
        elif env_type == 'ArcRelativeEnv' or env_type == 'ArcRelativeEnv2':
            self._ax.set_xlim(-2.5, 2.5)
            self._ax.set_ylim(-2.5, 2.5)
        else:
            self._ax.set_xlim(-1, 4)
            self._ax.set_ylim(-2, 2)

        # Show ideal trajectory
        if env_type == 'StraightEnv':
            line = plt.Line2D([0, goal[0]], [0, goal[1]],
                    color='c', ls=':')
            self._ax.add_line(line)
        elif env_type == 'ArcEnv':
            arc = patches.Arc((goal[0]/2, 0), goal[0], goal[0],
                    0, 180, 360, color='c', ls=':')
            self._ax.add_patch(arc)
        elif env_type == 'ArcRelativeEnv' or env_type == 'ArcRelativeEnv2':
            arc = patches.Arc((0, 0), 3, 3,
                    0, 0, 360, color='c', ls=':')
            self._ax.add_patch(arc)

        # Draw remaining simulation
        self._trajectory, = self._ax.plot(self._x, self._y, 'b-')
        if len(self._obstacles) != 0:
            for i in range(len(self._obstacles)):
                obstacle = self._obstacles[i]
                obstacle_x = obstacle[0]
                obstacle_y = obstacle[1]
                obstacle_r = obstacle[2]
                circle = plt.Circle((obstacle_x, obstacle_y),
                        obstacle_r, color='0.5')
                self._ax.add_artist(circle)
        if goal != None:
            circle = plt.Circle((goal[0], goal[1]), goal[2], fill=False)
            self._ax.add_artist(circle)

        # Indicates whether display has been created
        self._car = None


    def update(self, state, action):
        """
        Update visualization to show new state.
        """
        if action is not None:
            steer = action[1]
        else:
            steer = 0
        pos_x = state[0]
        pos_y = state[1]
        pos_yaw = state[2]
        self._x.append(pos_x)
        self._y.append(pos_y)
        self._trajectory.set_xdata(self._x)
        self._trajectory.set_ydata(self._y)
        self._draw_car(pos_x, pos_y, pos_yaw, steer)
        self._fig.canvas.draw()
        plt.pause(0.001)


    def reset(self):
        """
        Reset visualization, removing any visible trajectories.
        """
        self._x = []
        self._y = []
        self._trajectory.set_xdata(self._x)
        self._trajectory.set_ydata(self._y)


    def _draw_car(self, x, y, yaw, steer):
        """
        Draw car on simulation.
        """
        # Run when drawing for the first time
        if self._car is None:
            self._initialize_transforms()

        # Remove previous patches
        else:
            self._car.remove()
            self._wheel_fr.remove()
            self._wheel_fl.remove()
            self._wheel_rr.remove()
            self._wheel_rl.remove()

        # Get coordinate transformations
        pos_tf = np.array([
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]])
        pos_steer = np.array([
            [np.cos(steer), -np.sin(steer), 0],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]])
        pos_steer = np.array([
            [np.cos(steer), -np.sin(steer), 0],
            [np.sin(steer), np.cos(steer), 0],
            [0, 0, 1]])

        # Apply coordinate transform to chassis and wheels
        pos_body = np.dot(pos_tf, self._car_coords)
        pos_wheel_fr = ft.reduce(np.dot, [pos_tf, self._wheel_fr_tf,
            pos_steer, self._wheel_coords])
        pos_wheel_fl = ft.reduce(np.dot, [pos_tf, self._wheel_fl_tf,
            pos_steer, self._wheel_coords])
        pos_wheel_rr = np.dot(pos_tf, self._wheel_rr_tf)
        pos_wheel_rl = np.dot(pos_tf, self._wheel_rl_tf)

        # Draw car onto screen using polygon patches
        self._car = patches.Polygon(pos_body[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self._wheel_fr = patches.Polygon(pos_wheel_fr[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self._wheel_fl = patches.Polygon(pos_wheel_fl[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self._wheel_rr = patches.Polygon(pos_wheel_rr[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self._wheel_rl = patches.Polygon(pos_wheel_rl[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self._ax.add_patch(self._car)
        self._ax.add_patch(self._wheel_fr)
        self._ax.add_patch(self._wheel_fl)
        self._ax.add_patch(self._wheel_rr)
        self._ax.add_patch(self._wheel_rl)


    def _initialize_transforms(self):
        """
        Initialize transforms necessary to draw car on simulation.
        """
        L_f = self._params['L_f']
        L_r = self._params['L_r']
        tw = self._params['tw']
        wheel_dia = self._params['wheel_dia']
        wheel_w = self._params['wheel_w']
        tw2 = tw / 2

        self._car_coords = np.array([
            [-L_r, -L_r, L_f, L_f, -L_r],
            [-tw2, tw2, tw2, -tw2, -tw2],
            [1, 1, 1, 1, 1]])
        self._wheel_coords = np.array([
            [-wheel_dia, -wheel_dia, wheel_dia, wheel_dia, -wheel_dia],
            [-wheel_w, wheel_w, wheel_w, -wheel_w, -wheel_w],
            [1, 1, 1, 1, 1]])
        self._wheel_fr_tf = np.array([
            [1, 0, L_f],
            [0, 1, -tw2],
            [0, 0, 1]])
        self._wheel_fl_tf = np.array([
            [1, 0, L_f],
            [0, 1, tw2],
            [0, 0, 1]])
        self._wheel_rr_tf = np.dot(np.array([
            [1, 0, -L_f],
            [0, 1, -tw2],
            [0, 0, 1]]), self._wheel_coords)
        self._wheel_rl_tf = np.dot(np.array([
            [1, 0, -L_f],
            [0, 1, tw2],
            [0, 0, 1]]), self._wheel_coords)

