#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Class simulating running vehicle model defined in Model.py in an
environment.
"""

import argparse
import csv
import yaml

import evdev
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

    # Joystick defines
    JOYSTICK_THROTTLE_KEY = 304
    JOYSTICK_BRAKE_KEY = 307
    JOYSTICK_STEER_KEY = 16
    JOYSTICK_LEFT_VAL = -1
    JOYSTICK_RIGHT_VAL = 1
    JOYSTICK_SPEED_AMT = 0.5
    JOYSTICK_STEER_AMT = np.deg2rad(15)
    JOYSTICK_LEFT_THRESH = np.deg2rad(45)
    JOYSTICK_RIGHT_THRESH = np.deg2rad(-45)


    def __init__(self, simulation_mode, show_visuals):
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
        self.show_visuals = show_visuals

        # Initialize state and controls
        self.mode = simulation_mode
        state = np.zeros(6)
        state[0] = 0
        state[1] = 0
        state[2] = 0
        state[3] = 1
        state[4] = 0.216854
        state[5] = -0.772949
        self.X = state

        if self.mode == CarSimulation.CONSTANT_INPUT:
            cmd_vel = 4.253134
            steer = np.deg2rad(15)
            control = np.zeros(2)
            control[0] = cmd_vel
            control[1] = steer
            self.U = control
            self.dt = 0.02
        elif self.mode == CarSimulation.JOYSTICK_INPUT:
            self.device = evdev.InputDevice('/dev/input/event5')
            cmd_vel = 0.5
            steer = 0
            control = np.zeros(2)
            control[0] = cmd_vel
            control[1] = steer
            self.U = control
            self.dt = 0.02
        else:
            raise ValueError('Invalid simulation mode')

        # Get obstacles from CSV file
        #   Convention: (x, y, r) for each obstacle, which is a circle
        with open('obstacles.csv', 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            values = list(reader)
            obstacle_list = [[float(x) for x in row] for row in values]
            self.obstacles = np.array(obstacle_list)


    def run(self):
        """
        Update visualization using new states computed from model.
        """
        x = [self.X[0]]
        y = [self.X[1]]

        # Create visualization
        if self.show_visuals:
            plt.ion()
            fig = plt.figure()
            plt.axis('scaled')
            ax = fig.add_subplot(111)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            trajectory, = ax.plot(x, y, 'b-')
            for i in xrange(len(self.obstacles)):
                obstacle = self.obstacles[i]
                x_ = obstacle[0]
                y_ = obstacle[1]
                r_ = obstacle[2]
                circle = plt.Circle((x_, y_), r_, fill=False)
                ax.add_artist(circle)

        while True:

            # Get new control inputs if available
            if self.mode == CarSimulation.JOYSTICK_INPUT:
                events = self.device.read()
                try:
                    for event in events:
                        if event.code == CarSimulation.JOYSTICK_THROTTLE_KEY:
                            if event.value != 1:
                                self.U[0] += CarSimulation.JOYSTICK_SPEED_AMT
                        elif event.code == CarSimulation.JOYSTICK_BRAKE_KEY:
                            if event.value != 1 and self.U[0] > 0:
                                self.U[0] -= CarSimulation.JOYSTICK_SPEED_AMT
                        elif event.code == CarSimulation.JOYSTICK_STEER_KEY:
                            if event.value == -1 and \
                                    self.U[1] < CarSimulation.JOYSTICK_LEFT_THRESH:
                                self.U[1] += CarSimulation.JOYSTICK_STEER_AMT
                            elif event.value == 1 and \
                                    self.U[1] > CarSimulation.JOYSTICK_RIGHT_THRESH:
                                self.U[1] -= CarSimulation.JOYSTICK_STEER_AMT
                        break
                except IOError:
                    pass

            # Get new state from state and control inputs
            new_state = self.model.state_transition(self.X, self.U, self.dt)

            # Check collision against obstacle
            collision = self._check_collision(new_state)

            # Append new state to trajectory
            if not collision:
                self.X = new_state
                x.append(self.X[0])
                y.append(self.X[1])
            else:
                continue

            # Draw new state onto existing trajectory
            if self.show_visuals:
                trajectory.set_xdata(x)
                trajectory.set_ydata(y)
                self._draw_car(ax, self.X[0], self.X[1], self.X[2])
                fig.canvas.draw()


    def _check_collision(self, state):
        """
        Check if state collides with any of the obstacles.
        """
        x = state[0]
        y = state[1]
        point = state[0:2]
        for i in xrange(len(self.obstacles)):
            obstacle = self.obstacles[i]
            center = obstacle[0:2]
            radius = obstacle[2]
            if np.linalg.norm(center-point) < radius:
                return True

        return False


    def _draw_car(self, ax, x, y, yaw):
        """
        Draw car on simulation.
        """
        # Initialize coordinates and some transformations
        if self.car is None:
            tw2 = self.tw / 2
            self.car_coords = np.array([
                [-self.L_r, -self.L_r, self.L_f, self.L_f, -self.L_r],
                [-tw2, tw2, tw2, -tw2, -tw2],
                [1, 1, 1, 1, 1]])
            self.wheel_coords = np.array([
                [-self.wheel_dia, -self.wheel_dia, self.wheel_dia,
                    self.wheel_dia, -self.wheel_dia],
                [-self.wheel_w, self.wheel_w, self.wheel_w,
                    -self.wheel_w, -self.wheel_w],
                [1, 1, 1, 1, 1]])
            self.wheel_fr_tf = np.array([
                [1, 0, self.L_f],
                [0, 1, -self.tw/2],
                [0, 0, 1]])
            self.wheel_fl_tf = np.array([
                [1, 0, self.L_f],
                [0, 1, self.tw/2],
                [0, 0, 1]])
            self.wheel_rr_tf = np.dot(np.array([
                [1, 0, -self.L_f],
                [0, 1, -self.tw/2],
                [0, 0, 1]]), self.wheel_coords)
            self.wheel_rl_tf = np.dot(np.array([
                [1, 0, -self.L_f],
                [0, 1, self.tw/2],
                [0, 0, 1]]), self.wheel_coords)

        # Remove previous patches
        else:
            self.car.remove()
            self.wheel_fr.remove()
            self.wheel_fl.remove()
            self.wheel_rr.remove()
            self.wheel_rl.remove()

        # Get coordinate transformations
        steer = self.U[1]
        pos_tf = np.array([
            [np.cos(yaw), -np.sin(yaw), x],
            [np.sin(yaw), np.cos(yaw), y],
            [0, 0, 1]])
        pos_steer = np.array([
            [np.cos(steer), -np.sin(steer), 0],
            [np.sin(steer), np.cos(steer), 0],
            [0, 0, 1]])

        # Apply coordinate transform to chassis and wheels
        pos_body = np.dot(pos_tf, self.car_coords)
        pos_wheel_fr = reduce(np.dot, [pos_tf, self.wheel_fr_tf,
            pos_steer, self.wheel_coords])
        pos_wheel_fl = reduce(np.dot, [pos_tf, self.wheel_fl_tf,
            pos_steer, self.wheel_coords])
        pos_wheel_rr = np.dot(pos_tf, self.wheel_rr_tf)
        pos_wheel_rl = np.dot(pos_tf, self.wheel_rl_tf)

        # Draw car onto screen using polygon patches
        self.car = patches.Polygon(pos_body[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self.wheel_fr = patches.Polygon(pos_wheel_fr[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self.wheel_fl = patches.Polygon(pos_wheel_fl[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self.wheel_rr = patches.Polygon(pos_wheel_rr[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        self.wheel_rl = patches.Polygon(pos_wheel_rl[:2].T, linewidth=1,
                edgecolor='r', facecolor='none')
        ax.add_patch(self.car)
        ax.add_patch(self.wheel_fr)
        ax.add_patch(self.wheel_fl)
        ax.add_patch(self.wheel_rr)
        ax.add_patch(self.wheel_rl)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Show visualizations
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--show',
            dest='show_visuals', action='store_true',
            help='Whether to visually show the simulation.')
    parser_group.add_argument('--no-show',
            dest='show_visuals', action='store_false',
            help='Whether to visually show the simulation.')
    parser.set_defaults(show_visuals=True)

    # Simulation mode
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--constant',
            dest='mode', action='store_const',
            const=CarSimulation.CONSTANT_INPUT,
            help='What type of simulation mode to run.')
    parser_group.add_argument('--joystick',
            dest='mode', action='store_const',
            const=CarSimulation.JOYSTICK_INPUT,
            help='What type of simulation mode to run.')
    parser.set_defaults(mode=CarSimulation.CONSTANT_INPUT)

    return parser.parse_args()


def main():
    args = parse_arguments()
    show_visuals = args.show_visuals
    simulation_mode = args.mode
    simulation = CarSimulation(simulation_mode, show_visuals)
    simulation.run()


if __name__ == '__main__':
    main()

