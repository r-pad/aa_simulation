#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Miscellaneous utility functions.
"""

import numpy as np


def normalize_angle(angle):
    """
    Normalize angle from [-pi, pi].
    """
    angle = angle % (2*np.pi)
    if (angle >= np.pi):
        angle -= 2*np.pi
    return angle


def rotate_state(state, angle):
    """
    Rotate a state [x, y, yaw, x_dot, y_dot, yaw_dot] by a specified angle
    in radians.
    """
    if len(state) != 6:
        raise ValueError('Invalid state; dimension mismatch')

    x = state[0]
    y = state[1]
    yaw = state[2]
    x_dot = state[3]
    y_dot = state[4]
    yaw_dot = state[5]

    new_x = x*np.cos(angle) - y*np.sin(angle)
    new_y = y*np.cos(angle) + y*np.sin(angle)
    new_yaw = normalize_angle(yaw + angle)

    new_state = np.array([new_x, new_y, new_yaw, x_dot, y_dot, yaw_dot])
    return new_state


def translate_state(state, translation):
    """
    Translate a state [x, y, yaw, x_dot, y_dot, yaw_dot] by a translation
    [dx, dy].
    """
    if len(state) != 6:
        raise ValueError('Invalid state; dimension mismatch')
    if len(translation) != 2:
        raise ValueError('Invalid translation amount; dimension mismatch')

    state = np.array(state)
    state[0] += translation[0]
    state[1] += translation[1]
    return state
