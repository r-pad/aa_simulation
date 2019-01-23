#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Edward Ahn

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
