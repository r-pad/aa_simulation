#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Safety constraint for following circle while never driving away from the
circle
"""

import numpy as np

from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import SafetyConstraint


class CircleSafetyConstraint(SafetyConstraint, Serializable):
    """
    Always drive towards the circle
    """

    def __init__(self, max_value=1.0, **kwargs):
        """
        :param max_value: Upper threshold for constraint return
        """
        self.max_value = max_value
        Serializable.quick_init(self, locals())
        super(CircleSafetyConstraint, self).__init__(max_value, **kwargs)


    def evaluate(self, path):
        """
        Return True if constraint is violated.
        """
        observations = path['observations']
        actions = path['actions']
        dx = observations[:, 0]
        theta = observations[:, 1]
        steer = actions[:, 1]

        # Positive if driving away from circle, negative otherwise
        driving_away = -steer * (np.sign(theta) + np.sign(dx))
        return driving_away > 0

