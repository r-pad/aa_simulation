#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Safety constraint for following circle within epsilon distance away.
"""

from rllab.core.serializable import Serializable
from sandbox.cpo.safety_constraints.base import SafetyConstraint


class CircleSafetyConstraint(SafetyConstraint, Serializable):
    """
    Stay within epsilon distance from circular trajectory.
    """

    def __init__(self, max_value=1.0, eps=1.5, **kwargs):
        """
        :param max_value: Upper threshold for constraint return
        :param eps: Stay at most epsilon distance away from circle
        """
        super(CircleSafetyConstraint, self).__init__(max_value, **kwargs)
        self.max_value = max_value
        self.eps = eps


    def evaluate(self, path):
        """
        Return True if constraint is violated.
        """
        observations = path['observations']
        dx = observations[:, 0]
        return dx >= self.eps
