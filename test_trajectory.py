#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Run a simulation of a hardcoded trajectory on an arbitrary environment.
"""

import numpy as np

from rllab.envs.base import Env
from rllab.envs.normalized_env import normalize
from rllab.misc.console import query_yes_no
from rllab.misc.resolve import load_class


def main():
    env = normalize(load_class('car_simulation.envs.empty',
        Env, ["rllab", "envs"])())
    state = env.reset()
    env.render()

    t = 0
    max_t = 100

    while True:

        while t < max_t:
            if t < max_t / 2:
                action = np.array([1.0, np.deg2rad(15)])
            else:
                action = np.array([1.0, -np.deg2rad(15)])
            nextstate, reward, done, _ = env.step(action)
            env.render()
            t += 1

        if query_yes_no('Continue simulation?'):
            t = 0
        else:
            break


if __name__ == '__main__':
    main()
