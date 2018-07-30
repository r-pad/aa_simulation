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
    max_t = 50

    while True:

        while t < max_t:

            # Hardcoded trajectory
            if t == 5:
                action = np.array([1.0, 0.0873])
            elif t > 15 and t < 30:
                action = np.array([1.0, -0.0873])
            else:
                action = np.array([1.0, 0])

            nextstate, reward, done, _ = env.step(action)
            env.render()
            t += 1

        if query_yes_no('Continue simulation?'):
            t = 0
        else:
            break


if __name__ == '__main__':
    main()
