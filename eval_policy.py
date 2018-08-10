#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Evaluate a policy and publish metrics.
"""

import argparse
import cProfile
import pstats
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np

from rllab.envs.base import Env
from rllab.misc.resolve import load_class


def profile_code(profiler):
    print('\n')
    ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
    ps.print_stats(10)


def plot_metrics():
    pass


def rollout(env, agent, max_path_length, animated):
    """
    Based on rollout() in rllab.misc.utils, with metrics.
    """
    state = env.reset()
    env.record_metrics()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        action, _ = agent.get_action(state)
        nextstate, reward, done, _ = env.step(action)
        if done:
            break
        path_length += 1
        state = nextstate
        if animated:
            env.render()
    env.save_metrics()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--render', dest='render',
            action='store_true', help='Rendering')
    parser.add_argument('--no-render', dest='render',
            action='store_false', help='Rendering')
    parser.set_defaults(render=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    profiler = cProfile.Profile()
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']

    # Sample one rollout
    profiler.enable()
    path = rollout(env, policy, max_path_length=args.max_path_length,
                        animated=args.render)
    profiler.disable()
    if args.render:
        sys.stdout.write("Press <enter> to continue: ")
        input()

    # Policy analysis
    profile_code(profiler)
    plot_metrics()


if __name__ == "__main__":
    main()

