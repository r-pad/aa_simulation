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

from rllab.sampler.utils import rollout


def profile_code(profiler):
    """
    Use cProfile to profile code, listing functions with most
    cumulative time spent.
    """
    print('\n')
    ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
    ps.print_stats(10)


def plot_metrics(error, name, units):
    """
    Plot histogram of error.
    """
    mean = error.mean()
    std = error.std()
    maximum = error.max()
    minimum = error.min()
    stats = 'Mean = %.3f\nStd = %.3f\nMax = %.3f\nMin = %.3f' % \
            (mean, std, maximum, minimum)
    title = 'Distribution of %s Errors in Final Policy' % name

    plt.figure()
    plt.hist(error)
    plt.title(title)
    plt.xlabel('Error (%s)' % units)
    plt.ylabel('Number of Time Steps')
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.87, 0.9, stats, ha='center', va='center',
            transform=plt.gca().transAxes)
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=100000,
                        help='Speedup')
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
    plt.ion()

    # Sample one rollout
    profiler.enable()
    path = rollout(env, policy, max_path_length=args.max_path_length,
                        animated=args.render, speedup=args.speedup,
                        always_return_paths=True)
    profiler.disable()

    # Policy analysis
    profile_code(profiler)
    plot_metrics(path['env_infos']['dist'], 'Distance', 'm')
    plot_metrics(path['env_infos']['vel'], 'Velocity', 'm/s')

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == "__main__":
    main()

