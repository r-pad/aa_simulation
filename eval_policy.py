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
    print('\n')
    ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
    ps.print_stats(10)


def plot_metrics(env_infos):
    dist = env_infos['dist']
    vel = env_infos['vel']

    # Plot histogram of distance error
    plt.figure()
    plt.hist(dist)
    plt.title('Distribution of Distance Errors in Final Policy')
    plt.xlabel('Error (m)')
    plt.ylabel('Number of Time Steps')

    # Plot histogram of velocity error
    plt.figure()
    plt.hist(vel)
    plt.title('Distribution of Velocity Errors in Final Policy')
    plt.xlabel('Error (m/s)')
    plt.ylabel('Number of Time Steps')

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

    # Sample one rollout
    profiler.enable()
    path = rollout(env, policy, max_path_length=args.max_path_length,
                        animated=args.render, speedup=args.speedup,
                        always_return_paths=True)
    profiler.disable()

    # Policy analysis
    profile_code(profiler)
    plot_metrics(path['env_infos'])

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == "__main__":
    main()

