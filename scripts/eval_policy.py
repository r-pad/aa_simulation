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

from aa_simulation.envs.base_env import VehicleEnv


def profile_code(profiler):
    """
    Use cProfile to profile code, listing functions with most
    cumulative time spent.
    """
    print('\n')
    ps = pstats.Stats(profiler).strip_dirs().sort_stats('cumulative')
    ps.print_stats(10)


def plot_curve(data, name, units):
    """
    Plot data over time.
    """
    mean = data.mean()
    std = data.std()
    maximum = data.max()
    minimum = data.min()
    stats = 'Mean = %.5f\nStd = %.5f\nMax = %.5f\nMin = %.5f' % \
            (mean, std, maximum, minimum)
    title = '%s over Time in Final Policy' % name

    plt.figure()
    t = np.arange(data.size)
    plt.plot(t, data)
    plt.title(title)
    plt.xlabel('Time steps')
    plt.ylabel('%s (%s)' % (name, units))
    plt.axhline(mean, color='k', linestyle='dashed', linewidth=1)
    plt.axhline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axhline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.87, 0.9, stats, ha='center', va='center',
            transform=plt.gca().transAxes)


def plot_distribution(data, name, units):
    """
    Plot histogram showing distribution of data.
    """
    mean = data.mean()
    std = data.std()
    maximum = data.max()
    minimum = data.min()
    stats = 'Mean = %.5f\nStd = %.5f\nMax = %.5f\nMin = %.5f' % \
            (mean, std, maximum, minimum)
    title = 'Distribution of %s in Final Policy' % name

    plt.figure()
    plt.hist(data)
    plt.title(title)
    plt.xlabel('Error (%s)' % units)
    plt.ylabel('Number of Time Steps')
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(mean+std, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(mean-std, color='r', linestyle='dashed', linewidth=1)
    plt.text(0.87, 0.9, stats, ha='center', va='center',
            transform=plt.gca().transAxes)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='Path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--seed', type=int, default=9, help='Random seed')
    parser.add_argument('--speedup', type=float, default=100000,
                        help='Speedup')
    parser.add_argument('--skip', type=int, default=0,
                        help='Number of iterations to skip at start')
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
    skip = args.skip
    policy = data['policy']
    env = data['env']
    plt.ion()

    # Set fixed random seed
    np.random.seed(args.seed)

    # Sample one rollout
    profiler.enable()
    path = rollout(env, policy, max_path_length=args.max_path_length,
                        animated=args.render, speedup=args.speedup,
                        always_return_paths=True)
    profiler.disable()

    # Policy analysis
    profile_code(profiler)
    actions = path['actions']
    plot_curve(actions[:, 0][skip:], 'Commanded Speed', 'm/s')
    plot_curve(actions[:, 1][skip:], 'Commanded Steering Angle', 'rad')
    plot_curve(path['env_infos']['kappa'][skip:], 'Wheel Slip', 'kappa')
    plot_curve(path['env_infos']['dist'][skip:], 'Distance', 'm')
    plot_curve(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')
    plot_distribution(path['env_infos']['dist'][skip:], 'Distance', 'm')
    plot_distribution(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')
    plt.show()

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == "__main__":
    main()

