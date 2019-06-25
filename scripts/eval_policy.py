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

# Toggle option for displaying plots
show_plots = True


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
    print('Mean %s: %.5f' % (name, mean))

    if not show_plots:
        return

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

    if not show_plots:
        return

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
    parser.add_argument('--num_paths', type=int, default=1,
                        help='Number of rollouts to collect and evaluate')
    parser.add_argument('--render', dest='render',
            action='store_true', help='Rendering')
    parser.add_argument('--no-render', dest='render',
            action='store_false', help='Rendering')
    parser.set_defaults(render=False)
    parser.add_argument('--plots', dest='show_plots',
            action='store_true', help='Show plots')
    parser.add_argument('--no-plots', dest='show_plots',
            action='store_false', help='Show plots')
    parser.set_defaults(plots=True)
    parser.add_argument('--profile', dest='profile_code',
            action='store_true', help='Profile code that samples a rollout')
    parser.add_argument('--no-profile', dest='profile_code',
            action='store_false', help='Profile code that samples a rollout')
    parser.set_defaults(profile_code=False)
    args = parser.parse_args()
    return args


def main():
    global show_plots
    args = parse_arguments()
    profiler = cProfile.Profile()
    data = joblib.load(args.file)
    skip = args.skip
    policy = data['policy']
    env = data['env']
    env._dt = 0.035                 # Set dt to empirically measured dt
    np.random.seed(args.seed)
    show_plots = args.show_plots

    for run in range(args.num_paths):

        profiler.enable()
        path = rollout(env, policy, max_path_length=args.max_path_length,
                            animated=args.render, speedup=args.speedup,
                            always_return_paths=True)
        profiler.disable()
        if args.profile_code:
            profile_code(profiler)

        # Analyze rollout
        actions = path['actions']
        plot_curve(actions[:, 0][skip:], 'Commanded Speed', 'm/s')
        plot_curve(actions[:, 1][skip:], 'Commanded Steering Angle', 'rad')
        plot_curve(path['env_infos']['kappa'][skip:], 'Wheel Slip', 'kappa')
        plot_curve(path['env_infos']['dist'][skip:], 'Distance', 'm')
        plot_curve(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')
        plot_distribution(path['env_infos']['dist'][skip:], 'Distance', 'm')
        plot_distribution(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')
        print()

    plt.show()

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == "__main__":
    main()

