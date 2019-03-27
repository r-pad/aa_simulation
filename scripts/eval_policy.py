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
    plt.show()


def plot_error_curve(error, name, units):
    """
    Plot error over time.
    """
    title = '%s Error over Time in Final Policy' % name

    plt.figure()
    t = np.arange(error.size)
    plt.plot(t, error)
    plt.title(title)
    plt.xlabel('Time steps')
    plt.ylabel('Error (%s)' % units)
    if name == 'Distance':
        plt.gca().set_ylim((-0.25, 0.25))
    else:
        plt.gca().set_ylim((-0.7, 0.7))
    plt.show()


def plot_distribution(error, name, units):
    """
    Plot histogram showing distribution of error.
    """
    mean = error.mean()
    std = error.std()
    maximum = error.max()
    minimum = error.min()
    stats = 'Mean = %.5f\nStd = %.5f\nMax = %.5f\nMin = %.5f' % \
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


def rescale_actions(actions):
    vel_lb = VehicleEnv._MIN_VELOCITY
    vel_ub = VehicleEnv._MAX_VELOCITY
    steer_lb = -VehicleEnv._MAX_STEER_ANGLE
    steer_ub = VehicleEnv._MAX_STEER_ANGLE
    lb = np.array([vel_lb, steer_lb])
    ub = np.array([vel_ub, steer_ub])
    scaled_actions = []
    for i, action in enumerate(actions):
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        scaled_actions.append(scaled_action)
    return np.array(scaled_actions)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='Path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
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
    np.random.seed(9)

    # Sample one rollout
    profiler.enable()
    path = rollout(env, policy, max_path_length=args.max_path_length,
                        animated=args.render, speedup=args.speedup,
                        always_return_paths=True)
    profiler.disable()

    # Policy analysis
    profile_code(profiler)
    actions = rescale_actions(path['actions'])
    plot_curve(actions[:, 0][skip:], 'Commanded Speed', 'm/s')
    plot_curve(actions[:, 1][skip:], 'Commanded Steering Angle', 'rad')
    plot_error_curve(path['env_infos']['dist'][skip:], 'Distance', 'm')
    plot_error_curve(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')
    plot_distribution(path['env_infos']['dist'][skip:], 'Distance', 'm')
    plot_distribution(path['env_infos']['vel'][skip:], 'Velocity', 'm/s')

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == "__main__":
    main()

