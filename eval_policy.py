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

from rllab.misc.console import query_yes_no
from rllab.sampler.utils import rollout


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=10,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1000,
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
    pr = cProfile.Profile()
    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']

    # Sample one rollout
    pr.enable()
    path = rollout(env, policy, max_path_length=args.max_path_length,
                        animated=args.render, speedup=args.speedup)
    pr.disable()
    if args.render:
        sys.stdout.write("Press <enter> to continue: ")
        input()

    # Parse code profiling
    print('\n')
    ps = pstats.Stats(pr).strip_dirs().sort_stats('cumulative')
    ps.print_stats(10)



if __name__ == "__main__":
    main()

