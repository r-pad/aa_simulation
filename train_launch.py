#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train a policy using TRPO so that a vehicle follow a trajectory
resembling an arc from a circle using relative coordinates.
"""

import numpy as np

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.base import Env
from rllab.envs.normalized_env import normalize
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from car_simulation.envs.arc.arc_relative1 import ArcRelativeEnv1
from car_simulation.envs.arc.arc_relative2 import ArcRelativeEnv2


def run_task(vv, log_dir=None, exp_name=None):

    # Load environment
    radius = vv['radius']
    target_velocity = vv['target_velocity']
    if vv['relative_type'] == 1:
        env = normalize(ArcRelativeEnv1(radius, target_velocity))
    if vv['relative_type'] == 2:
        env = normalize(ArcRelativeEnv2(radius, target_velocity))

    # Save variant information for comparison plots
    variant_file = logger.get_snapshot_dir() + '/variant.json'
    logger.log_variant(variant_file, vv)

    # Train policy using TRPO
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        max_path_length=env.horizon,
        n_itr=500,
        discount=0.99,
        step_size=0.01,
        plot=False,
    )
    algo.train()


def main():

    # Set up multiple experiments at once
    vg = VariantGenerator()
    vg.add('relative_type', [1, 2])    # Relative state space convention
    vg.add('target_velocity', [0.7, 0.8, 0.9, 1.0])
    vg.add('radius', [1.0])
    vg.add('seed', [100, 200])
    print('Number of Configurations: ', len(vg.variants()))

    # Run each experiment variant
    for vv in vg.variants():
        run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            n_parallel=1,
            snapshot_mode='last',
            seed=vv['seed']
        )


if __name__ == '__main__':
    main()
