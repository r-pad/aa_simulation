#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Jiyuan Zhou

Train local planner using TRPO so that a vehicle can follow a sequence
of arbitrary curvatures.
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

from aa_simulation.envs.local_planner_env_straight import LocalPlannerEnvStraight


def run_task(vv, log_dir=None, exp_name=None):

    # Load environment
    target_velocity = vv['target_velocity']
    env = normalize(LocalPlannerEnvStraight(target_velocity))

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
        n_itr=1500,
        discount=0.99,
        step_size=0.01,
        plot=False,
    )
    algo.train()


def main():

    # Set up multiple experiments at once
    vg = VariantGenerator()
    vg.add('target_velocity', [0.7])
    vg.add('seed', [100])
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
