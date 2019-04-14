#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train local planner using TRPO so that a vehicle can follow a circular
trajectory with an arbitrary curvature.
"""

import numpy as np

import lasagne.nonlinearities as LN

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.core.network import MLP
from rllab.envs.base import Env
from rllab.envs.normalized_env import normalize
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from aa_simulation.envs.circle_env import CircleEnv


def run_task(vv, log_dir=None, exp_name=None):

    # Load environment
    env = normalize(CircleEnv(
        target_velocity=vv['target_velocity'],
        radius=vv['radius'],
        dt=vv['dt']
    ))

    # Save variant information for comparison plots
    variant_file = logger.get_snapshot_dir() + '/variant.json'
    logger.log_variant(variant_file, vv)

    # Build policy and baseline networks
    # Note: Mean of policy network set to analytically computed values for
    #       faster training (rough estimates for RL to finetune).
    wheelbase = 0.257
    target_velocity = vv['target_velocity']
    target_steering = np.arctan(wheelbase / vv['radius'])
    output_mean = np.array([vv['target_velocity'], target_steering])
    hidden_sizes = (32, 32)
    mean_network = MLP(
        input_shape=(env.spec.observation_space.flat_dim,),
        output_dim=env.spec.action_space.flat_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=LN.tanh,
        output_nonlinearity=None,
        output_b_init=output_mean
    )
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        mean_network=mean_network
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    # Train using TRPO
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=600,
        max_path_length=env.horizon,
        n_itr=600,
        discount=0.99,
        step_size=0.01,
        plot=False,
    )
    algo.train()


def main():

    # Set up multiple experiments at once
    vg = VariantGenerator()
    seeds = [100, 200]
    vg.add('seed', seeds)
    vg.add('target_velocity', [0.7])
    vg.add('radius', [1.0])
    vg.add('dt', [0.03])
    print('Number of Configurations: ', len(vg.variants()))

    # Run each experiment variant
    for vv in vg.variants():
        run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            n_parallel=len(seeds),
            snapshot_mode='last',
            seed=vv['seed']
        )


if __name__ == '__main__':
    main()
