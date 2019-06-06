#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train local planner using TRPO so that a vehicle can follow a
straight line.

-----------------------------------------------------------------
TODO:
 - allow fine-tuning and reset variance when doing so
 - set W_gain and init_std as functions of target velocity
-----------------------------------------------------------------
"""

import argparse

import joblib
import lasagne.init as LI
import lasagne.nonlinearities as LN
import numpy as np

from rllab.algos.trpo import TRPO
from rllab.core.network import MLP
from rllab.envs.base import Env
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.cpo.baselines.linear_feature_baseline import LinearFeatureBaseline

from aa_simulation.envs.straight_env import StraightEnv

# Pre-trained policy and baseline
policy = None
baseline = None


def run_task(vv, log_dir=None, exp_name=None):
    global policy
    global baseline

    # Load environment
    if not vv['use_ros']:
        env = StraightEnv(
            target_velocity=vv['target_velocity'],
            dt=vv['dt'],
            model_type=vv['model_type']
        )
    else:
        from aa_simulation.envs.straight_env_ros import StraightEnvROS
        env = StraightEnvROS(
            target_velocity=vv['target_velocity'],
            dt=vv['dt'],
            model_type=vv['model_type']
        )

    # Save variant information for comparison plots
    variant_file = logger.get_snapshot_dir() + '/variant.json'
    logger.log_variant(variant_file, vv)

    # Build policy and baseline networks
    # Note: Mean of policy network set to analytically computed values for
    #       faster training (rough estimates for RL to fine-tune).
    if policy is None or baseline is None:
        target_velocity = vv['target_velocity']
        target_steering = 0
        output_mean = np.array([target_velocity, target_steering])
        hidden_sizes = (32, 32)
        W_gain = 0.1
        init_std = 0.1
        mean_network = MLP(
            input_shape=(env.spec.observation_space.flat_dim,),
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=LN.tanh,
            output_nonlinearity=None,
            output_W_init=LI.GlorotUniform(gain=W_gain),
            output_b_init=output_mean
        )
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            init_std = init_std,
            mean_network=mean_network
        )
        baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=400,
        max_path_length=env.horizon,
        n_itr=600,
        discount=0.99,
        step_size=0.01,
        plot=False,
    )
    algo.train()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str,
            help='Path to snapshot file of pre-trained network')
    args = parser.parse_args()
    return args


def main():
    global policy
    global baseline

    # Load pre-trained network if available
    args = parse_arguments()
    if args.network is not None:
        data = joblib.load(args.network)
        policy = data['policy']
        baseline = data['baseline']

    # Set up multiple experiments at once
    vg = VariantGenerator()
    use_ros = False
    seeds = [100, 200]
    vg.add('seed', seeds)
    vg.add('target_velocity', [0.7, 1.5, 2.0, 2.5, 3.0])
    vg.add('dt', [0.1])
    vg.add('model_type', ['BrushTireModel'])
    vg.add('use_ros', [use_ros])
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

