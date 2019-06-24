#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train local planner using CPO so that a vehicle can follow a circular
trajectory with an arbitrary curvature as fast as possible within
an epsilon distance away from the trajectory.

------------------------------------------------------------
TODO:
    - reset variance for fine-tuning
    - make variants the same as circle/straight planner.py
------------------------------------------------------------
"""

import argparse

import joblib
import numpy as np

import lasagne.init as LI
import lasagne.nonlinearities as LN

from rllab.core.network import MLP
from rllab.envs.base import Env
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.cpo.algos.safe.cpo import CPO
from sandbox.cpo.baselines.linear_feature_baseline import LinearFeatureBaseline

from aa_simulation.envs.fast_circle_env import FastCircleEnv
from aa_simulation.safety_constraints.circle import CircleSafetyConstraint

# Pre-trained policy and baseline
policy = None
baseline = None


def run_task(vv, log_dir=None, exp_name=None):
    global policy
    global baseline

    trpo_stepsize = 0.01
    trpo_subsample_factor = 0.2

    # Check if variant is available
    if vv['model_type'] not in ['BrushTireModel', 'LinearTireModel']:
        raise ValueError('Unrecognized model type for simulating robot')
    if vv['robot_type'] not in ['MRZR', 'RCCar']:
        raise ValueError('Unrecognized robot type')

    # Load environment
    env = FastCircleEnv(
        target_velocity=vv['target_velocity'],
        radius=vv['radius'],
        dt=vv['dt'],
        model_type=vv['model_type'],
        robot_type=vv['robot_type']
    )

    # Save variant information for comparison plots
    variant_file = logger.get_snapshot_dir() + '/variant.json'
    logger.log_variant(variant_file, vv)

    # Build policy and baseline networks
    # Note: Mean of policy network set to analytically computed values for
    #       faster training (rough estimates for RL to finetune).
    if policy is None or baseline is None:
        wheelbase = 0.257
        target_velocity = vv['target_velocity']
        target_steering = np.arctan(wheelbase / vv['radius'])  # CCW
        output_mean = np.array([vv['target_velocity'], target_steering])
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
            hidden_sizes=hidden_sizes,
            init_std=init_std,
            mean_network=mean_network
        )
        baseline = LinearFeatureBaseline(
            env_spec=env.spec,
            target_key='returns'
        )

    safety_baseline = LinearFeatureBaseline(
        env_spec=env.spec,
        target_key='safety_returns'
    )

    safety_constraint = CircleSafetyConstraint(
        max_value=1.0,
        eps=vv['eps'],
        baseline=safety_baseline
    )

    algo = CPO(
        env=env,
        policy=policy,
        baseline=baseline,
        safety_constraint=safety_constraint,
        batch_size=600,
        max_path_length=env.horizon,
        n_itr=1000,
        discount=0.99,
        step_size=trpo_stepsize,
        gae_lambda=0.95,
        safety_gae_lambda=1,
        optimizer_args={'subsample_factor':trpo_subsample_factor},
        plot=False
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
        use_pretrained = True
    else:
        use_pretrained = False

    # Run multiple experiment variants at once
    vg = VariantGenerator()

    # Non-configurable parameters (do not change)
    vg.add('trajectory', ['Circle'])
    vg.add('objective', ['Fast'])
    vg.add('algo', ['CPO'])

    # Configurable parameters
    #   Options for model_type: 'BrushTireModel', 'LinearTireModel'
    #   Options for robot_type: 'MRZR', 'RCCar'
    # Note: There is no notion of a target velocity in CPO, but it does
    #       control the distribution of the initial state. See the function
    #       get_initial_state() in envs/circle_env.py for more information.
    robot_type = 'RCCar'
    seeds = [100, 200]
    vg.add('seed', seeds)
    vg.add('target_velocity', [1.0])
    vg.add('radius', [1.0])
    vg.add('dt', [0.1])
    vg.add('eps', [0.5])
    vg.add('model_type', ['BrushTireModel'])
    vg.add('robot_type', [robot_type])
    vg.add('pretrained', [use_pretrained])
    print('Number of Configurations: ', len(vg.variants()))

    # Run each experiment variant
    for vv in vg.variants():
        run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            n_parallel=4,
            snapshot_mode='last',
            seed=vv['seed']
        )


if __name__ == '__main__':
    main()
