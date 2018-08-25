#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train a policy using TRPO so that a vehicle follow a trajectory
resembling an arc from a circle.
"""

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.base import Env
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


def run_task(*_):
    env = normalize(load_class('car_simulation.envs.arc.arc_absolute',
        Env, ["rllab", "envs"])())

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
        n_itr=1000,
        discount=0.99,
        step_size=0.01,
        plot=False,
    )
    algo.train()


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",   # Keep snapshot params for last iteration
    seed=9,
    plot=False,
)
