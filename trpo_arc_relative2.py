#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Train a policy using TRPO so that a vehicle follow a trajectory
resembling an arc from a circle using relative coordinates
"""

import numpy as np

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.base import Env
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from car_simulation.envs.arc.arc_relative2 import ArcRelativeEnv2


def run_task(*_):
    radius = 1
    target_velocity = 0.7
    env = normalize(ArcRelativeEnv2(radius, target_velocity))

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


run_experiment_lite(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=np.random.randint(1, 1000000),
    plot=False,
)
