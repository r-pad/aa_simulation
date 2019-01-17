#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Jiyuan Zhou

Enable an agent to follow a hard coded trajectory in the form of
a square with rounded corners using trained straight and circle models.
"""
import argparse
import cProfile
import pstats
import sys
import time
import math
import yaml

import joblib
import matplotlib.pyplot as plt
import numpy as np

from rllab.misc import tensor_utils

from aa_simulation.envs.renderer import _Renderer


def render(renderer, state, action):
    """
    Render simulation environment.
    """
    renderer.update(state, action)


def modify_state_curve(state, move_param):
    """
    Convert state [x, y, yaw, x_dot, y_dot, yaw_dot] to
        [dx, theta, ddx, dtheta]
    """
    x_0, y_0, r = move_param
    x, y, yaw, x_dot, y_dot, yaw_dot = state

    x -= x_0
    y -= y_0

    vel = np.sqrt(np.square(x_dot) + np.square(y_dot))

    dx = np.sqrt(np.square(x) + np.square(y)) - r
    theta = _normalize_angle(np.arctan2(-x, y) + np.pi - yaw)
    ddx = x/(x**2 + y**2)**0.5*x_dot + y/(x**2 + y**2)**0.5*y_dot
    dtheta = x/(x**2 + y**2)*x_dot - y/(x**2 + y**2)*y_dot - yaw_dot

    return np.array([dx, theta, ddx, dtheta, vel, 1])


def _normalize_angle(angle):
    """
    Normalize angle to [-pi, pi).
    """
    angle = angle % (2*np.pi)
    if (angle >= np.pi):
        angle -= 2*np.pi
    return angle


def _normalize_angle2(angle):
    """
    Normalize angle to [0, 2 * pi).
    """
    angle = angle % (2*np.pi)
    return angle


def modify_state_straight(state, move_param):
    """
    Add target direction and target velocity to state, to feed
    in the NN.
    """
    x_0, y_0, target_dir = move_param
    x, y, yaw, x_dot, y_dot, dyaw = state
    target_dir = _normalize_angle2(target_dir)

    new_x, new_y = _cal_distance(x, y, move_param)
    yaw = _normalize_angle(yaw) - target_dir
    yaw = _normalize_angle(yaw)

    new_x_dot = x_dot * np.cos(target_dir) + y_dot * np.sin(target_dir)
    new_y_dot = y_dot * np.cos(target_dir) - x_dot * np.sin(target_dir)

    return np.array([new_y, yaw, new_x_dot, new_y_dot, dyaw])


def _cal_distance(x, y, move_param):
    # For arbitrary trajectory.
    init_x, init_y, target_dir = move_param

    # if _normalize_angle(target_dir) == math.pi / 2:
    #     next_x, next_y = init_x, init_y + 1
    #     print(1)
    #     return(0, - x + init_x)
    # elif _normalize_angle(target_dir) == -math.pi / 2:
    #     next_x, next_y = init_x, init_y - 1
    #     return(0, x - init_x)
    # else:
    #     next_x, next_y = init_x + 1, init_y + np.tan(target_dir)

    #print("x,y", x, y, init_x, init_y)
    position_dir = np.arctan2((y - init_y), (x - init_x))
    projection_dir = _normalize_angle(position_dir - target_dir)
    #print("yaws", position_dir, target_dir, projection_dir)
    dist = np.sqrt(np.square(x - init_x) + np.square(y - init_y))
    # new_y = np.absolute((next_y - init_y) * init_x + (init_x - next_x) * init_y\
    #              - init_x * next_y + next_x * init_y) / \
    #                 np.sqrt(np.square(next_y - init_y) + np.square(next_x - init_x))

    new_y = dist * np.sin(projection_dir)
    # new_x = dist * np.cos(projection_dir)
    new_x = 0
    # new_y = (y - init_y) * np.cos(target_dir) - (x - init_x) * np.sin(target_dir)
    #print("new y: ", new_y)
    # if (np.sin(projection_dir < 0)):
    #     new_y = new_y

    return (new_x, new_y)


def _check_point(state, way_point):
    # Potential bug!!! Can only follow a curve that is less than
    # 180 degrees! must be deal with in the hard coded trajectory
    # or higher level planner for the time being.
    x, y, _, _, _, _ = state
    check_point_x, check_point_y, direction = way_point
    direction = np.deg2rad(direction)

    state_direction = np.arctan2((y - check_point_y), (x - check_point_x))

    intersect_angle = _normalize_angle(state_direction - direction)

    return np.absolute(intersect_angle) <= math.pi / 2
    # return True


def rollout(env, agent, way_point=[], animated=False, speedup=1,
            always_return_paths=False, renderer=None, state=np.zeros(6),
            isCurve=False, move_param=[]):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    path_length = 0

    # update initial state!!!
    # Bad implementation. Just temporary.
    env._wrapped_env._state = state

    while _check_point(state, way_point):
        # print("State: ", state)
        # State observation convertion
        if isCurve:
            o = modify_state_curve(state, move_param)
        else:
            o = modify_state_straight(state, move_param)
        #
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        #

        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        path_length += 1
        if d:
            break

        o = next_o
        # Bad implementation. Just temporary.
        state = env._wrapped_env._state

        if animated:
            render(renderer, state, a)
            #env.render()
            timestep = 0.0001
            time.sleep(timestep / speedup)
    return state


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speedup', type=float, default=100000,
                        help='Speedup')
    parser.add_argument('--render', dest='render',
            action='store_true', help='Rendering')
    parser.add_argument('--no-render', dest='render',
            action='store_false', help='Rendering')
    parser.set_defaults(render=True)
    args = parser.parse_args()
    return args


def move(env, policy, args, way_point, renderer,\
         state, isCurve, move_param):
    final_state = rollout(env, policy, way_point=way_point,
                        animated=args.render, speedup=args.speedup,
                        always_return_paths=True, renderer=renderer,
                        state=state, isCurve=isCurve,\
                        move_param=move_param)
    return final_state


def init_render():
    stream = open('aa_simulation/envs/model_params.yaml', 'r')
    params = yaml.load(stream)
    obstacles = []
    goal = None
    return _Renderer(params, obstacles, goal, None)


def _check_curve_way_point(curve_param, way_point):
    center_x, center_y, curve_angle = curve_param
    curve_angle = _normalize_angle(curve_angle)

    if curve_angle <= math.pi / 2:
        return curve_param, way_point

    check_point_x, check_point_y, direction = way_point

    # Construct new way point


def main():
    args = parse_arguments()
    profiler = cProfile.Profile()

    data_curve = joblib.load("data/roundedsquare_demo/curve.pkl")
    policy_curve = data_curve['policy']
    env_curve = data_curve['env']

    data_straight = joblib.load("data/roundedsquare_demo/straight.pkl")
    policy_straight = data_straight['policy']
    env_straight = data_straight['env']

    plt.ion()

    # Set fixed random seed
    np.random.seed(9)

    # Sample one rollout
    profiler.enable()

    renderer = init_render()

    state = [-1, 0, np.deg2rad(-90), 0, 0, 0]
    render(renderer, state, None)

    # center positin x, center position y, radius
    curve_params = [[0, 0, 1], [2, 0, 1], [2, 2, 1], [0, 2, 1]]
    # start position x, start position y, target start yaw(direction)
    straight_params = [[0, -1, np.deg2rad(0)], [3, 0, np.deg2rad(90)],\
             [2, 3, np.deg2rad(-180)], [-1, 2, np.deg2rad(-90)]]
    # curve_step_size = [43, 44, 44, 44]
    # straight_step_size = [54, 55, 55, 56]

    way_points = [[0, -1, 180], [2, -1, 180], [3, 0, -90], [3, 2, -90],\
                    [2, 3, 0], [0, 3, 0], [-1, 2, 90], [-1, 0, 90]]

    point = 0
    for i in range(400):

        i %= 4
        # Turn left for 90 degrees
        point %= 8
        state = move(env_curve, policy_curve, args,\
                way_points[point], renderer, state,\
                True, curve_params[i])
        print(state)
        point += 1

        # Move straightly for length 2
        point %= 8
        state = move(env_straight, policy_straight, args,\
                way_points[point], renderer, state,\
                False, straight_params[i])
        print(state)
        point += 1

    profiler.disable()

    # Block until key is pressed
    sys.stdout.write("Press <enter> to continue: ")
    input()


if __name__ == '__main__':
    main()
