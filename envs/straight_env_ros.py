#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training a local planner to move in a straight line with ROS.
"""

import ackermann_msgs.msg
import nav_msgs.msg
import numpy as np
import rospy

from rllab.envs.base import Step

from aa_simulation.envs.straight_env import StraightEnv
from aa_simulation.misc.transformations import euler_from_quaternion


class StraightEnvROS(StraightEnv):
    """
    Simulation environment for an RC car following a straight
    line trajectory with ROS.
    """

    def __init__(self, target_velocity, dt, model_type):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(StraightEnv, self).__init__(target_velocity, dt, model_type)

        # Reward function parameters
        self._lambda1 = 0.25

        # Target line to follow. See get_initial_state() function for info.
        self.target_y = 0

        # Initialize nodes and set up ROS topics
        rospy.init_node('rl_planner')
        self.publisher = rospy.Publisher('commands/keyboard',
                ackermann_msgs.msg.AckermannDriveStamped, queue_size=1)
        rospy.Subscriber('ekf_localization/odom', nav_msgs.msg.Odometry,
                self.odometry_callback)

        self._action_published = False
        self.current_state = np.zeros(6)


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.

        To make training from scratch easier, set the target straight line to
        be located not on y=0 but rather at y=self.current_state[1] which is
        the current y-value.
        """
        self.target_y = self.current_state[1]
        return self.current_state


    def step(self, action):
        """
        Move one iteration forward by sending action to robot via ROS.
        """
        self._action = action

        # Publish action via ROS
        msg = ackermann_msgs.msg.AckermannDriveStamped()
        msg.drive.speed = action[0]
        msg.drive.steering_angle = action[1]
        self.publisher.publish(msg)
        self._action_published = True

        # Wait for next state reading
        while self._action_published:
            continue
        next_observation = self._state_to_observation(nextstate)
        self._state = nextstate

        reward, info = self.get_reward(nextstate, action)
        return Step(observation=next_observation, reward=reward,
                done=False, dist=info['dist'], vel=info['vel'],
                kappa=self._model.kappa)


    def get_reward(self, state, action):
        """
        Reward function definition.
        """
        x, y, _, x_dot, y_dot, _ = state
        velocity = np.sqrt(np.square(x_dot) + np.square(y_dot))
        vel_diff = velocity - self.target_velocity
        distance = y - self.target_y

        reward = -np.absolute(distance)
        reward -= self._lambda1 * np.square(vel_diff)

        info = {}
        info['dist'] = distance
        info['vel'] = vel_diff
        return reward, info

    def odometry_callback(self, odom):
        """
        Callback function for odometry state updates.
        """
        x = odom.pose.pose.position.x
        y = odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        rpy = euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw = rpy[2]
        x_dot = odom.twist.twist.linear.x
        y_dot = odom.twist.twist.linear.y
        yaw_dot = odom.twist.twist.angular.z
        self.current_state = [x, y, yaw, x_dot, y_dot, yaw_dot]
        self._action_published = False
        return

