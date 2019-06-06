#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training local planner to follow circles of
arbitrary curvature with ROS.

----------------------------------------------
Potential TODOs:
    - Read next state after three messages
----------------------------------------------
"""

import ackermann_msgs.msg
import nav_msgs.msg
import numpy as np
import rospy

from rllab.envs.base import Step

from aa_simulation.envs.circle_env import CircleEnv
from aa_simulation.misc.transformations import euler_from_quaternion


class CircleEnvROS(CircleEnv):
    """
    Simulation environment for an RC car following a circular
    arc trajectory using relative coordinates with ROS.
    """

    def __init__(self, target_velocity, radius, dt, model_type):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(CircleEnvROS, self).__init__(
            target_velocity=target_velocity,
            dt=dt,
            radius=radius,
            model_type=model_type
        )

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
        """
        return self.current_state


    def step(self, action):
        """
        Move one iteration forward by sending action to robot via ROS.
        """
        if action[0] < 0:   # Only allow forward direction
            action[0] = 0
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
        nextstate = self.current_state
        self._state = nextstate

        reward, info = self.get_reward(nextstate, action)
        return Step(observation=info['observation'], reward=reward, done=False,
                dist=info['dist'], vel=info['vel'], kappa=self._model.kappa)


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

