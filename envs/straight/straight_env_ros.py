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

from aa_simulation.envs.straight.straight_env import StraightEnv
from aa_simulation.misc.transformations import euler_from_quaternion
from aa_simulation.misc.utils import rotate_state, translate_state


class StraightEnvROS(StraightEnv):
    """
    Simulation environment for an RC car following a straight
    line trajectory with ROS.
    """

    def __init__(self, target_velocity, dt, model_type, robot_type):
        """
        Initialize super class, parameters, ROS-related modules.
        """
        super(StraightEnv, self).__init__(
            target_velocity=target_velocity,
            dt=dt,
            model_type=model_type,
            robot_type=robot_type
        )

        # Reward function parameters
        self._lambda1 = 0.25

        # State in frame of target straight line. See get_initial_state()
        # function for info.
        self._current_state = np.zeros(6)

        # Target line to follow. See get_initial_state() function for info.
        self._target_y = 0
        self._target_yaw = 0

        # Initialize nodes and set up ROS topics
        rospy.init_node('rl_planner')
        self.publisher = rospy.Publisher('commands/keyboard',
                ackermann_msgs.msg.AckermannDriveStamped, queue_size=1)
        rospy.Subscriber('ekf_localization/odom', nav_msgs.msg.Odometry,
                self._odometry_callback)
        self._sensor_stamp = rospy.Time.now()

        # Wait this number of timesteps before computing another action (this
        # is similar to setting a larger dt during training)
        self._num_states_needed = 3
        self._num_states_received = 0


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.

        To make training from scratch easier, set the target straight line to
        a random straight line, which is chosen by adding some random noise
        to the robot's current yaw, and adding a random y translation. Now, we
        view this line from a reference frame that transforms this line to be
        y=0. The variable self._current_state is the state of the robot in this
        new reference frame.
        """
        # Create private copy of state to avoid race conditions
        state = np.array(self._current_state)

        # Transform state back to inertial frame
        state = rotate_state(translate_state(state, [0, self._target_y])),
                self._target_yaw)

        # Choose random target yaw value based on current yaw
        self._target_yaw = state[2] + np.random.uniform(-np.pi/6, np.pi/6)

        # Choose random target y with respect to new target yaw
        rotated_y = rotate_state(state, -self._target_yaw)[1]
        self._target_y = rotated_y + np.random.uniform(-0.25, 0.25)

        # Set current state to the robot's state in the frame of the target
        # straight line
        self._set_current_state(state)

        return self._current_state


    def step(self, action):
        """
        Move one iteration forward by sending action to robot via ROS.
        """
        self._action = action
        if action[0] < 0:   # Only allow forward direction
            action[0] = 0

        # Publish action via ROS
        msg = ackermann_msgs.msg.AckermannDriveStamped()
        msg.drive.speed = action[0]
        msg.drive.steering_angle = action[1]
        msg.header.stamp = self._sensor_stamp
        self.publisher.publish(msg)

        # Wait for next state readings
        self._num_states_received = self._num_states_needed
        while self._num_states_received > 0:
            time.sleep(0.00001)
        nextstate = self._current_state
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
        _, y, _, x_dot, y_dot, _ = state
        velocity = np.sqrt(x_dot**2 + y_dot**2)
        distance = y

        reward = -np.absolute(distance)
        reward -= self._lambda1 * (velocity - self.target_velocity)**2

        info = {}
        info['dist'] = distance
        info['vel'] = velocity
        return reward, info


    def _odometry_callback(self, odom):
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
        self._set_current_state(np.array([x, y, yaw, x_dot, y_dot, yaw_dot]))
        self._sensor_stamp = odom.header.stamp
        if self_num_states_received > 0:
            self._num_states_received -= 1


    def _set_current_state(self, state):
        """
        Set current state to the specified state in the frame of the target
        straight line.
        """
        self._current_state = translate_state(
                rotate_state(state, -self._target_yaw), [0, -self._target_y])

