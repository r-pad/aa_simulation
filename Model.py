#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Class defining kinematics and dynamics of a RWD vehicle.

Code based on MATLAB simulation code written by Emily Yunan, located
at https://github.com/jsford/FFAST.
"""

import numpy as np
from scipy.integrate import odeint


class VehicleModel(object):
    """
    Vehicle modeled as a three degrees of freedom bicycle model
    with brush tire model for tire dynamics.
    """

    def __init__(self, params):
        """
        Initialize model parameters from dictionary format to instance
        variables.
        """
        # Vehicle parameters
        self.m = params['m']                # Mass
        self.L_f = params['L_f']            # CoG to front axle length
        self.L_r = params['L_r']            # CoG to rear axle length
        self.L = self.L_f + self.L_r        # Front to rear axle length
        self.load_f = params['load_f']      # Load on front axle
        self.load_r = params['load_r']      # Load on rear axle

        # Wheel parameters
        self.C_x = params['C_x']            # Longitudinal stiffness
        self.C_alpha = params['C_alpha']    # Cornering stiffness
        self.I_z = params['I_z']            # Moment of inertia
        self.mu = params['mu']              # Coeff. of friction
        self.mu_s = params['mu_s']          # Sliding coeff. of friction


    def state_transition(self, X, U, dt):
        """
        Update state after some timestep.
        """
        t = np.array([0, dt])
        X_new = odeint(self._dynamics, X, t, args=(U,), mxstep=5000000)
        return X_new[1]


    def _dynamics(self, X, t, U):
        """
        Use dynamics model to compute X_dot from X, U.
        """
        pos_x = X[0]
        pos_y = X[1]
        pos_yaw = VehicleModel.wraptopi(X[2]);
        v_x = X[3]
        v_y = X[4]
        yaw_rate = X[5]
        cmd_vx = U[0]
        delta = U[1]

        # Tire slip angle (zero when stationary)
        if np.abs(v_x) < 0.01 and np.abs(v_x) < 0.01:
            alpha_f = 0
            alpha_r = 0
        else:
            alpha_f = np.arctan2((v_y + self.L_f*yaw_rate), v_x) - delta
            alpha_r = np.arctan2((v_y - self.L_r*yaw_rate), v_x)

        # Compute forces on tires using brush tire model
        F_yf = self._tire_dynamics_front(alpha_f)
        F_xr, F_yr = self._tire_dynamics_rear(v_x, cmd_vx, alpha_r)

        # Find dX
        T_z = self.L_f*F_yf*np.cos(delta) - self.L_r*F_yr
        ma_x = F_xr - F_yf*np.sin(delta)
        ma_y = F_yf*np.cos(delta) + F_yr

        # Acceleration with damping
        yaw_rate_dot = T_z/self.I_z - 0.02*yaw_rate
        v_x_dot = ma_x/self.m + yaw_rate*v_y - 0.025*v_x
        v_y_dot = ma_y/self.m - yaw_rate*v_x - 0.025*v_y

        # Translate to inertial frame
        v = np.sqrt(v_x**2 + v_y**2)
        beta = np.arctan2(v_y,v_x)
        pos_x_dot = v*np.cos(beta+pos_yaw)
        pos_y_dot = v*np.sin(beta+pos_yaw)

        X_dot = np.zeros(6)
        X_dot[0] = pos_x_dot
        X_dot[1] = pos_y_dot
        X_dot[2] = yaw_rate
        X_dot[3] = v_x_dot
        X_dot[4] = v_y_dot
        X_dot[5] = yaw_rate_dot

        return X_dot


    def _tire_dynamics_front(self, alpha):
        """
        Front tire dynamics.
        """
        # alpha > pi/2 is invalid because of the use of tan(). Since
        # alpha > pi/2 means vehicle moving backwards, Fy's sign has
        # to be reversed, hence we multiply by sign(alpha)
        if abs(alpha) > np.pi/2:
            alpha = (np.pi-abs(alpha))*np.sign(alpha)

        # Compute slip angle where total sliding occurs alpha_sl
        alpha_sl = np.arctan(3*self.mu*self.load_f/self.C_alpha)

        if abs(alpha) <= alpha_sl:
            tan = np.tan(alpha)
            first = -self.C_alpha * tan
            second = self.C_alpha**2 / (3*self.mu*self.load_f) *\
                    np.abs(tan) * tan
            third = -self.C_alpha**3 / (27*self.mu**2*self.load_f**2) *\
                    tan**3
            Fy = first + second + third
        else:
            Fy = -self.mu*self.load_f*np.sign(alpha)

        return Fy


    def _tire_dynamics_rear(self, v_x, wheel_vx, alpha):
        """
        Rear tire dynamics.
        """
        # Find longitudinal wheel slip K (kappa)
        if (np.abs(wheel_vx-v_x) < 0.01 or
                (np.abs(wheel_vx) < 0.01 and np.abs(v_x) < 0.01)):
            K = 0
        # Infinite slip, longitudinal saturation
        elif abs(v_x) < 0.01:
            Fx = np.sign(wheel_vx)*self.mu*self.load_r
            Fy = 0
            return Fx, Fy
        else:
            K = (wheel_vx-v_x)/np.abs(v_x);

        # Instead of avoiding -1, now look for positive equivalent
        if K < 0:
            spin_dir = -1
            K = np.abs(K)
        else:
            spin_dir = 1;

        # alpha > pi/2 is invalid because of the use of tan(). Since
        # alpha > pi/2 means vehicle moving backwards, Fy's sign has
        # to be reversed, hence we multiply by sign(alpha)
        if abs(alpha) > np.pi/2:
            alpha = (np.pi-abs(alpha))*np.sign(alpha)

        # Compute combined slip value gamma
        gamma = np.sqrt( self.C_x**2 * (K/(1+K))**2 + self.C_alpha**2
                * (np.tan(alpha)/(1+K))**2 )

        if gamma <= 3*self.mu*self.load_r:
            F = gamma - 1/(3*self.mu*self.load_r)*gamma**2 + \
                    1/(27*self.mu**2*self.load_r**2)*gamma**3
        else:
            F = self.mu_s * self.load_r

        if gamma == 0:
            Fx = 0
            Fy = 0
        else:
            Fx = self.C_x/gamma * (K/(1+K)) * F * spin_dir
            Fy = -self.C_alpha/gamma * (np.tan(alpha)/(1+K)) * F

        return Fx, Fy


    @staticmethod
    def wraptopi(val):
        """
        Wrap radian value to the interval [-pi, pi].
        """
        pi = np.pi
        val = val - 2*pi*np.floor((val+pi)/(2*pi));
        return val

