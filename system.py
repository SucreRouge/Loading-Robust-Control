import argparse
import logging
import sys
import tensorflow as tf
import gym
from gym import wrappers
import random
import datetime
# from hydraulic.baselines.common import set_global_seeds
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
# import matlab.engine
from sklearn import svm
import control as cl

# # PID part
# class Env2(object):
#     def __init__(self):
#         self.q_pump = 80
#         self.n_pump = 1500
#         self.q_motor = 0
#         self.p_sys = 0
#         self.p_set = 0
#         self.T_set = 0
#         self.Q_pump = 0
#         self.Q_motor = 0
#         self.n_set = 0
#         self.n_ctrl = 0
#         self.eff_pump = lambda x: -0.1 * (x / 30) + 1
#         self.eff_motor = lambda x: -0.1 * (x / 30) + 1
#         self.observation = [self.Q_pump, self.T_set, self.n_set]
#         self.action = [[self.q_motor, self.p_set, self.n_ctrl]]
#         self.reward = 0
#         self.count = 0
#         self.done = 0
#         self.info = None
#         self.ob_space = 2
#         self.ac_space = 1
#         self.scope = []
#
#     def step(self, action, out_action):  # out_action = [p_set, q_pump]
#         # self.action = action
#         self.done = np.array([False])
#         [[self.q_motor]] = action
#         self.p_set = out_action[0]
#         self.q_pump = out_action[1]
#         self.Q_motor = self.q_motor * self.n_pump / 1000 / self.eff_motor(self.p_set)
#         self.Q_pump = self.q_pump * self.n_pump / 1000 * self.eff_pump(self.p_set)
#         f_trace = lambda x: x - 20 if x - 20 > 0 else 0
#         self.q_trace = f_trace(self.q_pump)
#         f_relief = lambda x, y: x - y if x > y else 0
#         self.Q_relief = f_relief(self.Q_pump, self.Q_motor)
#         if (self.Q_pump < self.Q_motor):
#             self.p_sys = 0
#         else:
#             self.p_sys = self.p_set
#
#         self.observation = [self.Q_pump, self.p_set]
#         self.scope = [self.Q_motor, self.Q_relief, self.q_trace, self.p_sys, self.p_set]
#         # self.action = [self.q_motor, self.p_set]
#         # self.reward = [- (self.Q_pump - self.Q_motor) - (self.p_set - self.p_sys)]
#         # self.reward = [-0.01*(self.Q_pump - self.q_motor * self.n_pump / 1000) - 100*(self.p_set - self.p_sys)]
#         compare_p = lambda x, y: -1 if x > y else 1
#         self.reward = [0.1 * self.q_motor + 100 * compare_p(self.p_set, self.p_sys)]
#         if (self.p_sys != self.p_set):
#             self.count += 1
#         if (self.count == 10):
#             self.done = np.array([True])
#             self.count = 0
#         return [self.observation, self.reward, self.done, self.info, self.scope]
#         # test
#
#     def reset(self):
#         self.__init__()
#         return self.observation
#


class Env1(object):
    def __init__(self):
        # q_pump最大为125
        self.q_pump = 80
        self.n_pump = 1500
        self.q_motor = 0
        self.p_sys = 0
        self.p_set = 0
        self.Q_pump = 0
        self.Q_motor = 0
        self.eff_pump = lambda x: -0.1 * (x / 30) + 1
        self.eff_motor = lambda x: -0.1 * (x / 30) + 1
        self.observation = [self.Q_pump, self.p_set]
        self.action = [[self.q_motor]]
        self.reward = 0
        self.count = 0
        self.done = 0
        self.info = None
        self.ob_space = 2
        self.ac_space = 1
        self.scope = []

    def step(self, action, out_action):  # out_action = [p_set, q_pump]
        # self.action = action
        self.done = np.array([False])
        [[self.q_motor]] = action
        self.p_set = out_action[0]
        self.q_pump = out_action[1]
        self.Q_motor = self.q_motor * self.n_pump / 1000 / self.eff_motor(self.p_set)
        self.Q_pump = self.q_pump * self.n_pump / 1000 * self.eff_pump(self.p_set)
        f_trace = lambda x:x-20 if x-20>0 else 0
        self.q_trace = f_trace(self.q_pump)
        f_relief = lambda x,y:x-y if x>y else 0
        self.Q_relief =f_relief(self.Q_pump,self.Q_motor)
        if (self.Q_pump < self.Q_motor):
            self.p_sys = 0
        else:
            self.p_sys = self.p_set

        self.observation = [self.Q_pump/self.n_pump, self.p_set]
        self.scope = [self.Q_motor, self.Q_relief, self.q_trace, self.p_sys, self.p_set]
        # self.action = [self.q_motor, self.p_set]
        # self.reward = [- (self.Q_pump - self.Q_motor) - (self.p_set - self.p_sys)]
        # self.reward = [-0.01*(self.Q_pump - self.q_motor * self.n_pump / 1000) - 100*(self.p_set - self.p_sys)]
        compare_p = lambda x,y:-1 if x>y else 1
        self.reward = [0.1 * self.q_motor + 100 * compare_p(self.p_set, self.p_sys)]
        if (self.p_sys != self.p_set):
            self.count += 1
        if (self.count == 10):
            self.done = np.array([True])
            self.count = 0
        return [self.observation, self.reward, self.done, self.info, self.scope]
    # test
    def reset(self):
        self.__init__()
        return self.observation


class hydraulic(object):
    def __init__(self):
        self.n_ctrl = 0
        self.n_real = 0
        self.T_elec = 0
        self.eff_elec = 0.95

        self.T_motor = 0
        self.K_w = 0
        self.w = 0
        self.J_w = 0

        self.N_MAX=3500
        self.T_MAX=3500

        self.n_real_OLD = 0

        self.q_test = 75
        self.effmec_test = lambda x: 1 / 300 * x + 0.85
        self.p_relief = 0
        self.q_motor = 0
        self.eff_motor = lambda x: -0.1 * (x / 30) + 1
        self.eff_test = lambda x: -0.1 * (x / 30) + 1
        self.effmec_motor = lambda x: 1 / 300 * x + 0.85

        self.total_T = 0
        self.total_T_OLD = 0
        # self.n_real_OLD=0 #上一次的n_real
        self.T0_elec = 100  # 电机的额定转矩
        self.J_alpha = 1
        self.k_w = 1
        self.trans_elec = cl.tf([1], [self.J_alpha, self.k_w])
        self.info = None
        self.ob_space = 6
        self.ac_space = 2
        self.scope = []
        # observation
        self.n_set = 0
        self.T_set = 0
        self.n_real = 0
        self.T_test = 0
        self.p_sys = 0
        self.Q_test = 0
        self.observation = [self.n_set, self.n_real, self.T_set, self.T_test, self.p_sys,
                            self.Q_test/self.n_real if self.n_real !=0 else 0]

    def runsampletime(self, sampletime, action, q_peishi, set_action):
        self.n_ctrl = action[0][0]
        self.p_relief = action[0][1]
        #self.q_motor = q_peishi
        self.q_motor = np.reshape(q_peishi,[1,])
        self.n_set = set_action[0]
        self.T_set = set_action[1]
        # 不管采样时间多少，系统计算间隔为0.001
        for i in range(int(sampletime / 0.001)):
            self.total_T_OLD = self.total_T
            self.n_real_OLD = self.n_real
            self.T_elec = self.Elec_Torque(self.n_ctrl, self.n_real)
            self.p_sys = self.p_relief if self.q_motor * self.eff_motor(self.p_sys) > self.q_test / self.eff_test(
                self.p_sys) else 0
            self.T_test = (0.5/np.pi) * self.p_sys * self.q_test * self.effmec_test(self.p_sys)
            self.T_motor = (0.5/np.pi) * self.p_sys * self.q_motor / self.effmec_motor(self.p_sys)
            # 用于加速的总扭矩
            self.total_T = self.T_elec * self.eff_elec + self.T_test - self.T_motor
            t = np.array([0, 0.001])
            U = np.array([self.total_T_OLD, self.total_T])
            y = cl.lsim(self.trans_elec, U, t, X0=self.n_real_OLD)
            self.n_real = y[0][1]
            self.Q_test = self.q_test * self.n_real * self.eff_test(self.p_sys)
        self.reward = -(np.square((self.n_set-self.n_real)/self.N_MAX)+np.square((self.T_set-self.T_test)/self.T_MAX))
        self.done = np.array([True])
        self.scope = [self.n_real, self.T_test]
        return [self.observation, self.reward, self.done, self.info, self.scope]

    def runrealsystem(self, sampletime, action, q_peishi, set_action):
        self.n_ctrl = action[0][0]
        self.p_relief = action[0][1]
        self.q_motor = q_peishi
        self.n_set = set_action[0]
        self.T_set = set_action[1]
        return [self.observation, self.reward, self.done, self.info, self.scope]

    def reset(self):
        self.n_ctrl = 0
        self.n_real = 0
        self.T_elec = 0
        self.eff_elec = 0.95
        self.T_test = 0
        self.T_motor = 0
        self.K_w = 0
        self.w = 0
        self.J_w = 0
        self.n_real = 0
        self.n_real_OLD = 0
        self.p_sys = 0
        self.q_test = 75
        self.p_relief = 0
        self.q_motor = 0
        self.Q_test = 0
        self.total_T = 0
        self.total_T_OLD = 0

    def Elec_Torque(self, n_ctrl, n_real):
        if (n_real <= n_ctrl - 50):
            return (n_real - 0) / (2950 / (0.25 * self.T0_elec)) + 1.25 * self.T0_elec
        elif ((n_ctrl - 50 < n_real) and (n_real < n_ctrl + 50)):
            return (n_real - (n_ctrl - 50)) / (
            (-50 - 50) / (1.5 * self.T0_elec + 1.5 * self.T0_elec)) + 1.5 * self.T0_elec
        else:
            return (n_real - (n_ctrl + 50)) / (2950 / (0.25 * self.T0_elec)) + (-1.5 * self.T0_elec)






"""
__title__ = '$Package_name'
__author__ = '$USER'
__mtime__ = '$DATE'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
