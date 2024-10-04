import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from utils.aux_fncs import generate_normalized_2d_sq_matrix


class ConsumerAgent:
    '''

    '''
    def __init__(self, b_matrix_learning=True, policy_length=1):
        # States and Observations
        self.success = np.array([False, True])
        self.distance = np.array(["SHORT", "MID-SHORT", "MID", "MID-LONG", "LONG"])
        self.fps = np.array(["12", "16", "20", "26", "30"])
        self.res = np.array(["120p", "180p", "240p", "360p", "480p", "720p"])

        self.num_states = [len(self.success), len(self.distance), len(self.fps), len(self.res)]
        self.num_observations = [len(self.success), len(self.distance), len(self.fps), len(self.res)]
        self.num_factors = len(self.num_states)

        # Controls
        self.u_fps = ["DECREASE", "STAY", "INCREASE"]
        self.u_res = ["DECREASE", "STAY", "INCREASE"]

        self.num_controls = [len(self.u_fps), len(self.u_res)]
        self.A_dependency = [[0], [1], [2], [3]]
        self.B_factor_list = [[0, 3], [1, 2], [2], [3]]
        self.B_factor_control_list = [[1], [0], [0], [1]]

        A_shapes = [[o_dim] + self.num_states for o_dim in self.num_observations]
        self.A = utils.obj_array_zeros(A_shapes)
        self.B = utils.obj_array(self.num_factors)
        self.C = utils.obj_array_zeros(self.num_observations)
        self.D = utils.obj_array_zeros(self.num_states)

        self.policy_length = policy_length
        self.b_matrix_learning = b_matrix_learning

    def generate_A(self):
        '''

        :return:
        '''

        A_matrices = [np.eye(self.success.size), np.eye(self.distance.size), np.eye(self.fps.size), np.eye(self.res.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3]}
        ranges = [len(self.success), len(self.distance), len(self.fps), len(self.res)]

        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            if i == 0:
                                self.A[i][:, :, jj, kk, ll] = index_to_A[0]
                            if i == 1:
                                self.A[i][:, ii, :, kk, ll] = index_to_A[1]
                            if i == 2:
                                self.A[i][:, ii, jj, :, ll] = index_to_A[2]
                            if i == 3:
                                self.A[i][:, ii, jj, kk, :] = index_to_A[3]

        # A_latency = np.eye(self.latency.size)
        # A_resolution = np.eye(self.resolution.size)
        #
        # for ii in range(len(self.resolution)):
        #     self.A[0][:, :, ii] = A_latency
        #
        # for ii in range(len(self.latency)):
        #     self.A[1][:, ii, :] = A_resolution

    def generate_B_to_learn(self):
        '''

        :return:
        '''
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # Latency improves when FPS is decreased
        self.B[0][:, :, 0] = np.array([[0.3, 0.2], [0.7, 0.8]])
        self.B[0][:, :, 1] = np.eye(self.num_states[0])
        self.B[0][:, :, 2] = np.array([[0.8, 0.7], [0.2, 0.3]])

        # Resolution improves when Resolution is increased
        self.B[1][:, :, 0] = np.array([[0.7, 0.7], [0.3, 0.3]])
        self.B[1][:, :, 1] = np.eye(self.num_states[1])
        self.B[1][:, :, 2] = np.array([[0.2, 0.2], [0.8, 0.8]])

    def generate_B_unknown(self):
        '''

        :return:
        '''
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # Success transition matrices
        # succ(t+1), succ(t), res(t), act(res)
        self.B[0][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 120p & Decr
        self.B[0][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 120p & stay
        self.B[0][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 120p & Incr
        self.B[0][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 180p & Decr
        self.B[0][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 180p & stay
        self.B[0][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 180p & Incr
        self.B[0][:, :, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 240p & Decr
        self.B[0][:, :, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 240p & stay
        self.B[0][:, :, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 240p & Incr
        self.B[0][:, :, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 360p & Decr
        self.B[0][:, :, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 360p & stay
        self.B[0][:, :, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 360p & Incr
        self.B[0][:, :, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 480p & Decr
        self.B[0][:, :, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 480p & stay
        self.B[0][:, :, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 480p & Incr
        self.B[0][:, :, 5, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 720p & Decr
        self.B[0][:, :, 5, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 720p & stay
        self.B[0][:, :, 5, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # Res 720p & Incr

        # Distance transition matrices
        # dist(t+1), dist(t), fps(t), act(fps)
        self.B[1][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])  # fps 12, decr
        self.B[1][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])  # fps 12, stay
        self.B[1][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[1]) # fps 30, stay
        self.B[1][:, :, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[1]) # fps 30, incr

        # FPS matrices
        # FPS(t+1), FPS(t), act(FPS)
        self.B[2][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[2])

        # RES matrices
        # Res(t+1), Res(t), act(res)
        self.B[3][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])  # RESOLUTION decreases by 1 unit if it can
        self.B[3][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])  # RESOLUTION stays the same if it is not changed.
        self.B[3][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])  # RESOLUTION increases by 1 unit if it ca

    def generate_B(self):
        '''

        :return:
        '''
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # Success transition matrices
        # succ(t+1), succ(t), res(t), act(res)
        self.B[0][:, :, 0, 0] = np.eye(self.num_states[0])  # Res 120p & Decr
        self.B[0][:, :, 0, 1] = np.eye(self.num_states[0])  # Res 120p & stay
        self.B[0][:, :, 0, 2] = np.array([[0, 0], [1, 1]])  # Res 120p & Incr
        self.B[0][:, :, 1, 0] = np.array([[1, 1], [0, 0]])  # Res 180p & Decr
        self.B[0][:, :, 1, 1] = np.eye(self.num_states[0])  # Res 180p & stay
        self.B[0][:, :, 1, 2] = np.array([[0, 0], [1, 1]])  # Res 180p & Incr
        self.B[0][:, :, 2, 0] = np.array([[1, 1], [0, 0]])  # Res 240p & Decr
        self.B[0][:, :, 2, 1] = np.eye(self.num_states[0])  # Res 240p & stay
        self.B[0][:, :, 2, 2] = np.array([[0, 0], [1, 1]])  # Res 240p & Incr
        self.B[0][:, :, 3, 0] = np.eye(self.num_states[0])  # Res 360p & Decr
        self.B[0][:, :, 3, 1] = np.eye(self.num_states[0])  # Res 360p & stay
        self.B[0][:, :, 3, 2] = np.array([[0, 0], [1, 1]])  # Res 360p & Incr
        self.B[0][:, :, 4, 0] = np.eye(self.num_states[0])  # Res 480p & Decr
        self.B[0][:, :, 4, 1] = np.eye(self.num_states[0])  # Res 480p & stay
        self.B[0][:, :, 4, 2] = np.array([[0, 0], [1, 1]])  # Res 480p & Incr
        self.B[0][:, :, 5, 0] = np.eye(self.num_states[0])  # Res 720p & Decr
        self.B[0][:, :, 5, 1] = np.eye(self.num_states[0])  # Res 720p & stay
        self.B[0][:, :, 5, 2] = np.array([[0, 0], [1, 1]])  # Res 720p & Incr

        # Distance transition matrices
        # dist(t+1), dist(t), fps(t), act(fps)
        self.B[1][:, :, 0, 0] = np.eye(self.num_states[1])  # fps 12, decr
        self.B[1][:, :, 0, 1] = np.eye(self.num_states[1])  # fps 12, stay
        self.B[1][:, :, 0, 2] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])  # fps 12, incr
        self.B[1][:, :, 1, 0] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
        self.B[1][:, :, 1, 1] = np.eye(self.num_states[1])
        self.B[1][:, :, 1, 2] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[1][:, :, 2, 0] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
        self.B[1][:, :, 2, 1] = np.eye(self.num_states[1])
        self.B[1][:, :, 2, 2] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[1][:, :, 3, 0] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])
        self.B[1][:, :, 3, 1] = np.eye(self.num_states[1])
        self.B[1][:, :, 3, 2] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[1][:, :, 4, 0] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]]) # fps 30, decr
        self.B[1][:, :, 4, 1] = np.eye(self.num_states[1]) # fps 30, stay
        self.B[1][:, :, 4, 2] = np.eye(self.num_states[1]) # fps 30, incr

        # FPS matrices
        # FPS(t+1), FPS(t), act(FPS)
        self.B[2][:, :, 0] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[2][:, :, 1] = np.eye(self.num_states[2])
        self.B[2][:, :, 2] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])

        # RES matrices
        # Res(t+1), Res(t), act(res)
        self.B[3][:, :, 0] = np.array(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0]])  # RESOLUTION decreases by 1 unit if it can
        self.B[3][:, :, 1] = np.eye(self.num_states[3])  # RESOLUTION stays the same if it is not changed.
        self.B[3][:, :, 2] = np.array(
            [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 1]])  # RESOLUTION increases by 1 unit if it can

        # # Latency improves when FPS is decreased
        # self.B[0][:, :, 0] = np.array([[0, 0], [1, 1]])
        # self.B[0][:, :, 1] = np.eye(self.num_states[0])
        # self.B[0][:, :, 2] = np.array([[1, 1], [0, 0]])
        #
        # # Resolution improves when Resolution is increased
        # self.B[1][:, :, 0] = np.array([[1, 1], [0, 0]])
        # self.B[1][:, :, 1] = np.eye(self.num_states[1])
        # self.B[1][:, :, 2] = np.array([[0, 0], [1, 1]])

    def generate_CD(self):
        # Vector C Goal distribution
        self.C[0] = np.array([0.25, 3])  # Priority on Success
        self.C[1] = np.array([3, 2.5, 2, 0.5, 0.1])  # Distance should be short
        self.C[2] = np.zeros(self.num_states[2])
        self.C[3] = np.zeros(self.num_states[3])

        # Vector D - Prior state distribution - UNKNOWN
        D = [np.ones(ns) / ns for ns in self.num_states]
        for idx, arr in enumerate(D):
            self.D[idx] = arr

    def generate_uniform_dirichlet_dist(self):
        pA = utils.dirichlet_like(self.A)
        pB = utils.dirichlet_like(self.B)
        return pA, pB
    def generate_agent(self):
        self.generate_A()

        if self.b_matrix_learning:
            self.generate_B_unknown()
        else:
            self.generate_B()

        self.generate_CD()
        # Important: Learning
        pA, pB = self.generate_uniform_dirichlet_dist()
        # return Agent(A=self.A, B=self.B, C=self.C, D=self.D, policy_len=self.policy_length,
        #              num_controls=self.num_controls, B_factor_list=self.B_factor_list,
        #              B_factor_control_list=self.B_factor_control_list)
        return Agent(A=self.A, pA=pA, B=self.B, pB=pB, C=self.C, D=self.D, policy_len=self.policy_length,
                     num_controls=self.num_controls, B_factor_list=self.B_factor_list, lr_pB=1, gamma=12,
                     B_factor_control_list=self.B_factor_control_list, action_selection='deterministic', alpha=12,
                     use_param_info_gain=True, inference_algo="VANILLA")

    def observation(self, observation_list):
        state = [['BAD', 'GOOD'], ['BAD', 'GOOD']]

        agent_observation = list()
        for idx, obs in enumerate(observation_list):
            agent_observation.append(state[idx][:].index(obs))

        return agent_observation

    def translate_action(self, action_list):
        latency_action = ['DECREASE', 'STAY', 'INCREASE']
        resolution_action = ['DECREASE', 'STAY', 'INCREASE']

        action = {'fps_demand': latency_action[int(action_list[0])], 'resolution_demand': resolution_action[int(action_list[1])]}

        return action