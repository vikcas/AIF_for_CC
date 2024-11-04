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
        self.cost = np.array(['LOW', 'MID', 'HIGH'])
        self.share_info = np.array([False, True])

        self.num_states = [len(self.success), len(self.distance), len(self.fps), len(self.res), len(self.cost), len(self.share_info)]
        self.num_observations = [len(self.success), len(self.distance), len(self.fps), len(self.res), len(self.cost), len(self.share_info)]
        self.num_factors = len(self.num_states)

        # Controls
        self.u_share_info = np.array([False, True])

        self.num_controls = [len(self.u_share_info)]
        self.A_dependency = [[0], [1], [2], [3], [4], [5]]
        self.B_factor_list = [[0, 3], [1, 2], [2], [3], [4, 5], [5]]
        self.B_factor_control_list = [[0], [0], [0], [0], [0], [0]]

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

        A_matrices = [np.eye(self.success.size), np.eye(self.distance.size), np.eye(self.fps.size),
                      np.eye(self.res.size), np.eye(self.cost.size), np.eye(self.share_info.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3], 4: A_matrices[4], 5: A_matrices[5]}
        ranges = [len(self.success), len(self.distance), len(self.fps), len(self.res), len(self.cost), len(self.share_info)]

        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            for mm in range(ranges[4]):
                                for nn in range(ranges[5]):
                                    if i == 0:
                                        self.A[i][:, :, jj, kk, ll, mm, nn] = index_to_A[0]
                                    if i == 1:
                                        self.A[i][:, ii, :, kk, ll, mm, nn] = index_to_A[1]
                                    if i == 2:
                                        self.A[i][:, ii, jj, :, ll, mm, nn] = index_to_A[2]
                                    if i == 3:
                                        self.A[i][:, ii, jj, kk, :, mm, nn] = index_to_A[3]
                                    if i == 4:
                                        self.A[i][:, ii, jj, kk, ll, :, nn] = index_to_A[4]
                                    if i == 5:
                                        self.A[i][:, ii, jj, kk, ll, mm, :] = index_to_A[5]

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
        # succ(t+1), succ(t), res(t), act(share_state)
        # If share_state is false, we assume identity
        # If share_state is true, we assume improvement
        for idx, _ in enumerate(self.res):
            self.B[0][:, :, idx, 0] = np.eye(self.num_states[0])
            self.B[0][:, :, idx, 1] = np.array([[0, 0], [1, 1]])

        # Distance transition matrices
        # dist(t+1), dist(t), fps(t), act(share_state)
        # If share_state is false, we assume identity
        # If share_state is true, we assume improvement
        for idx, _ in enumerate(self.fps):
            self.B[1][:, :, idx, 0] = np.eye(self.num_states[1])
            self.B[1][:, :, idx, 1] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])

        # FPS matrices
        # fps(t+1), fps(t), act(share_state)
        # We are assumimng that we do not know becuase we do not care, however, we know a little more.
        # I.e., To improve distance we want high FPS.
        self.B[2][:, :, 0] = np.eye(self.num_states[2])
        # self.B[2][:, :, 1] = np.array([[.5, .3, 0, 0, 0], [.5, .4, .3, 0, 0], [0, .3, .4, .3, 0], [0, 0, .3, .4, .5], [0, 0, 0, .3, .5]])
        self.B[2][:, :, 1] = np.array(
            [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])

        # RES matrices
        # Res(t+1), Res(t), act(share_state)
        self.B[3][:, :, 0] = np.eye(self.num_states[3])
        # self.B[3][:, :, 1] = np.array([[.5, .3, 0, 0, 0, 0], [.5, .4, .3, 0, 0, 0], [0, .3, .4, .3, 0, 0], [0, 0, .3, .4, .3, 0], [0, 0, 0, .3, .4, .5], [0, 0, 0, 0, .3, .5]])
        self.B[3][:, :, 1] = np.array(
            [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]])

        # Cost Transition matrix
        # cost(t+1), cost(t), s_info(t), act(s_info)
        self.B[4][:, :, 0, 0] = np.eye(self.num_states[4])  # It is False and action is False, stays de same
        self.B[4][:, :, 0, 1] = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 1]])  # It is False and action goes to True, it increases costs
        self.B[4][:, :, 1, 0] = np.array(
            [[1, 1, 0], [0, 0, 1], [0, 0, 0]])  # It is True and action goes False, it decreases costs
        self.B[4][:, :, 1, 1] = np.eye(self.num_states[4])  # It is True and action is True, stays de same

        # Share Info
        # s_info(t+1), s_info(t), act(s_info)
        self.B[5][:, :, 0] = np.array([[1, 1], [0, 0]])
        self.B[5][:, :, 1] = np.array([[0, 0], [1, 1]])

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
        # succ(t+1), succ(t), res(t), act(share_state)
        for idx, _ in enumerate(self.res):
            self.B[0][:, :, idx, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])
            self.B[0][:, :, idx, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])

        # Distance transition matrices
        # dist(t+1), dist(t), fps(t), act(share_state)
        for idx, _ in enumerate(self.fps):
            self.B[1][:, :, idx, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
            self.B[1][:, :, idx, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])

        # FPS matrices
        # fps(t+1), fps(t), act(share_state)
        self.B[2][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[2])

        # RES matrices
        # Res(t+1), Res(t), act(share_state)
        self.B[3][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])

        # Cost Transition matrix
        # cost(t+1), cost(t), s_info(t), act(s_info)
        self.B[4][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[4])

        # Share Info
        # s_info(t+1), s_info(t), act(s_info)
        self.B[5][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[5])
        self.B[5][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[5])

    def generate_CD(self):
        # Vector C Goal distribution
        self.C[0] = np.array([0.25, 3])  # Priority on Success
        self.C[1] = np.array([3, 2.5, 2, 0.5, 0.1])  # Distance should be short
        self.C[2] = np.zeros(self.num_states[2])
        self.C[3] = np.zeros(self.num_states[3])
        self.C[4] = np.array([3, 2.5, 0.5])
        self.C[5] = np.zeros(self.num_states[5])

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
        self.generate_CD()

        if self.b_matrix_learning:
            self.generate_B_unknown()
            pA, pB = self.generate_uniform_dirichlet_dist()
            return Agent(A=self.A, pA=pA, B=self.B, pB=pB, C=self.C, D=self.D, policy_len=self.policy_length,
                         num_controls=self.num_controls, B_factor_list=self.B_factor_list, lr_pB=0.2, gamma=25,
                         B_factor_control_list=self.B_factor_control_list, action_selection='deterministic', alpha=16,
                         use_param_info_gain=True, inference_algo="VANILLA")
            # return Agent(A=self.A, pA=pA, B=self.B, pB=pB, C=self.C, D=self.D, policy_len=self.policy_length,
            #              num_controls=self.num_controls, B_factor_list=self.B_factor_list,
            #              B_factor_control_list=self.B_factor_control_list, action_selection='deterministic',
            #              use_param_info_gain=True, inference_algo="VANILLA")
        else:
            self.generate_B()
            return Agent(A=self.A, B=self.B, C=self.C, D=self.D, policy_len=self.policy_length,
                         num_controls=self.num_controls, B_factor_list=self.B_factor_list,
                         B_factor_control_list=self.B_factor_control_list)

    def translate_action(self, action_list):
        share_info_action = [False, True]
        action = {'share_info': share_info_action[int(action_list[0])]}

        return action