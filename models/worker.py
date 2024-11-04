import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from utils.aux_fncs import generate_normalized_2d_sq_matrix


class WorkerAgent:
    '''

    '''
    def __init__(self, b_matrix_learning=True, policy_length=1):
        # States and Observations
        self.in_time = np.array([False, True])
        self.execution_time = np.array(["LOW", "MID-LOW", "MID", "MID-HIGH", "HIGH"])
        self.fps = np.array(["12", "16", "20", "26", "30"])
        self.cost = np.array(['LOW', 'MID', 'HIGH'])
        self.share_info = np.array([False, True])
        self.gpu = np.array(["OFF", "ON"])

        self.num_states = [len(self.in_time), len(self.execution_time), len(self.fps), len(self.cost), len(self.share_info), len(self.gpu)]
        self.num_observations = [len(self.in_time), len(self.execution_time), len(self.fps), len(self.cost), len(self.share_info), len(self.gpu)]
        self.num_factors = len(self.num_states)

        # Controls
        self.u_share_info = np.array([False, True])
        self.u_gpu = np.array(["SWITCH_OFF", "STAY", "SWITCH_ON"])

        self.num_controls = [len(self.u_share_info), len(self.u_gpu)]

        self.A_factor_list = [[0], [1], [2], [3], [4], [5]]
        self.B_factor_list = [[0, 1, 2], [1, 5], [2], [3, 4, 5], [4], [5]]
        self.B_factor_control_list = [[0], [1], [0], [0, 1], [0], [1]]

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
        A_matrices = [np.eye(self.in_time.size), np.eye(self.execution_time.size), np.eye(self.fps.size),
                      np.eye(self.cost.size), np.eye(self.share_info.size), np.eye(self.gpu.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3], 4: A_matrices[4],
                      5: A_matrices[5]}
        ranges = [len(self.in_time), len(self.execution_time), len(self.fps), len(self.cost), len(self.share_info), len(self.gpu)]
        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            for mm in range(ranges[4]):
                                for nn in range(ranges[5]):
                                    if i == 0:
                                        self.A[i][:, :, jj, kk, ll, mm, nn] = index_to_A[i]
                                    if i == 1:
                                        self.A[i][:, ii, :, kk, ll, mm, nn] = index_to_A[i]
                                    if i == 2:
                                        self.A[i][:, ii, jj, :, ll, mm, nn] = index_to_A[i]
                                    if i == 3:
                                        self.A[i][:, ii, jj, kk, :, mm, nn] = index_to_A[i]
                                    if i == 4:
                                        self.A[i][:, ii, jj, kk, ll, :, nn] = index_to_A[i]
                                    if i == 5:
                                        self.A[i][:, ii, jj, kk, ll, mm, :] = index_to_A[i]

    def generate_B(self):
        '''

        :return:
        '''
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if
                             i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # in_time transition.
        # [in_time (t+1), in_time(t), execution_time(t), fps(t), acton(share_state)]
        # When share state is FALSE - Transition matrices are the identity, i.e. no change
        for idx, _ in enumerate(self.execution_time):
            for idy, _ in enumerate(self.fps):
                self.B[0][:, :, idx, idy, 0] = np.eye(self.num_states[0])
        # Otherwise it is assumed to be improved.
        for idx, _ in enumerate(self.execution_time):
            for idy, _ in enumerate(self.fps):
                self.B[0][:, :, idx, idy, 1] = np.array([[0, 0], [1, 1]])

        # Execution time.
        # exe (t+1), exe(t), gpu(t), act(gpu)
        self.B[1][:, :, 0, 0] = np.eye(self.num_states[1])  # GPU OFF & ACT OFF
        self.B[1][:, :, 0, 1] = np.eye(self.num_states[1])  # GPU OFF & ACT STAY
        self.B[1][:, :, 0, 2] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])  # GPU OFF & ACT ON
        self.B[1][:, :, 1, 0] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])  # GPU ON & ACT OFF
        self.B[1][:, :, 1, 1] = np.eye(self.num_states[1])
        self.B[1][:, :, 1, 2] = np.eye(self.num_states[1])

        # FPS
        # fps(t+1), fps(t), act(share_state)
        self.B[2][:, :, 0] = np.eye(self.num_states[2])
        # self.B[2][:, :, 1] = np.array([[.5, .3, 0, 0, 0], [.5, .4, .3, 0, 0], [0, .3, .4, .3, 0], [0, 0, .3, .4, .5], [0, 0, 0, .3, .5]])
        self.B[2][:, :, 1] = np.array(
            [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])

        # Cost Transition matrix
        # cost(t+1), cost(t), s_info(t), gpu(t), act(s_info), act(gpu)
        # cost changes when there is change on the s_info or gpu state.
        # Conflicts, one increase, while the other decrease are considered as neutral - no change
        for si_id, _ in enumerate(self.share_info):
            for gpu_id, _ in enumerate(self.gpu):
                for u_si_id, _ in enumerate(self.u_share_info):
                    for u_gpu_id, _ in enumerate(self.u_gpu):
                        # Check Comm activated:
                        if si_id == 0 and u_si_id == 1:
                            # Check GPU de-activated
                            if gpu_id == 1 and u_gpu_id == 0:
                                # Maintain cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.eye(self.num_states[3])
                                continue
                            else:
                                # Increase cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                                    [[0, 0, 0], [1, 0, 0], [0, 1, 1]])
                                continue
                        # Check Comm de-activated
                        if si_id == 1 and u_si_id == 0:
                            # Check GPU activated
                            if gpu_id == 0 and u_gpu_id == 2:
                                # Maintain cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.eye(self.num_states[3])
                                continue
                            else:
                                # Decrease cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                                    [[1, 1, 0], [0, 0, 1], [0, 0, 0]])
                                continue
                        # Check GPU activated
                        if gpu_id == 0 and u_gpu_id == 2:
                            # Check comm de-activated
                            if si_id == 1 and u_si_id == 0:
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.eye(self.num_states[3])
                                continue
                            else:  # Increase cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                                    [[0, 0, 0], [1, 0, 0], [0, 1, 1]])
                                continue
                        # Check GPU de-activated
                        if gpu_id == 1 and u_gpu_id == 0:
                            # Check Comm activated
                            if si_id == 0 and u_si_id == 1:  #  Maintain cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id] = np.eye(self.num_states[3])
                                continue
                            else:  # Decrease cost
                                self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                                    [[1, 1, 0], [0, 0, 1], [0, 0, 0]])
                                continue
                        # Any other case maintains
                        self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.eye(self.num_states[3])

                        #
                        #     # Check GPU Staying OR GPU being activated
                        #     if u_gpu_id == 1 or (gpu_id == 0 and u_gpu_id == 2):
                        #         # Increase cost
                        #         self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                        #             [[0, 0, 0], [1, 0, 0], [0, 1, 1]])
                        #     # Check GPU de-activated
                        #     elif gpu_id == 1 and u_gpu_id == 0:
                        #         # Maintain cost
                        #         self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.eye(self.num_states[3])
                        #     else:
                        #         # Any other case - Increase cost.
                        #         self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                        #             [[0, 0, 0], [1, 0, 0], [0, 1, 1]])
                        #
                        #
                        #
                        # if (si_id == 0 and u_si_id == 1) and ((gpu_id == 0 and u_gpu_id == 0) or (gpu_id == 1 and u_gpu_id == 2) or u_gpu_id == 1):
                        #     self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
                        # elif (si_id == 1 and u_si_id == 0) and ((gpu_id == 0 and u_gpu_id == 0) or (gpu_id == 1 and u_gpu_id == 2) or u_gpu_id == 1):
                        #     self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])
                        # elif (gpu_id == 0 and u_gpu_id == 2) and ((si_id == u_si_id) or (si_id == 0 and u_si_id == 1)):
                        #     self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                        #         [[0, 0, 0], [1, 0, 0], [0, 1, 1]])
                        # elif (gpu_id == 1 and u_gpu_id == 0) and ((si_id == u_si_id) or (si_id == 1 and u_si_id == 0)):
                        #     self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.array(
                        #         [[1, 1, 0], [0, 0, 1], [0, 0, 0]])
                        # else:
                        #     self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = np.eye(self.num_states[3])

        # self.B[3][:, :, 0, 0] = np.eye(self.num_states[3])  # It is False and action is False, stays de same
        # self.B[3][:, :, 0, 1] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])  # It is False and action goes to True, it increases costs
        # self.B[3][:, :, 1, 0] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])  # It is True and action goes False, it decreases costs
        # self.B[3][:, :, 1, 1] = np.eye(self.num_states[3])  # It is True and action is True, stays de same

        # Share Info
        # s_info(t+1), s_info(t), act(s_info)
        self.B[4][:, :, 0] = np.array([[1, 1], [0, 0]])
        self.B[4][:, :, 1] = np.array([[0, 0], [1, 1]])

        # GPU
        # gpu(t+1), gpu(t), act(gpu)
        self.B[5][:, :, 0] = np.array([[1, 1], [0, 0]])
        self.B[5][:, :, 1] = np.eye(self.num_states[5])
        self.B[5][:, :, 2] = np.array([[0, 0], [1, 1]])

    def generate_B_unknown(self):
        '''

        :return:
        '''
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if
                             i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # in_time transition.
        # [in_time (t+1), in_time(t), execution_time(t), fps(t), acton(share_state)]
        # When share state is FALSE - Transition matrices are the identity, i.e. no change
        for idx, _ in enumerate(self.execution_time):
            for idy, _ in enumerate(self.fps):
                self.B[0][:, :, idx, idy, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])
                self.B[0][:, :, idx, idy, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])

        # Execution time.
        # exe (t+1), exe(t), gpu(t), act(gpu)
        self.B[1][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])

        # FPS
        # fps(t+1), fps(t), act(share_state)
        self.B[2][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[2])

        # Cost Transition matrix
        # cost(t+1), cost(t), s_info(t), act(s_info)
        # self.B[3][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        # self.B[3][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        # self.B[3][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        # self.B[3][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        for si_id, _ in enumerate(self.share_info):
            for gpu_id, _ in enumerate(self.gpu):
                for u_si_id, _ in enumerate(self.u_share_info):
                    for u_gpu_id, _ in enumerate(self.u_gpu):
                        self.B[3][:, :, si_id, gpu_id, u_si_id, u_gpu_id] = generate_normalized_2d_sq_matrix(self.num_states[3])

        # Share Info
        # s_info(t+1), s_info(t), act(s_info)
        self.B[4][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[4])

        # GPU
        # gpu(t+1), gpu(t), act(gpu)
        self.B[5][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[5])
        self.B[5][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[5])
        self.B[5][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[5])

    def generate_CD(self):
        # Vector C Goal distribution
        self.C[0] = np.array([0.1, 3])  # In time has to be TRUE
        self.C[1] = np.array([3, 2.5, 2, 0.25, 0.1]) # Exe time has to be low - but not an SLO
        self.C[2] = np.zeros(self.num_states[2])  # FPS no direct interest
        self.C[3] = np.array([3, 2.5, 0.5])  # Cost interest on being low
        self.C[4] = np.zeros(self.num_states[4])  # Share info no direct interest
        self.C[5] = np.zeros(self.num_states[5])  # No direct interest on GPU status

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
        gpu_action = ['SWITCH OFF', 'STAY', 'SWITCH ON']

        action = {'share_info': share_info_action[int(action_list[0])], 'gpu_action': gpu_action[int(action_list[1])]}

        return action