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
        self.req_gpu = np.array(["OFF", "STAY", "ON"])
        self.gpu = np.array(["OFF", "ON"])

        self.num_states = [len(self.in_time), len(self.execution_time), len(self.fps), len(self.req_gpu), len(self.gpu)]
        self.num_observations = [len(self.in_time), len(self.execution_time), len(self.fps), len(self.req_gpu), len(self.gpu)]
        self.num_factors = len(self.num_states)

        # Controls
        self.u_req_fps = ["DECREASE", "STAY", "INCREASE"]
        self.u_gpu = ["SWITCH_OFF", "STAY", "SWITCH_ON"]

        self.num_controls = [len(self.u_req_fps), len(self.u_gpu)]

        self.A_factor_list = [[0], [1], [2], [3], [4]]
        self.B_factor_list = [[0, 1, 2], [1, 4], [2], [3, 4], [4]]
        self.B_factor_control_list = [[0], [1], [0], [1], [1]]

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
        A_matrices = [np.eye(self.in_time.size), np.eye(self.execution_time.size), np.eye(self.fps.size), np.eye(self.req_gpu.size), np.eye(self.gpu.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3], 4: A_matrices[4]}
        ranges = [len(self.in_time), len(self.execution_time), len(self.fps), len(self.req_gpu), len(self.gpu)]
        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            for mm in range(ranges[4]):
                                if i == 0:
                                    self.A[i][:,:,jj,kk,ll,mm] = index_to_A[i]
                                if i == 1:
                                    self.A[i][:, ii, :, kk, ll, mm] = index_to_A[i]
                                if i == 2:
                                    self.A[i][:, ii, jj, :, ll, mm] = index_to_A[i]
                                if i == 3:
                                    self.A[i][:, ii, jj, kk, :, mm] = index_to_A[i]
                                if i == 4:
                                    self.A[i][:, ii, jj, kk, ll, :] = index_to_A[i]
        # A_in = np.eye(self.in_time.size)
        # A_et = np.eye(self.execution_time.size)
        # A_fps = np.eye(self.fps.size)
        # A_req_gpu = np.eye(self.req_gpu.size)
        # A_gpu = np.eye(self.gpu.size)

        # for ii in range(len(self.gpu_status)):
        #     self.A[0][:, :, ii] = A_in
        #
        # for ii in range(len(self.in_time)):
        #     self.A[1][:, ii, :] = A_gpu

    def generate_B_to_learn(self):
        '''

        :return:
        '''
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # in_time transition when GPU is off, act on RES is decrease, and GPU is switched off.
        # [in_time (t+1), in_time(t), GPU(t), action(resolution), acton(GPU)]
        self.B[0][:, :, 0, 0, 0] = np.array([[0.1, 0.1], [0.9, 0.9]])
        # in_time transition when GPU is off, act on RES is decrease, and GPU is not touched.
        self.B[0][:, :, 0, 0, 1] = np.array([[0.2, 0.2], [0.8, 0.8]])
        # in_time transition when GPU is off, act on RES is decrease, and GPU is switched on.
        self.B[0][:, :, 0, 0, 2] = np.array([[0.3, 0.3], [0.7, 0.7]])
        # in_time transition when GPU is off, act on RES is STAY, and GPU is switched off.
        self.B[0][:, :, 0, 1, 0] = np.eye(self.num_states[0])
        # in_time transition when GPU is off, act on RES is STAY, and GPU is not touched.
        self.B[0][:, :, 0, 1, 1] = np.eye(self.num_states[0])
        # in_time transition when GPU is off, act on RES is STAY, and GPU is switched on.
        self.B[0][:, :, 0, 1, 2] = np.array([[0.1, 0.1], [0.9, 0.9]])
        # in_time transition when GPU is off, act on RES is increase, and GPU is switched off.
        self.B[0][:, :, 0, 2, 0] = np.array([[0.8, 0.8], [0.2, 0.2]])
        # in_time transition when GPU is off, act on RES is increase, and GPU is not touched.
        self.B[0][:, :, 0, 2, 1] = np.array([[0.8, 0.8], [0.2, 0.2]])
        # in_time transition when GPU is off, act on RES is increase, and GPU is switched on.
        self.B[0][:, :, 0, 2, 2] = np.array([[0.7, 0.7], [0.3, 0.3]])  # We assume stronger contribution of the GPU than RES
        # in_time transition when GPU is on, act on RES is decrease, and GPU is switched off.
        self.B[0][:, :, 1, 0, 0] = np.array([[0.7, 0.7], [0.3, 0.3]])
        # in_time transition when GPU is on, act on RES is decrease, and GPU is not touched.
        self.B[0][:, :, 1, 0, 1] = np.array([[0.2, 0.2], [0.8, 0.8]])
        # in_time transition when GPU is on, act on RES is decrease, and GPU is switched on.
        self.B[0][:, :, 1, 0, 2] = np.array([[0.1, 0.1], [0.9, 0.9]])
        # in_time transition when GPU is on, act on RES is STAY, and GPU is switched off.
        self.B[0][:, :, 1, 1, 0] = np.eye(self.num_states[0])
        # in_time transition when GPU is on, act on RES is STAY, and GPU is not touched.
        self.B[0][:, :, 1, 1, 1] = np.eye(self.num_states[0])
        # in_time transition when GPU is on, act on RES is STAY, and GPU is switched on.
        self.B[0][:, :, 1, 1, 2] = np.eye(self.num_states[0])
        # in_time transition when GPU is on, act on RES is increase, and GPU is switched off.
        self.B[0][:, :, 1, 2, 0] = np.array([[0.9, 0.9], [0.1, 0.1]])
        # in_time transition when GPU is on, act on RES is increase, and GPU is not touched.
        self.B[0][:, :, 1, 2, 1] = np.eye(self.num_states[0])
        # in_time transition when GPU is on, act on RES is increase, and GPU is switched on.
        self.B[0][:, :, 1, 2, 2] = np.eye(self.num_states[0])  # We assume stronger contribution of the GPU than RES

        # GPU transition when switch off GPU
        self.B[1][:, :, 0] = np.array([[1, 1], [0, 0]])
        # GPU transition when no touch GPU
        self.B[1][:, :, 1] = np.eye(self.num_states[1])
        # GPU transition when switch on GPU
        self.B[1][:, :, 2] = np.array([[0, 0], [1, 1]])

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
        # [in_time (t+1), in_time(t), execution_time(t), fps(t), acton(FPS)]
        self.B[0][:, :, 0, 0, 0] = np.array([[0, 0], [1, 1]])  # exec L & FPS 12 & FPS Decr
        self.B[0][:, :, 0, 0, 1] = np.array([[0, 0], [1, 1]])  # exec L & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 0, 2] = np.array([[0, 0], [1, 1]])  # exec L & FPS 12 & FPS Incr
        self.B[0][:, :, 0, 1, 0] = np.array([[0, 0], [1, 1]])  # exec L & FPS 16 & FPS Decr
        self.B[0][:, :, 0, 1, 1] = np.array([[0, 0], [1, 1]])  # exec L & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 1, 2] = np.array([[0, 0], [1, 1]])  # exec L & FPS 16 & FPS Incr
        self.B[0][:, :, 0, 2, 0] = np.array([[0, 0], [1, 1]])  # exec L & FPS 20 & FPS Decr
        self.B[0][:, :, 0, 2, 1] = np.array([[0, 0], [1, 1]])  # exec L & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 2, 2] = np.array([[0, 0], [1, 1]])  # exec L & FPS 20 & FPS Incr
        self.B[0][:, :, 0, 3, 0] = np.array([[0, 0], [1, 1]])  # exec L & FPS 26 & FPS Decr
        self.B[0][:, :, 0, 3, 1] = np.array([[0, 0], [1, 1]])  # exec L & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 3, 2] = np.array([[0, 0], [1, 1]])  # exec L & FPS 26 & FPS Incr
        self.B[0][:, :, 0, 4, 0] = np.array([[0, 0], [1, 1]])  # exec L & FPS 30 & FPS Decr
        self.B[0][:, :, 0, 4, 1] = np.array([[0, 0], [1, 1]])  # exec L & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 4, 2] = np.array([[0, 0], [1, 1]])  # exec L & FPS 30 & FPS Incr

        self.B[0][:, :, 1, 0, 0] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 12 & FPS Decr
        self.B[0][:, :, 1, 0, 1] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 0, 2] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 12 & FPS Incr
        self.B[0][:, :, 1, 1, 0] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 16 & FPS Decr
        self.B[0][:, :, 1, 1, 1] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 1, 2] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 16 & FPS Incr
        self.B[0][:, :, 1, 2, 0] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 20 & FPS Decr
        self.B[0][:, :, 1, 2, 1] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 2, 2] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 20 & FPS Incr
        self.B[0][:, :, 1, 3, 0] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 26 & FPS Decr
        self.B[0][:, :, 1, 3, 1] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 3, 2] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 26 & FPS Incr
        self.B[0][:, :, 1, 4, 0] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 30 & FPS Decr
        self.B[0][:, :, 1, 4, 1] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 4, 2] = np.array([[0, 0], [1, 1]])  # exec M-L & FPS 30 & FPS Incr

        self.B[0][:, :, 2, 0, 0] = np.array([[0, 0], [1, 1]])  # exec M & FPS 12 & FPS Decr
        self.B[0][:, :, 2, 0, 1] = np.array([[0, 0], [1, 1]])  # exec M & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 0, 2] = np.array([[0, 0], [1, 1]])  # exec M & FPS 12 & FPS Incr
        self.B[0][:, :, 2, 1, 0] = np.array([[0, 0], [1, 1]])  # exec M & FPS 16 & FPS Decr
        self.B[0][:, :, 2, 1, 1] = np.array([[0, 0], [1, 1]])  # exec M & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 1, 2] = np.array([[0, 0], [1, 1]])  # exec M & FPS 16 & FPS Incr
        self.B[0][:, :, 2, 2, 0] = np.array([[0, 0], [1, 1]])  # exec M & FPS 20 & FPS Decr
        self.B[0][:, :, 2, 2, 1] = np.array([[0, 0], [1, 1]])  # exec M & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 2, 2] = np.array([[1, 1], [0, 0]])  # exec M & FPS 20 & FPS Incr
        self.B[0][:, :, 2, 3, 0] = np.array([[0, 0], [1, 1]])  # exec M & FPS 26 & FPS Decr
        self.B[0][:, :, 2, 3, 1] = np.array([[0, 0], [1, 1]])  # exec M & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 3, 2] = np.array([[1, 1], [0, 0]])  # exec M & FPS 26 & FPS Incr
        self.B[0][:, :, 2, 4, 0] = np.eye(self.num_states[0])  # exec M & FPS 30 & FPS Decr
        self.B[0][:, :, 2, 4, 1] = np.eye(self.num_states[0])  # exec M & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 4, 2] = np.eye(self.num_states[0])  # exec M & FPS 30 & FPS Incr

        self.B[0][:, :, 3, 0, 0] = np.eye(self.num_states[0])  # exec M-H & FPS 12 & FPS Decr
        self.B[0][:, :, 3, 0, 1] = np.eye(self.num_states[0])  # exec M-H & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 0, 2] = np.array([[0, 0], [1, 1]])  # exec M-H & FPS 12 & FPS Incr
        self.B[0][:, :, 3, 1, 0] = np.array([[0, 0], [1, 1]])  # exec M-H & FPS 16 & FPS Decr
        self.B[0][:, :, 3, 1, 1] = np.eye(self.num_states[0])  # exec M-H & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 1, 2] = np.eye(self.num_states[0])  # exec M-H & FPS 16 & FPS Incr
        self.B[0][:, :, 3, 2, 0] = np.array([[0, 0], [1, 1]])  # exec M-H & FPS 20 & FPS Decr
        self.B[0][:, :, 3, 2, 1] = np.eye(self.num_states[0])  # exec M-H & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 2, 2] = np.array([[1, 1], [0, 0]])  # exec M-H & FPS 20 & FPS Incr
        self.B[0][:, :, 3, 3, 0] = np.eye(self.num_states[0])  # exec M-H & FPS 26 & FPS Decr
        self.B[0][:, :, 3, 3, 1] = np.eye(self.num_states[0])  # exec M-H & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 3, 2] = np.array([[1, 1], [0, 0]])  # exec M-H & FPS 26 & FPS Incr
        self.B[0][:, :, 3, 4, 0] = np.array([[1, 1], [0, 0]])  # exec M-H & FPS 30 & FPS Decr
        self.B[0][:, :, 3, 4, 1] = np.eye(self.num_states[0])  # exec M-H & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 4, 2] = np.eye(self.num_states[0])  # exec M-H & FPS 30 & FPS Incr

        self.B[0][:, :, 4, 0, 0] = np.eye(self.num_states[0])  # exec H & FPS 12 & FPS Decr
        self.B[0][:, :, 4, 0, 1] = np.eye(self.num_states[0])  # exec H & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 0, 2] = np.array([[1, 1], [0, 0]])  # exec H & FPS 12 & FPS Incr
        self.B[0][:, :, 4, 1, 0] = np.array([[0, 0], [1, 1]])  # exec H & FPS 16 & FPS Decr
        self.B[0][:, :, 4, 1, 1] = np.eye(self.num_states[0])  # exec H & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 1, 2] = np.array([[1, 1], [0, 0]])  # exec H & FPS 16 & FPS Incr
        self.B[0][:, :, 4, 2, 0] = np.eye(self.num_states[0])  # exec H & FPS 20 & FPS Decr
        self.B[0][:, :, 4, 2, 1] = np.eye(self.num_states[0])  # exec H & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 2, 2] = np.array([[1, 1], [0, 0]])  # exec H & FPS 20 & FPS Incr
        self.B[0][:, :, 4, 3, 0] = np.array([[1, 1], [0, 0]])  # exec H & FPS 26 & FPS Decr
        self.B[0][:, :, 4, 3, 1] = np.eye(self.num_states[0])  # exec H & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 3, 2] = np.array([[1, 1], [0, 0]])  # exec H & FPS 26 & FPS Incr
        self.B[0][:, :, 4, 4, 0] = np.array([[1, 1], [0, 0]])  # exec H & FPS 30 & FPS Decr
        self.B[0][:, :, 4, 4, 1] = np.eye(self.num_states[0])  # exec H & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 4, 2] = np.eye(self.num_states[0])  # exec H & FPS 30 & FPS Incr

        # Execution time.
        # exe (t+1), exe(t), gpu(t), act(gpu)
        self.B[1][:, :, 0, 0] = np.eye(self.num_states[1])  # GPU OFF & ACT OFF
        self.B[1][:, :, 0, 1] = np.eye(self.num_states[1])  # GPU OFF & ACT STAY
        self.B[1][:, :, 0, 2] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])  # GPU OFF & ACT ON
        self.B[1][:, :, 1, 0] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])  # GPU ON & ACT OFF
        self.B[1][:, :, 1, 1] = np.eye(self.num_states[1])
        self.B[1][:, :, 1, 2] = np.eye(self.num_states[1])

        # FPS
        # fps(t+1), fps(t), act(req_fps)
        self.B[2][:, :, 0] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[2][:, :, 1] = np.eye(self.num_states[2])
        self.B[2][:, :, 2] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])

        # GPU request
        # req_gpu(t+1), req_gpu(t), gpu(t), act(gpu)
        self.B[3][:, :, 0, 0] = np.eye(self.num_states[3])
        self.B[3][:, :, 0, 1] = np.eye(self.num_states[3])
        self.B[3][:, :, 0, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])
        self.B[3][:, :, 1, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
        self.B[3][:, :, 1, 1] = np.eye(self.num_states[3])
        self.B[3][:, :, 1, 2] = np.eye(self.num_states[3])

        # GPU
        # gpu(t+1), gpu(t), act(gpu)
        self.B[4][:, :, 0] = np.array([[1, 1], [0, 0]])
        self.B[4][:, :, 1] = np.eye(self.num_states[4])
        self.B[4][:, :, 2] = np.array([[0, 0], [1, 1]])

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
        # [in_time (t+1), in_time(t), execution_time(t), fps(t), action(FPS)]
        self.B[0][:, :, 0, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 12 & FPS Decr
        self.B[0][:, :, 0, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 12 & FPS Incr
        self.B[0][:, :, 0, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 16 & FPS Decr
        self.B[0][:, :, 0, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 16 & FPS Incr
        self.B[0][:, :, 0, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 20 & FPS Decr
        self.B[0][:, :, 0, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 20 & FPS Incr
        self.B[0][:, :, 0, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 26 & FPS Decr
        self.B[0][:, :, 0, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 26 & FPS Incr
        self.B[0][:, :, 0, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 30 & FPS Decr
        self.B[0][:, :, 0, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 0, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec L & FPS 30 & FPS Incr

        self.B[0][:, :, 1, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 12 & FPS Decr
        self.B[0][:, :, 1, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 12 & FPS Incr
        self.B[0][:, :, 1, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 16 & FPS Decr
        self.B[0][:, :, 1, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 16 & FPS Incr
        self.B[0][:, :, 1, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 20 & FPS Decr
        self.B[0][:, :, 1, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 20 & FPS Incr
        self.B[0][:, :, 1, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 26 & FPS Decr
        self.B[0][:, :, 1, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 26 & FPS Incr
        self.B[0][:, :, 1, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 30 & FPS Decr
        self.B[0][:, :, 1, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 1, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-L & FPS 30 & FPS Incr

        self.B[0][:, :, 2, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 12 & FPS Decr
        self.B[0][:, :, 2, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 12 & FPS Incr
        self.B[0][:, :, 2, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 16 & FPS Decr
        self.B[0][:, :, 2, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 16 & FPS Incr
        self.B[0][:, :, 2, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 20 & FPS Decr
        self.B[0][:, :, 2, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 20 & FPS Incr
        self.B[0][:, :, 2, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 26 & FPS Decr
        self.B[0][:, :, 2, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 26 & FPS Incr
        self.B[0][:, :, 2, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 30 & FPS Decr
        self.B[0][:, :, 2, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 2, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M & FPS 30 & FPS Incr

        self.B[0][:, :, 3, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 12 & FPS Decr
        self.B[0][:, :, 3, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 12 & FPS Incr
        self.B[0][:, :, 3, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 16 & FPS Decr
        self.B[0][:, :, 3, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 16 & FPS Incr
        self.B[0][:, :, 3, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 20 & FPS Decr
        self.B[0][:, :, 3, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 20 & FPS Incr
        self.B[0][:, :, 3, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 26 & FPS Decr
        self.B[0][:, :, 3, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 26 & FPS Incr
        self.B[0][:, :, 3, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 30 & FPS Decr
        self.B[0][:, :, 3, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 3, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec M-H & FPS 30 & FPS Incr

        self.B[0][:, :, 4, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 12 & FPS Decr
        self.B[0][:, :, 4, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 12 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 12 & FPS Incr
        self.B[0][:, :, 4, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 16 & FPS Decr
        self.B[0][:, :, 4, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 16 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 16 & FPS Incr
        self.B[0][:, :, 4, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 20 & FPS Decr
        self.B[0][:, :, 4, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 20 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 20 & FPS Incr
        self.B[0][:, :, 4, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 26 & FPS Decr
        self.B[0][:, :, 4, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 26 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 26 & FPS Incr
        self.B[0][:, :, 4, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 30 & FPS Decr
        self.B[0][:, :, 4, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 30 & FPS STAY --> This could be identity
        self.B[0][:, :, 4, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # exec H & FPS 30 & FPS Incr

        # Execution time.
        # exe (t+1), exe(t), gpu(t), act(gpu)
        self.B[1][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[1])
        self.B[1][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[1])

        # FPS
        # fps(t+1), fps(t), act(req_fps)
        self.B[2][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[2])
        self.B[2][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[2])

        # GPU request
        # req_gpu(t+1), req_gpu(t), gpu(t), act(gpu)
        self.B[3][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])

        # GPU
        # gpu(t+1), gpu(t), act(gpu)
        self.B[4][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[4])

    def generate_CD(self):
        # Vector C Goal distribution
        self.C[0] = np.array([0.1, 3])  # In time has to be TRUE
        self.C[1] = np.array([2.5, 2.5, 2, 0.25, 0.1])
        self.C[2] = np.zeros(self.num_states[2])
        self.C[3] = np.array([0.5, 2, 0.5])
        self.C[4] = np.zeros(self.num_states[4])

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
        # IMPORTANT: Learning
        pA, pB = self.generate_uniform_dirichlet_dist()
        # return Agent(A=self.A, B=self.B, C=self.C, D=self.D, policy_len=self.policy_length,
        #              num_controls=self.num_controls, B_factor_list=self.B_factor_list,
        #              B_factor_control_list=self.B_factor_control_list)

        return Agent(A=self.A, pA=pA, B=self.B, pB=pB, C=self.C, D=self.D, policy_len=self.policy_length,
                    num_controls=self.num_controls, B_factor_list=self.B_factor_list, lr_pB=1, gamma=12,
                    B_factor_control_list=self.B_factor_control_list, action_selection='deterministic', alpha=12,
                    use_param_info_gain=True, inference_algo="VANILLA")

    def observation(self, observation_list):
        state = [[False, True], ['OFF', 'ON']]
        agent_observation = list()
        for idx, obs in enumerate(observation_list):
            agent_observation.append(state[idx][:].index(obs))

        return agent_observation

    def translate_action(self, action_list):
        resolution_action = ['DECREASE', 'STAY', 'INCREASE']
        gpu_action = ['SWITCH OFF', 'STAY', 'SWITCH ON']

        action = {'resolution_demand': resolution_action[int(action_list[0])], 'gpu_action': gpu_action[int(action_list[1])]}

        return action