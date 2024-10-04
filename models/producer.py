import pymdp
import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from utils.aux_fncs import generate_normalized_2d_sq_matrix


class ProducerAgent:
    '''

    '''
    def __init__(self, b_matrix_learning=True, policy_length=1):

        self.e = np.array(['LOW', 'MID', 'HIGH'])  # Energy consumption
        self.w_req_fps = np.array(['DECREASE', 'STAY', 'INCREASE'])  # Worker satisfaction for Resolution
        self.c_req_fps = np.array(['DECREASE', 'STAY', 'INCREASE'])  # Consumer satisfaction for FPS
        self.c_req_res = np.array(['DECREASE', 'STAY', 'INCREASE'])  # Consumer satisfaction for Resolution
        self.fps = np.array(['12', '16', '20', '26', '30'])  # FPS state
        self.res = np.array(['120p', '180p', '240p', '360p', '480p', '720p'])  # Resolution state
        self.gpu = np.array(['OFF', 'ON'])  # GPU state

        self.num_states = [len(self.e), len(self.w_req_fps), len(self.c_req_fps), len(self.c_req_res),  len(self.fps), len(self.res), len(self.gpu)]
        self.num_observations = [len(self.e), len(self.w_req_fps), len(self.c_req_fps), len(self.c_req_res), len(self.fps), len(self.res), len(self.gpu)]

        # Controls
        self.u_res = np.array(['DECREASE', 'STAY', 'INCREASE'])
        self.u_fps = np.array(['DECREASE', 'STAY', 'INCREASE'])
        self.u_gpu = np.array(['OFF', 'STAY', 'ON'])


        # u_producer = np.array(['DECREASE R', 'DECREASE F', 'STAY', 'INCREASE R', 'INCREASE F'])
        self.num_controls = [len(self.u_res), len(self.u_fps), len(self.u_gpu)]
        self.num_factors = len(self.num_states)

        # Dependencies of state factors to other state factors - using indices.
        # Energy index 0 depends on res and fps index 4 and 5# IMPORTANT: dependency REMOVED
        # worker and consumer request (1,2) on resolution depends on resolution (4)
        # consumer request (3) on fps depends on fps (5)
        # FPS and RES do not depend on other states
        self.B_factor_list = [[0, 6], [1, 4], [2, 4], [3, 5], [4], [5], [6]]  # Remember always to add self state

        # Dependencies of factors wrt. actions
        # Energy index 0 is controlled by FPS (1) and RES (0)
        # worker and consumer request (1,2) is control by RES (0)
        # consumer request (3) is controlled by FPS (1)
        # RES is controlled by RES and FPS by FPS
        self.B_factor_control_list = [[2], [0], [0], [1], [0], [1], [2]]  # TODO: Check! order of FPS and RES

        A_shapes = [[o_dim] + self.num_states for o_dim in self.num_observations]
        # print(A_shapes)
        self.A = utils.obj_array_zeros(A_shapes)
        self.B = utils.obj_array(self.num_factors)
        self.C = utils.obj_array_zeros(self.num_observations)
        self.D = utils.obj_array_zeros(self.num_states)
        self.policy_length = policy_length
        self.b_matrix_learning = b_matrix_learning

    def generate_A(self):
        # We observe the state as an identity
        A_matrices = [np.eye(self.e.size), np.eye(self.w_req_fps.size), np.eye(self.c_req_fps.size),
                      np.eye(self.c_req_res.size), np.eye(self.fps.size), np.eye(self.res.size), np.eye(self.gpu.size)]
        index_to_A = {0: A_matrices[0], 1: A_matrices[1], 2: A_matrices[2], 3: A_matrices[3], 4: A_matrices[4],
                      5: A_matrices[5], 6: A_matrices[6]}

        ranges = [len(self.e), len(self.w_req_fps), len(self.c_req_fps), len(self.c_req_res), len(self.fps),
                  len(self.res), len(self.gpu)]

        for i in range(len(ranges)):
            for ii in range(ranges[0]):
                for jj in range(ranges[1]):
                    for kk in range(ranges[2]):
                        for ll in range(ranges[3]):
                            for mm in range(ranges[4]):
                                for nn in range(ranges[5]):
                                    for oo in range(ranges[6]):
                                        if i == 0:
                                            self.A[i][:, :, jj, kk, ll, mm, nn, oo] = index_to_A[i]
                                        if i == 1:
                                            self.A[i][:, ii, :, kk, ll, mm, nn, oo] = index_to_A[i]
                                        if i == 2:
                                            self.A[i][:, ii, jj, :, ll, mm, nn, oo] = index_to_A[i]
                                        if i == 3:
                                            self.A[i][:, ii, jj, kk, :, mm, nn, oo] = index_to_A[i]
                                        if i == 4:
                                            self.A[i][:, ii, jj, kk, ll, :, nn, oo] = index_to_A[i]
                                        if i == 5:
                                            self.A[i][:, ii, jj, kk, ll, mm, :, oo] = index_to_A[i]
                                        if i == 6:
                                            self.A[i][:, ii, jj, kk, ll, mm, nn, :] = index_to_A[i]


    def generate_B(self):
        # ACTIONS -
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # DEFINITION OF B MATRIX FOR ENERGY TRANSITION [energy(t+1), energy(t), GPU(t), action(GPU)]
        self.B[0][:, :, 0, 0] = np.eye(self.num_states[0])                      #  GPU OFF & ACTION OFF - Energy stays
        self.B[0][:, :, 0, 1] = np.eye(self.num_states[0])                      #  GPU OFF & ACTION STAY - Energy stays
        self.B[0][:, :, 0, 2] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # GPU OFF & ACTION ON - Energy increases
        self.B[0][:, :, 1, 0] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # GPU ON & ACTION OFF - Energy decreases
        self.B[0][:, :, 1, 1] = np.eye(self.num_states[0])                      # GPU ON & ACTION STAY - Energy stays
        self.B[0][:, :, 1, 2] = np.eye(self.num_states[0])                      # GPU ON & ACTION ON - Energy stays

        # DEFINITION OF B MATRIX FOR WORKER AND CONSUMER REQUEST ON FPS [worker_req(t+1), worker_req(t), FPS(t), action(FPS)]
        for idx in [1, 2]:
            self.B[idx][:, :, 0, 0] = [[0, 0, 0], [1, 1 , 0], [0, 0, 1]]                         # FPS 12 & Decr - identity
            self.B[idx][:, :, 0, 1] = np.eye(self.num_states[1])                          # FPS 12 & Stay - identity
            self.B[idx][:, :, 0, 2] = np.array([[0, 1, 0], [1, 0 , 1], [0, 0, 0]])     # FPS 12 & Incr
            self.B[idx][:, :, 1, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])     # FPS 16 & Decr
            self.B[idx][:, :, 1, 1] = np.eye(self.num_states[1])                          # FPS 16 & Stay - identity
            self.B[idx][:, :, 1, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])     # FPS 16 & Incr
            self.B[idx][:, :, 2, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])     # FPS 20 & Decr
            self.B[idx][:, :, 2, 1] = np.eye(self.num_states[1])                          # FPS 20 & Stay
            self.B[idx][:, :, 2, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])     # FPS 20 & Incr
            self.B[idx][:, :, 3, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])     # FPS 26 & Decr
            self.B[idx][:, :, 3, 1] = np.eye(self.num_states[1])                          # FPS 26 & Stay
            self.B[idx][:, :, 3, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])     # FPS 26 & Incr
            self.B[idx][:, :, 4, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])     # FPS 30 & Decr
            self.B[idx][:, :, 4, 1] = np.eye(self.num_states[1])                          # FPS 30 & Stay
            self.B[idx][:, :, 4, 2] = np.array([[1, 0, 0], [0, 1 , 1], [0, 0, 0]])     # FPS 30 & Incr


        # DEFINITION OF B MATRIX FOR CONSUMER REQUEST ON RESOLUTION TRANSITION - [consumer_req(t+1), consumer_req(t), resolution(t), action(res)]
        self.B[3][:, :, 0, 0] = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]])     # RES min & Decr
        self.B[3][:, :, 0, 1] = np.eye(self.num_states[2])                      # RES min & stay
        self.B[3][:, :, 0, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])    # RES min & Incr
        self.B[3][:, :, 1, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])    # Res 180p & Decr
        self.B[3][:, :, 1, 1] = np.eye(self.num_states[2])                      # Res 180p & Stay
        self.B[3][:, :, 1, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])    # RES 180p & Incr
        self.B[3][:, :, 2, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])    # Res 240p & Decr
        self.B[3][:, :, 2, 1] = np.eye(self.num_states[2])                      # Res 240p & Stay
        self.B[3][:, :, 2, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])    # RES 240p & Incr
        self.B[3][:, :, 3, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])    # Res 360p & Decr
        self.B[3][:, :, 3, 1] = np.eye(self.num_states[2])                      # Res 360p & Stay
        self.B[3][:, :, 3, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])    # RES 360p & Incr
        self.B[3][:, :, 4, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])    # Res 480p & Decr
        self.B[3][:, :, 4, 1] = np.eye(self.num_states[2])                      # Res 480p & Stay
        self.B[3][:, :, 4, 2] = np.array([[1, 1, 0], [0, 0 , 1], [0, 0, 0]])    # RES 480p & Incr
        self.B[3][:, :, 5, 0] = np.array([[0, 0, 0], [1, 0 , 0], [0, 1, 1]])    # Res 720p & Decr
        self.B[3][:, :, 5, 1] = np.eye(self.num_states[2])                      # Res 720p & Stay
        self.B[3][:, :, 5, 2] = np.array([[1, 0, 0], [ 0, 1 , 1], [0, 0, 0]])   # Res 720p & Decr

        # DEFINITION OF B MATRIX FOR FPS change on FPS.
        # FPS decreases by 1 unit if it can
        self.B[4][:, :, 0] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[4][:, :, 1] = np.eye(self.num_states[4])  # FPS stays the same if it is not changed.
        # FPS decreases by 1 unit if it can
        self.B[4][:, :, 2] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])

        # DEFINITION OF B MATRIX FOR Resolution change on Resolution
        self.B[5][:, :, 0] = np.array(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0]])  # RESOLUTION decreases by 1 unit if it can
        self.B[5][:, :, 1] = np.eye(self.num_states[5])  # RESOLUTION stays the same if it is not changed.
        self.B[5][:, :, 2] = np.array(
            [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 1]])  # RESOLUTION increases by 1 unit if it can

        # DEFINITION OF B MATRIX FOR GPU status. GPU (t+1), GPU(t), Control(demand on GPU)
        self.B[6][:, :, 0] = np.array([[1, 1],[0, 0]])
        self.B[6][:, :, 1] = np.eye(self.num_states[6])
        self.B[6][:, :, 2] = np.array([[0, 0], [1, 1]])

    def generate_B_unknown(self):
        # ACTIONS -
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if
                             i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # DEFINITION OF B MATRIX FOR ENERGY TRANSITION [energy(t+1), energy(t), GPU(t), action(GPU)]
        self.B[0][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # GPU OFF & ACTION OFF - Energy stays
        self.B[0][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # GPU OFF & ACTION STAY - Energy stays
        self.B[0][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # GPU OFF & ACTION ON - Energy increases
        self.B[0][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[0])  # GPU ON & ACTION OFF - Energy decreases
        self.B[0][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[0])  # GPU ON & ACTION STAY - Energy stays
        self.B[0][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[0])  # GPU ON & ACTION ON - Energy stays

        # DEFINITION OF B MATRIX FOR WORKER AND CONSUMER REQUEST ON FPS [worker_req(t+1), worker_req(t), FPS(t), action(FPS)]
        for idx in [1, 2]:
            self.B[idx][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[idx])
            self.B[idx][:, :, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[idx])

        # DEFINITION OF B MATRIX FOR CONSUMER REQUEST ON RESOLUTION TRANSITION - [consumer_req(t+1), consumer_req(t), resolution(t), action(res)]
        self.B[3][:, :, 0, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 0, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 0, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 1, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 2, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 2, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 2, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 3, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 3, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 3, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 4, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 4, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 4, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 5, 0] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 5, 1] = generate_normalized_2d_sq_matrix(self.num_states[3])
        self.B[3][:, :, 5, 2] = generate_normalized_2d_sq_matrix(self.num_states[3])

        # DEFINITION OF B MATRIX FOR FPS change on FPS.
        # FPS decreases by 1 unit if it can
        self.B[4][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[4])
        self.B[4][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[4])
        # FPS decreases by 1 unit if it can
        self.B[4][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[4])

        # DEFINITION OF B MATRIX FOR Resolution change on Resolution
        self.B[5][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[5])
        self.B[5][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[5])
        self.B[5][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[5])

        # DEFINITION OF B MATRIX FOR GPU status. GPU (t+1), GPU(t), Control(demand on GPU)
        self.B[6][:, :, 0] = generate_normalized_2d_sq_matrix(self.num_states[6])
        self.B[6][:, :, 1] = generate_normalized_2d_sq_matrix(self.num_states[6])
        self.B[6][:, :, 2] = generate_normalized_2d_sq_matrix(self.num_states[6])

    def generate_B_not_in_use(self):
        # ACTIONS -
        for factor in range(self.num_factors):
            lagging_shape = [ns for i, ns in enumerate(self.num_states) if i in self.B_factor_list[factor]]
            control_shape = [na for i, na in enumerate(self.num_controls) if i in self.B_factor_control_list[factor]]
            factor_shape = [self.num_states[factor]] + lagging_shape + control_shape
            self.B[factor] = np.zeros(factor_shape)

        # DEFINITION OF B MATRIX FOR ENERGY TRANSITION [energy(t+1), energy(t), action(resolution), action(FPS)]
        self.B[0][:, :, 0, 0] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])             # DECREASE ENERGY
        self.B[0][:, :, 0, 1] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # DECREASE ENERGY (Probable)
        self.B[0][:, :, 0, 2] = np.eye(self.num_states[0])                              # Identity - assume same effect RES and FPS
        self.B[0][:, :, 1, 0] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # DECREASE ENERGY (Probable)
        self.B[0][:, :, 1, 1] = np.eye(self.num_states[0])                              # Identity
        self.B[0][:, :, 1, 2] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # INCREASE ENERGY (Probable)
        self.B[0][:, :, 2, 0] = np.eye(self.num_states[0])                              # Identity - assume same effect RES and FPS
        self.B[0][:, :, 2, 1] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # INCREASE ENERGY (Probable)
        self.B[0][:, :, 2, 2] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])             # INCREASE ENERGY

        # DEFINITION OF B MATRIX FOR WORKER REQUEST ON RESOLUTION TRANSITION [worker_req(t+1), worker_req(t), resolution(t), action(res)]
        self.B[1][:, :, 0, 0] = np.eye(self.num_states[1])                          # IF resolution is at its min it can not be decreased.
        self.B[1][:, :, 0, 1] = np.eye(self.num_states[1])                          # IF resolution is not moving, all stays the same.
        self.B[1][:, :, 0, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[1][:, :, 1, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[1][:, :, 1, 1] = np.eye(self.num_states[1])                          # IF resolution is not moving, all stays the same.
        self.B[1][:, :, 1, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[1][:, :, 2, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[1][:, :, 2, 1] = np.eye(self.num_states[1])                          # IF resolution is not moving, all stays the same.
        self.B[1][:, :, 2, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[1][:, :, 3, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[1][:, :, 3, 1] = np.eye(self.num_states[1])                          # IF resolution is not moving, all stays the same.
        self.B[1][:, :, 3, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[1][:, :, 4, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[1][:, :, 4, 1] = np.eye(self.num_states[1])                          # IF resolution is not moving, all stays the same.
        self.B[1][:, :, 4, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[1][:, :, 5, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[1][:, :, 5, 1] = np.eye(self.num_states[1])                          # IF resolution is not moving, all stays the same.
        self.B[1][:, :, 5, 2] = np.eye(self.num_states[1])                          # If Resolution is at its max ir can not be increased.

        # DEFINITION OF B MATRIX FOR CONSUMER REQUEST ON RESOLUTION TRANSITION - SAME AS PREVIOUS MATRICES [consumer_req(t+1), consumer_req(t), resolution(t), action(res)]
        self.B[2][:, :, 0, 0] = np.eye(self.num_states[2])                          # IF resolution is at its min it can not be decreased.
        self.B[2][:, :, 0, 1] = np.eye(self.num_states[2])                          # IF resolution is not moving, all stays the same.
        self.B[2][:, :, 0, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[2][:, :, 1, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[2][:, :, 1, 1] = np.eye(self.num_states[2])                          # IF resolution is not moving, all stays the same.
        self.B[2][:, :, 1, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[2][:, :, 2, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[2][:, :, 2, 1] = np.eye(self.num_states[2])                          # IF resolution is not moving, all stays the same.
        self.B[2][:, :, 2, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[2][:, :, 3, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[2][:, :, 3, 1] = np.eye(self.num_states[2])                          # IF resolution is not moving, all stays the same.
        self.B[2][:, :, 3, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[2][:, :, 4, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[2][:, :, 4, 1] = np.eye(self.num_states[2])                          # IF resolution is not moving, all stays the same.
        self.B[2][:, :, 4, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE Resolution
        self.B[2][:, :, 5, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE Resolution
        self.B[2][:, :, 5, 1] = np.eye(self.num_states[2])                          # IF resolution is not moving, all stays the same.
        self.B[2][:, :, 5, 2] = np.eye(self.num_states[2])                          # If Resolution is at its max ir can not be increased.

        # DEFINITION OF B MATRIX FOR CONSUMER REQUEST ON FPS TRANSITION [consumer req(t+1), consumer_req(t), fps(t), action(fps)]
        self.B[3][:, :, 0, 0] = np.eye(self.num_states[3])                          # IF FPS is at its min it can not be decreased.
        self.B[3][:, :, 0, 1] = np.eye(self.num_states[3])                          # IF FPS is not moving, all stays the same.
        self.B[3][:, :, 0, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE FPS
        self.B[3][:, :, 1, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE FPS
        self.B[3][:, :, 1, 1] = np.eye(self.num_states[3])                          # IF FPS is not moving, all stays the same.
        self.B[3][:, :, 1, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE FPS
        self.B[3][:, :, 2, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE FPS
        self.B[3][:, :, 2, 1] = np.eye(self.num_states[3])                          # IF FPS is not moving, all stays the same.
        self.B[3][:, :, 2, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # FPS Resolution
        self.B[3][:, :, 3, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # FPS Resolution
        self.B[3][:, :, 3, 1] = np.eye(self.num_states[3])                          # IF FPS is not moving, all stays the same.
        self.B[3][:, :, 3, 2] = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])     # INCREASE FPS
        self.B[3][:, :, 4, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])     # DECREASE FPS
        self.B[3][:, :, 4, 1] = np.eye(self.num_states[3])                          # IF FPS is not moving, all stays the same.
        self.B[3][:, :, 4, 2] = np.eye(self.num_states[3])                          # If FPS is at its max ir can not be increased.

        # DEFINITION OF B MATRIX FOR Resolution change on Resolution
        self.B[4][:, :, 0] = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]])  # RESOLUTION decreases by 1 unit if it can
        self.B[4][:, :, 1] = np.eye(self.num_states[4])  # RESOLUTION stays the same if it is not changed.
        self.B[4][:, :, 2] = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]])  # RESOLUTION increases by 1 unit if it can

        # DEFINITION OF B MATRIX FOR FPS change on FPS.
        # FPS decreases by 1 unit if it can
        self.B[5][:, :, 0] = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0]])
        self.B[5][:, :, 1] = np.eye(self.num_states[5])  # FPS stays the same if it is not changed.
        # FPS decreases by 1 unit if it can
        self.B[5][:, :, 2] = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]])

    def generate_CD(self):
        # Vector C GOAL Distribution
        self.C[0] = np.array([2, 1.5, 0.1])  # Low energy is priority
        self.C[1] = np.array([0.25, 2, 0.25])  # Worker satisfaction is priority (FPS)
        self.C[2] = np.array([0.5, 2, 0.75])  # Consumer (fps) satisfaction is priority
        self.C[3] = np.array([0.5, 2, 0.75])  # Consumer (res) satisfaction is priority
        self.C[4] = np.array([0, 0, 0, 0, 0])  # No specific priority for FPS
        self.C[5] = np.array([0, 0, 0, 0, 0, 0])  # No specific priority for RES
        self.C[6] = np.array([0, 0])  # No specific priority for GPU

        # Vector D - Prior state distribution - UNKNOWN (uniformly distributed)
        D = [np.ones(ns) / ns for ns in self.num_states]
        for idx, arr in enumerate(D):
            self.D[idx] = arr
        # self.D = np.array(D)

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
        state = [['LOW', 'MID', 'HIGH'], ['DECREASE', 'STAY', 'INCREASE'], ['DECREASE', 'STAY', 'INCREASE'],
                 ['DECREASE', 'STAY', 'INCREASE'], ['120p', '180p', '240p', '360p', '480p', '720p'],
                 [12, 16, 20, 26, 30]]

        # agent_observation = utils.obj_array(len(observation_list))
        agent_observation = list()
        # agent_observation = np.zeros(shape=len(observation_list), dtype=int)
        for idx, obs in enumerate(observation_list):
            agent_observation.append(state[idx][:].index(obs))

        return agent_observation

    def translate_action(self, action_list):
        resolution_action = ['DECREASE', 'STAY', 'INCREASE']
        fps_action = ['DECREASE', 'STAY', 'INCREASE']
        gpu_action = ['OFF', 'STAY', 'ON']

        action = {'resolution': resolution_action[int(action_list[0])], 'fps': fps_action[int(action_list[1])], 'gpu': gpu_action[int(action_list[2])]}

        return action
