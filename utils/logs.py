import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import itertools

class Logger:
    def __init__(self, debug=False):

        self.debug = debug
        self.efe = dict()
        self.efe['producer'] = dict()
        self.efe['producer']['min_efe'] = list()
        self.efe['producer']['efe'] = list()
        self.efe['producer']['expected_utility'] = list()  #  key is 'r'
        self.efe['producer']['ig_states'] = list()  #  key is 'ig_s' --> ig: information_gain
        self.efe['producer']['ig_pB'] = list()
        self.efe['producer']['ig_pA'] = list()
        self.efe['producer']['policy_beliefs'] = list()
        self.efe['worker'] = dict()
        self.efe['worker']['efe'] = list()
        self.efe['worker']['min_efe'] = list()
        self.efe['worker']['expected_utility'] = list()  # key is 'r'
        self.efe['worker']['ig_states'] = list()  # key is 'ig_s' --> ig: information_gain
        self.efe['worker']['ig_pB'] = list()
        self.efe['worker']['ig_pA'] = list()
        self.efe['worker']['policy_beliefs'] = list()
        self.efe['consumer'] = dict()
        self.efe['consumer']['efe'] = list()
        self.efe['consumer']['min_efe'] = list()
        self.efe['consumer']['expected_utility'] = list()  # key is 'r'
        self.efe['consumer']['ig_states'] = list()  # key is 'ig_s' --> ig: information_gain
        self.efe['consumer']['ig_pB'] = list()
        self.efe['consumer']['ig_pA'] = list()
        self.efe['consumer']['policy_beliefs'] = list()

        self.action = dict()
        self.action['producer'] = dict()
        self.action['producer']['res'] = list()
        self.action['producer']['fps'] = list()
        self.action['producer']['gpu'] = list()
        self.action['worker'] = dict()
        self.action['worker']['fps'] = list()
        self.action['worker']['gpu'] = list()
        self.action['consumer'] = dict()
        self.action['consumer']['fps'] = list()
        self.action['consumer']['res'] = list()

        self.state = dict()
        self.state['producer'] = dict()
        self.state['producer']['energy'] = list()
        self.state['producer']['w_req_fps'] = list()
        self.state['producer']['c_req_fps'] = list()
        self.state['producer']['c_req_res'] = list()
        self.state['producer']['fps'] = list()
        self.state['producer']['res'] = list()
        self.state['producer']['gpu'] = list()
        self.state['worker'] = dict()
        self.state['worker']['in_time'] = list()
        self.state['worker']['exe_time'] = list()
        self.state['worker']['fps'] = list()
        self.state['worker']['req_gpu'] = list()
        self.state['worker']['gpu'] = list()
        self.state['consumer'] = dict()
        self.state['consumer']['success'] = list()
        self.state['consumer']['distance'] = list()
        self.state['consumer']['fps'] = list()
        self.state['consumer']['res'] = list()

        self.obs = dict()
        self.obs['producer'] = dict()
        self.obs['producer']['energy'] = list()
        self.obs['producer']['w_req_fps'] = list()
        self.obs['producer']['c_req_fps'] = list()
        self.obs['producer']['c_req_res'] = list()
        self.obs['producer']['fps'] = list()
        self.obs['producer']['res'] = list()
        self.obs['producer']['gpu'] = list()
        self.obs['worker'] = dict()
        self.obs['worker']['in_time'] = list()
        self.obs['worker']['exe_time'] = list()
        self.obs['worker']['fps'] = list()
        self.obs['worker']['req_gpu'] = list()
        self.obs['worker']['gpu'] = list()
        self.obs['consumer'] = dict()
        self.obs['consumer']['success'] = list()
        self.obs['consumer']['distance'] = list()
        self.obs['consumer']['fps'] = list()
        self.obs['consumer']['res'] = list()

        self.slos = dict()
        self.slos['energy'] = list()
        self.slos['in_time'] = list()
        self.slos['success'] = list()
        self.slos['distance'] = list()

        self.e = ['LOW', 'MID', 'HIGH']  # Energy consumption
        self.w_req_fps = ['DECREASE', 'STAY', 'INCREASE']  # Worker satisfaction for Resolution
        self.c_req_fps = ['DECREASE', 'STAY', 'INCREASE']  # Consumer satisfaction for FPS
        self.c_req_res = ['DECREASE', 'STAY', 'INCREASE']  # Consumer satisfaction for Resolution
        self.fps = ['12', '16', '20', '26', '30']  # FPS state
        self.res = ['120p', '180p', '240p', '360p', '480p', '720p']  # Resolution state

        self.in_time = np.array([False, True])
        self.execution_time = np.array(["LOW", "MID-LOW", "MID", "MID-HIGH", "HIGH"])
        self.req_gpu = np.array(["OFF", "STAY", "ON"])
        self.gpu = np.array(["OFF", "ON"])

        self.success = np.array([False, True])
        self.distance = np.array(["SHORT", "MID-SHORT", "MID", "MID-LONG", "LONG"])

        self.u_res = np.array(['DECREASE', 'STAY', 'INCREASE'])
        self.u_fps = np.array(['DECREASE', 'STAY', 'INCREASE'])
        self.u_gpu = np.array(['OFF', 'STAY', 'ON'])
        self.u_req_fps = ["DECREASE", "STAY", "INCREASE"]
        self.u_gpu = ["SWITCH_OFF", "STAY", "SWITCH_ON"]

    def append_efe(self, key, policy_beliefs, neg_efe, dict_efe):
        self.efe[key]['min_efe'].append(min(neg_efe))
        self.efe[key]['efe'].append(neg_efe)
        self.efe[key]['expected_utility'].append(dict_efe['r'])
        self.efe[key]['ig_states'].append(dict_efe['ig_s'])
        self.efe[key]['ig_pB'].append(dict_efe['ig_pB'])
        self.efe[key]['ig_pA'].append(dict_efe['ig_pA'])
        self.efe[key]['policy_beliefs'].append(policy_beliefs)

    def append_action(self, key, value):
        if key == 'producer':
            self.action[key]['res'].append(int(value[0]))
            self.action[key]['fps'].append(int(value[1]))
            self.action[key]['gpu'].append(int(value[2]))
            if self.debug:
                print('---- Producer actions ----')
                print('Resolution ' +str(self.u_res[self.action[key]['res'][-1]]))
                print('FPS ' + str(self.u_fps[self.action[key]['fps'][-1]]))
                print('GPU request: ' + str(self.u_gpu[self.action[key]['gpu'][-1]]))
        if key == 'worker':
            self.action[key]['fps'].append(int(value[0]))
            self.action[key]['gpu'].append(int(value[1]))
            if self.debug:
                print('---- Worker actions ----')
                print('FPS request: ' + str(self.u_req_fps[self.action[key]['fps'][-1]]))
                print('GPU : ' + str(self.u_gpu[self.action[key]['gpu'][-1]]))
        if key == 'consumer':
            self.action[key]['fps'].append(int(value[0]))
            self.action[key]['res'].append(int(value[1]))
            if self.debug:
                print('---- Consumer actions ----')
                print('FPS request: ' + str(self.u_fps[self.action[key]['fps'][-1]]))
                print('Resolution request: ' + str(self.u_res[self.action[key]['res'][-1]]))

    def append_state(self, key, value):
        if key == 'producer':
            self.state[key]['energy'].append(np.argmax(value[0]))
            self.state[key]['w_req_fps'].append(np.argmax(value[1]))
            self.state[key]['c_req_fps'].append(np.argmax(value[2]))
            self.state[key]['c_req_res'].append(np.argmax(value[3]))
            self.state[key]['fps'].append(np.argmax(value[4]))
            self.state[key]['res'].append(np.argmax(value[5]))
            self.state[key]['gpu'].append(np.argmax(value[6]))
            if self.debug:
                print('---- Producer states ----')
                print('Energy state: ' + str(self.e[self.state[key]['energy'][-1]]))
                print('Worker request FPS state: ' + str(self.w_req_fps[self.state[key]['w_req_fps'][-1]]))
                print('Consumer request FPS state: ' + str(self.c_req_fps[self.state[key]['c_req_fps'][-1]]))
                print('Consumer request Res state: ' + str(self.c_req_res[self.state[key]['c_req_res'][-1]]))
                print('FPS state: ' + str(self.fps[self.state[key]['fps'][-1]]))
                print('Res state: ' + str(self.res[self.state[key]['res'][-1]]))
        if key == 'worker':
            self.state[key]['in_time'].append(np.argmax(value[0]))
            self.state[key]['exe_time'].append(np.argmax(value[1]))
            self.state[key]['fps'].append(np.argmax(value[2]))
            self.state[key]['req_gpu'].append(np.argmax(value[3]))
            self.state[key]['gpu'].append(np.argmax(value[4]))
            if self.debug:
                print('---- Worker states ----')
                print('In time state: ' + str(self.in_time[self.state[key]['in_time'][-1]]))
                print('Exec time state: ' + str(self.execution_time[self.state[key]['exe_time'][-1]]))
                print('Producer request GPU: ' + str(self.req_gpu[self.state[key]['req_gpu'][-1]]))
                print('GPU state: ' + str(self.gpu[self.state[key]['gpu'][-1]]))
        if key == 'consumer':
            self.state[key]['success'].append(np.argmax(value[0]))
            self.state[key]['distance'].append(np.argmax(value[1]))
            self.state[key]['fps'].append(np.argmax(value[2]))
            self.state[key]['res'].append(np.argmax(value[3]))
            if self.debug:
                print('---- Consumer states ----')
                print('Success state: ' + str(self.success[self.state[key]['success'][-1]]))
                print('Distance state: ' + str(self.distance[self.state[key]['distance'][-1]]))
        # FIXME: Now we compute SLO each step, we can use a running average to reduce computation needs.

    def append_obs(self, key, value):
        if key == 'producer':
            if self.debug:
                print('---- Producer obs ----')
                print('Energy observation: ' + str(value[0]))
                print('W FPS Request observation: ' + str(value[1]))
                print('C FPS Request observation: ' + str(value[2]))
                print('C RES Request observation: ' + str(value[3]))
                print('FPS observation: ' + str(value[4]))
                print('Res observation: ' + str(value[5]))
                print('GPU observation: ' + str(value[6]))
            # self.obs[key]['energy'].append(value[0])
            # self.obs[key]['w_req_fps'].append(value[1])
            # self.obs[key]['c_req_fps'].append(value[2])
            # self.obs[key]['c_req_res'].append(value[3])
            # self.obs[key]['fps'].append(value[4])
            # self.obs[key]['res'].append(value[5])
            # self.obs[key]['gpu'].append(value[6])
        if key == 'worker':
            if self.debug:
                print('---- Worker obs ----')
                print('In time observation: ' + str(value[0]))
                print('Exec time observation: ' + str(value[1]))
                print('FPS observation: ' + str(value[2]))
                print('P GPU Request observation: ' + str(value[3]))
                print('GPU observation: ' + str(value[4]))
            # self.obs[key]['in_time'].append(value[0])
            # self.obs[key]['exe_time'].append(value[1])
            # self.obs[key]['fps'].append(value[2])
            # self.obs[key]['req_gpu'].append(value[3])
            # self.obs[key]['gpu'].append(value[4])
        if key == 'consumer':
            if self.debug:
                print('---- Consumer obs ----')
                print('Success observation: ' + str(value[0]))
                print('Distance observation: ' + str(value[1]))
                print('FPS observation: ' + str(value[2]))
                print('RES observation: ' + str(value[3]))
            # self.obs[key]['success'].append(value[0])
            # self.obs[key]['distance'].append(value[1])
            # self.obs[key]['fps'].append(value[2])
            # self.obs[key]['res'].append(value[3])

    def compute_slos(self):
        # Energy SLO
        count_energy = 0
        for val in self.state['producer']['energy']:
            if val == 0 or val == 1:
                count_energy += 1
        self.slos['energy'].append(count_energy/len(self.state['producer']['energy']))
        print('Current energy SLO fulfillment: ' + str(self.slos['energy'][-1]))
        # In time SLO
        count_in_time = 0
        for val in self.state['worker']['in_time']:
            if val == 1:
                count_in_time += 1
        self.slos['in_time'].append(count_in_time/len(self.state['worker']['in_time']))
        print('Current in_time SLO fulfillment: ' + str(self.slos['in_time'][-1]))
        # Success SLO
        count_success = 0
        for val in self.state['consumer']['success']:
            if val == 1:
                count_success += 1
        self.slos['success'].append(count_success/len(self.state['consumer']['success']))
        print('Current success SLO fulfillment: ' + str(self.slos['success'][-1]))
        count_distance = 0
        for val in self.state['consumer']['distance']:
            if val < 3:
                count_distance += 1
        self.slos['distance'].append(count_distance/len(self.state['consumer']['distance']))
        print('Current distance SLO fulfillment: ' + str(self.slos['distance'][-1]))

    def print(self, iteration_id='test1'):
        # EFE
        plt.figure(figsize=(10,6))
        plt.plot(self.efe['producer']['min_efe'], label='producer', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.efe['worker']['min_efe'], label='worker', color='red', linestyle='--', linewidth=2)
        plt.plot(self.efe['consumer']['min_efe'], label='consumer', color='green', linestyle=':', linewidth=2)
        plt.title('Expected Free Energy', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('EFE', fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id+'-EFE.png',dpi=300, bbox_inches='tight')
        plt.close()

        # Producer EFE per policy
        plt.figure(figsize=(10,6))
        matrix = np.array(self.efe['producer']['efe'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=3))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Expected Free Energy per policy')
        plt.title('EFE Heatmap for Producer')
        plt.savefig('results/figures/' + iteration_id+'-EFE-producer.png',dpi=300, bbox_inches='tight')
        plt.close()
        # Worker EFE per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['worker']['efe'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=2))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Expected Free Energy per policy')
        plt.title('EFE Heatmap for Worker')
        plt.savefig('results/figures/' + iteration_id + '-EFE-worker.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Worker EFE per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['consumer']['efe'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=2))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Expected Free Energy per policy')
        plt.title('EFE Heatmap for Consumer')
        plt.savefig('results/figures/' + iteration_id + '-EFE-consumer.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Producer Utility per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['producer']['expected_utility'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=3))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Utility per policy')
        plt.title('Pragmatic value Heatmap for Producer')
        plt.savefig('results/figures/' + iteration_id + '-PV-producer.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Worker EFE per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['worker']['expected_utility'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=2))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Utility per policy')
        plt.title('Pragmatic value Heatmap for Worker')
        plt.savefig('results/figures/' + iteration_id + '-PV-worker.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Worker EFE per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['consumer']['expected_utility'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=2))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Utility per policy')
        plt.title('Pragmatic value Heatmap for Consumer')
        plt.savefig('results/figures/' + iteration_id + '-PV-consumer.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Producer Utility per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['producer']['ig_pB'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=3))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Information gain per policy')
        plt.title('Information gain Heatmap for Producer')
        plt.savefig('results/figures/' + iteration_id + '-IG-producer.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Worker EFE per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['worker']['ig_pB'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=2))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Information gain per policy')
        plt.title('Information gain Heatmap for Worker')
        plt.savefig('results/figures/' + iteration_id + '-IG-worker.png', dpi=300, bbox_inches='tight')
        plt.close()
        # Worker EFE per policy
        plt.figure(figsize=(10, 6))
        matrix = np.array(self.efe['consumer']['ig_pB'])
        matrix = matrix.T
        vector_elements = [0, 1, 2]
        row_names = list(itertools.product(vector_elements, repeat=2))
        sns.heatmap(matrix, cmap='viridis', cbar=True, xticklabels=np.arange(matrix.shape[1]), yticklabels=row_names)
        # Set axis labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Information gain per policy')
        plt.title('Information gain Heatmap for Consumer')
        plt.savefig('results/figures/' + iteration_id + '-IG-consumer.png', dpi=300, bbox_inches='tight')
        plt.close()


        # ACTIONS - Producer
        plt.figure(figsize=(10,6))
        plt.plot(self.action['producer']['res'], label='Resolution', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.action['producer']['fps'], label='FPS', color='red', linestyle='--', linewidth=2)
        plt.title('Producer action on Resolution and FPS', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Action on FPS/Resolution', fontsize=14)
        plt.yticks([0,1,2], ['DECREASE', 'KEEP', 'INCREASE'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[producer-actions]-FPS_RES.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(self.action['producer']['gpu'], label='GPU', color='blue', linestyle='-', linewidth=2)
        plt.title('Producer demand on GPU', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Demand on GPU', fontsize=14)
        plt.yticks([0, 1, 2], ['SWITCH OFF', 'KEEP', 'SWITCH ON'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[producer-actions]-GPU.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ACTIONS - Worker
        plt.figure(figsize=(10,6))
        plt.plot(self.action['worker']['fps'], label='FPS', color='blue', linestyle='-', linewidth=2)
        plt.title('Worker demand on FPS', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Demand on FPS', fontsize=14)
        plt.yticks([0, 1, 2], ['DECREASE', 'KEEP', 'INCREASE'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[worker-actions]-FPS.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(self.action['worker']['gpu'], label='GPU', color='blue', linestyle='-', linewidth=2)
        plt.title('Worker action on GPU', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Action on GPU', fontsize=14)
        plt.yticks([0, 1, 2], ['SWITCH OFF', 'KEEP', 'SWITCH ON'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[worker-actions]-GPU.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ACTIONS - Consumer
        plt.figure(figsize=(10, 6))
        plt.plot(self.action['consumer']['res'], label='Resolution', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.action['consumer']['fps'], label='FPS', color='red', linestyle='--', linewidth=2)
        plt.title('Consumer demand on RESOLUTION and FPS', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Demand on FPS/RESOLUTION', fontsize=14)
        plt.yticks([0, 1, 2], ['DECREASE', 'KEEP', 'INCREASE'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[consumer-actions]-FPS_RES.png', dpi=300, bbox_inches='tight')
        plt.close()

        # STATE - Producer
        plt.figure(figsize=(10, 6))
        plt.plot(self.state['producer']['energy'], label='Energy', color='blue', linestyle='-', linewidth=2)
        plt.title('Energy - Producer state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Energy', fontsize=14)
        plt.yticks([0, 1, 2], ['LOW', 'MEDIUM', 'HIGH'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[producer-states]-energy.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['producer']['w_req_fps'], label='Worker on FPS', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.state['producer']['c_req_fps'], label='Consumer on FPS', color='red', linestyle='--', linewidth=2)
        plt.plot(self.state['producer']['c_req_res'], label='Consumer on Resolution', color='green', linestyle=':', linewidth=2)
        plt.title('Requests to Producer - Producer state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Requests to Producer', fontsize=14)
        plt.yticks([0, 1, 2], ['DECREASE', 'KEEP', 'INCREASE'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-requests2producer.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['producer']['fps'], label='FPS', color='blue', linestyle='-', linewidth=2)
        plt.title('FPS State', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('FPS', fontsize=14)
        plt.yticks([0, 1, 2, 3, 4], ['12', '16', '20', '26', '30'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-fps.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['producer']['res'], label='Resolution', color='blue', linestyle='-', linewidth=2)
        plt.title('Resolution state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Resolution', fontsize=14)
        plt.yticks([0, 1, 2, 3, 4, 5], ['120p', '180p', '240p', '360p', '480p', '720p'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-res.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['worker']['gpu'], label='GPU', color='blue', linestyle='-', linewidth=2)
        plt.title('GPU state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('GPU', fontsize=14)
        plt.yticks([0, 1], ['OFF', 'ON'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-gpu.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Worker state
        plt.figure(figsize=(10, 6))
        plt.plot(self.state['worker']['in_time'], label='In Time', color='blue', linestyle='-', linewidth=2)
        plt.title('In Time state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('In Time', fontsize=14)
        plt.yticks([0, 1], ['FALSE', 'TRUE'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-in_time.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['worker']['exe_time'], label='Execution Time', color='blue', linestyle='-', linewidth=2)
        plt.title('Execution Time state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Execution Time', fontsize=14)
        plt.yticks([0, 1, 2, 3, 4], ["LOW", "MID-LOW", "MID", "MID-HIGH", "HIGH"])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-exec_time.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['worker']['req_gpu'], label='Request on GPU', color='blue', linestyle='-', linewidth=2)
        plt.title('Request on GPU state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Request GPU', fontsize=14)
        plt.yticks([0, 1, 2], ['SWITCH OFF', 'KEEP', 'SWITCH ON'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-req2GPU.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['consumer']['success'], label='Success', color='blue', linestyle='-', linewidth=2)
        plt.title('Success state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Success', fontsize=14)
        plt.yticks([0, 1], ['FALSE', 'TRUE'])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-success.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.state['consumer']['distance'], label='Distance', color='blue', linestyle='-', linewidth=2)
        plt.title('Distance state', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Distance', fontsize=14)
        plt.yticks([0, 1, 2, 3, 4], ["SHORT", "MID-SHORT", "MID", "MID-LONG", "LONG"])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[states]-distance.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.slos['energy'], label='Energy', color='blue', linestyle='-', linewidth=2)
        plt.plot(self.slos['in_time'], label='In Time', color='red', linestyle='--', linewidth=2)
        plt.plot(self.slos['success'], label='Success rate', color='green', linestyle='-.', linewidth=2)
        plt.plot(self.slos['distance'], label='Distance', color='purple', linestyle=':', linewidth=2)
        plt.title('SLOs fulfillment', fontsize=20)
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('Fulfillment rate', fontsize=14)
        #plt.yticks([0, 1, 2, 3, 4], ["SHORT", "MID-SHORT", "MID", "MID-LONG", "LONG"])
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('results/figures/' + iteration_id + '-[SLOs].png', dpi=300, bbox_inches='tight')
        plt.close()

    # def save(self):
