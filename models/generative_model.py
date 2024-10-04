import pandas as pd
import numpy as np

class GenerativeModel:
    def __init__(self, rnd_gen, device_id='Xavier', SLO_thresholds='SLO_thresholds'):
        self.rnd_gen = rnd_gen
        self.path = 'data/data_' + device_id + '.csv'
        if device_id == 'Xavier':
            self.path = 'data/Xavier_extended_dataset.csv'
        else:
            self.path = 'data/data_PC.csv'
        self.used_lines = list()
        self.last_line = None
        self.params_history = {'res': list(), 'fps': list(), 'GPU': list()}
        self.iterations = 0
        self.s_threshold = pd.read_csv('data/' + SLO_thresholds + '.csv')
        self.data_pd = pd.read_csv(self.path)
        self.data_pd = self.data_pd.sample(frac=1).reset_index(drop=True)
        self.resolution_dict = {'25440': '120p', '57600': '180p', '102240': '240p', '230400': '360p',
                                '409920': '480p', '921600': '720p', '120p': 25440, '180p': 57600, '240p': 102240,
                                '360p': 230400, '480p': 409920, '720p': 921600}
        self.fps_options = [12, 16, 20, 26, 30]
        self.res_options = ['120p', '180p', '240p', '360p', '480p', '720p']
        # TODO: always have a last line, which is the first line to start with some initial values.

    def start_env(self, initial_params):
        fps_condition = self.data_pd['fps'] == initial_params['fps']
        res_condition = self.data_pd['pixel'] == self.resolution_dict.get(initial_params['resolution'])
        gpu_condition = self.data_pd['GPU'] == initial_params['GPU']

        # print(initial_params['fps'])
        # print(self.resolution_dict.get(initial_params['resolution']))
        # print(initial_params['GPU'])

        self.params_history['fps'].append(initial_params['fps'])
        self.params_history['res'].append(initial_params['resolution'])
        self.params_history['GPU'].append(initial_params['GPU'])

        all_conditions = fps_condition & res_condition & gpu_condition


        candidate_rows_pd = self.data_pd.loc[all_conditions]

        # Get the number of rows in the DataFrame (after potential slicing)
        num_rows = len(candidate_rows_pd)
        # Randomly select an index within valid bounds (0 to num_rows-1)
        random_index = self.rnd_gen.integers(low=0, high=num_rows - 1)
        # Access the row using the random index
        first_row = candidate_rows_pd.iloc[random_index]
        self.last_line = first_row  # Initial assigment for last line.

        return self.prepare_next_obs(first_row)


    def reset_env(self):
        self.used_lines = list()
        self.last_line = None
        self.params_history = {'res': list(), 'fps': list(), 'GPU': list()}
        self.iterations = 0

    def apply_action(self, producer_action, worker_action, consumer_action):
        '''

        :param producer_action: action on Resolution, action on FPS: [DECREASE/STAY/INCREASE,DECREASE/STAY/INCREASE]
        :param worker_action: action on demand for Resolution, action on GPU: [DECREASE/STAY/INCREASE,SWITCH OFF/STAY/SWITCH OFF]
        :param consumer_action: action on demand for Latency, action on demand for Resolution: [DECREASE/STAY/INCREASE,DECREASE/STAY/INCREASE]
        :return:
        '''
        last_fps = self.last_line['fps']
        last_resolution = self.resolution_dict[str(self.last_line['pixel'])]
        last_gpu = self.last_line['GPU']

        act_res = producer_action['resolution']
        act_fps = producer_action['fps']
        act_gpu = worker_action['gpu_action']

        res_index = self.res_options.index(last_resolution)
        if act_res == "DECREASE":
            if res_index == 0:
                next_res = self.res_options[res_index]
            else:
                next_res = self.res_options[res_index - 1]
        elif act_res == "INCREASE":
            if res_index == len(self.res_options) - 1:
                next_res = self.res_options[res_index]
            else:
                next_res = self.res_options[res_index + 1]
        else:
            next_res = self.res_options[res_index]

        next_resolution_key = self.resolution_dict.get(next_res)

        fps_index = self.fps_options.index(last_fps)
        if act_fps == "DECREASE":
            if fps_index == 0:
                next_fps = self.fps_options[fps_index]
            else:
                next_fps = self.fps_options[fps_index - 1]
        elif act_fps == "INCREASE":
            if fps_index == len(self.fps_options) - 1:
                next_fps = self.fps_options[fps_index]
            else:
                next_fps = self.fps_options[fps_index + 1]
        else:
            next_fps = self.fps_options[fps_index]

        if act_gpu == "SWITCH ON":
            next_gpu = True
        elif act_gpu == "SWITCH OFF":
            next_gpu = False
        else:
            next_gpu = last_gpu

        fps_condition = self.data_pd['fps'] == next_fps
        res_condition = self.data_pd['pixel'] == next_resolution_key
        gpu_condition = self.data_pd['GPU'] == next_gpu
        all_conditions = (fps_condition) & (res_condition) & (gpu_condition)

        print('##### Next parameters to find in dataset #####')
        self.params_history['fps'].append(next_fps)
        self.params_history['res'].append(self.resolution_dict[str(next_resolution_key)])
        self.params_history['GPU'].append(next_gpu)
        print('Next fps: ' + str(self.params_history['fps'][-1]))
        print('Next resolution: ' + str(self.params_history['res'][-1]))
        print('Next GPU: ' + str(self.params_history['GPU'][-1]))
        print('##########')

        candidate_rows_pd = self.data_pd[all_conditions]

        row_found = False

        if self.last_line.name in candidate_rows_pd.index:
            # Get the location of the current index
            idx_loc = candidate_rows_pd.index.get_loc(self.last_line.name)

            # Check if there's a next row
            if idx_loc + 1 < len(candidate_rows_pd):
                # Return the next row
                next_row = candidate_rows_pd.iloc[idx_loc + 1]
                row_found = True
            else:
                print("No next row available (last row in the DataFrame).")
        else:
            print("Previous index does not exist in the DataFrame.")
            # list_indexes = list()
            list_indexes = [a for a in candidate_rows_pd.index]
            self.rnd_gen.shuffle(list_indexes)
            for idx in list_indexes:
                try:
                    if idx not in self.used_lines:
                        self.used_lines.append(idx)
                        next_row = self.data_pd.iloc[idx]
                        row_found = True
                        break
                except IndexError:
                    print(f"Index {idx} is out of bounds. Exiting loop...")

        # for idx in candidate_rows_pd.index:
        #     try:
        #         if idx not in self.used_lines:
        #             self.used_lines.append(idx)
        #             next_row = self.data_pd.iloc[idx]
        #             row_found = True
        #             break
        #     except IndexError:
        #         print(f"Index {idx} is out of bounds. Exiting loop...")

        if row_found == False & len(candidate_rows_pd) > 0:
            print('This is a warning, no parameter row found for action selected.')
            # Get the number of rows in the DataFrame (after potential slicing)
            num_rows = len(candidate_rows_pd)
            # Randomly select an index within valid bounds (0 to num_rows-1)
            random_index = self.rnd_gen.integers(low=0, high=num_rows - 1)
            # Access the row using the random index
            next_row = candidate_rows_pd.iloc[random_index]
        if row_found == False & len(candidate_rows_pd) == 0:
            print('This is a warning, no parameter row found for action selected.')
            #  FIXME: In case we can't find a row with the parameter we discard GPU & RES.
            condition_no_gpu_no_res = fps_condition
            candidate_rows_pd = self.data_pd[condition_no_gpu_no_res]
            num_rows = len(candidate_rows_pd)
            # Randomly select an index within valid bounds (0 to num_rows-1)
            random_index = self.rnd_gen.integers(low=0, high=num_rows - 1)
            # Access the row using the random index
            next_row = candidate_rows_pd.iloc[random_index]

        self.last_line = next_row
        return self.prepare_next_obs(next_row)

    def prepare_next_obs(self, row):
        '''

        :param row:
        :return: Observations for the 3 pomdps.
        Producer: ENERGY (LOW-MID-HIGH);  (0-7) [7-8] (8-Inf)
                  WORKER_FPS (DECREASE-STAY-INCREASE);
                  CONSUMER_FPS (DECREASE-STAY-INCREASE);
                  CONSUMER_RES (DECREASE-STAY-INCREASE)
                  RESOLUTION ('120p', '180p', '240p', '360p', '480p', '720p')
                  FPS ('12', '16', '20', '26', '30')
                  GPU ('ON', 'OFF')
        Worker:
                execution time ("LOW", "MID-LOW", "MEDIUM", "MID-HIGH", "HIGH"); || 0-15, 16-30, 31-45, 46-60, 61+
                In time (False, True);
                FPS ('12', '16', '20', '26', '30')
                Producer request GPU ("OFF", "STAY", "ON")
                GPU STATUS ('OFF', 'ON')
        Consumer:
                Success ('FALSE', 'TRUE');
                Distance ("SHORT", "MID-SHORT", "MID", "MID-LONG", "LONG") || 0-25 / 26-50 / 51-75 / 76-100 / + 101
                FPS ("12", "16", "20", "26", "30")
                RES ("120p", "180p", "240p", "360p", "480p", "720p")
        '''

        observations = {"producer": {
            "o_e": [0, 0, 0],
            "o_w_fps": [0, 0, 0],
            "o_c_fps": [0, 0, 0],
            "o_c_res": [0, 0, 0],
            "o_fps": [0, 0, 0, 0, 0],
            "o_res": [0, 0, 0, 0, 0, 0],
            "o_gpu": [0, 0]
        }, "worker": {
            "o_exec_time": [0, 0, 0, 0, 0],
            "o_in_time": [0, 0],
            "o_req_gpu": [0, 0, 0],
            "o_fps": [0, 0, 0, 0, 0],
            "o_gpu": [0, 0]
        }, "consumer": {
            "o_success": [0, 0],
            "o_distance": [0, 0, 0, 0, 0],
            "o_fps": [0, 0, 0, 0, 0],
            "o_res": [0, 0, 0, 0, 0, 0]
        }
        }

        # PRODUCER ENERGY
        # LOW-MID-HIGH: (0-7) [7,8] (8,inf)
        # Extra energy for using GPU (1W)... It is there to trade off perf with energy.
        if row['GPU']:
            energy_consumption = row['consumption'] + 1
        else:
            energy_consumption = row['consumption']

        if energy_consumption < 7.0:
            observations["producer"]["o_e"][0] = 1
        elif energy_consumption > 8.0:
            observations["producer"]["o_e"][2] = 1
        else:
            observations["producer"]["o_e"][1] = 1

        # ALL FPS
        fps_param = row['fps']
        if fps_param == 12:
            observations["producer"]["o_fps"][0] = 1
            observations["worker"]["o_fps"][0] = 1
            observations["consumer"]["o_fps"][0] = 1
        if fps_param == 16:
            observations["producer"]["o_fps"][1] = 1
            observations["worker"]["o_fps"][1] = 1
            observations["consumer"]["o_fps"][1] = 1
        if fps_param == 20:
            observations["producer"]["o_fps"][2] = 1
            observations["worker"]["o_fps"][2] = 1
            observations["consumer"]["o_fps"][2] = 1
        if fps_param == 26:
            observations["producer"]["o_fps"][3] = 1
            observations["worker"]["o_fps"][3] = 1
            observations["consumer"]["o_fps"][3] = 1
        if fps_param == 30:
            observations["producer"]["o_fps"][4] = 1
            observations["worker"]["o_fps"][4] = 1
            observations["consumer"]["o_fps"][4] = 1

        # Producer and Consumer Resolution
        pixel_value = row['pixel']
        res_param = self.resolution_dict[str(pixel_value)]
        # res_param = self.res_options[pixel_value]
        if res_param == "120p":
            observations["producer"]["o_res"][0] = 1
            observations["consumer"]["o_res"][0] = 1
        if res_param == "180p":
            observations["producer"]["o_res"][1] = 1
            observations["consumer"]["o_res"][1] = 1
        if res_param == "240p":
            observations["producer"]["o_res"][2] = 1
            observations["consumer"]["o_res"][2] = 1
        if res_param == "360p":
            observations["producer"]["o_res"][3] = 1
            observations["consumer"]["o_res"][3] = 1
        if res_param == "480p":
            observations["producer"]["o_res"][4] = 1
            observations["consumer"]["o_res"][4] = 1
        if res_param == "720p":
            observations["producer"]["o_res"][5] = 1
            observations["consumer"]["o_res"][5] = 1

        # PRODUCER WORKER GPU
        gpu_param = row['GPU']
        if gpu_param:
            observations["producer"]["o_gpu"][1] = 1
            observations["worker"]["o_gpu"][1] = 1
        else:
            observations["producer"]["o_gpu"][0] = 1
            observations["worker"]["o_gpu"][0] = 1

        # EXECUTION TIME - WORKER
        exe_time = row['execution_time']
        if exe_time <= 15:
            observations["worker"]["o_exec_time"][0] = 1
        elif 15 < exe_time <= 30:
            observations["worker"]["o_exec_time"][1] = 1
        elif 30 < exe_time <= 45:
            observations["worker"]["o_exec_time"][2] = 1
        elif 45 < exe_time <= 60:
            observations["worker"]["o_exec_time"][3] = 1
        elif exe_time > 60:
            observations["worker"]["o_exec_time"][4] = 1

        # WORKER IN TIME:
        in_time_limit = 1000 / fps_param
        if exe_time <= in_time_limit:
            observations["worker"]["o_in_time"][1] = 1
        else:
            observations["worker"]["o_in_time"][0] = 1

        # CONSUMER SUCCESS
        success = row['success']
        if success:
            observations["consumer"]["o_success"][1] = 1
        else:
            observations["consumer"]["o_success"][0] = 1

        # CONSUMER DISTANCE
        distance = row['distance']
        if distance <= 25:
            observations["consumer"]["o_distance"][0] = 1
        if 25 < distance <= 50:
            observations["consumer"]["o_distance"][1] = 1
        if 50 < distance <= 75:
            observations["consumer"]["o_distance"][2] = 1
        if 75 < distance <= 100:
            observations["consumer"]["o_distance"][3] = 1
        if distance > 100:
            observations["consumer"]["o_distance"][4] = 1

        # TO PRODUCER - Worker FPS request:
        if observations["worker"]["o_in_time"][0] == 1:  # In time is false, we ask to decrease FPS
            observations["producer"]["o_w_fps"][0] = 1
        else:
            observations["producer"]["o_w_fps"][1] = 1  # In time is True, we ask to STAY FPS

        # TO PRODUCER - Consumer FPS request:
        if observations['consumer']['o_distance'][3] == 1 or observations['consumer']['o_distance'][4] == 1:
            # If distance is too high, we increase FPS
            observations["producer"]["o_c_fps"][2] = 1
        else:
            observations["producer"]["o_c_fps"][1] = 1  # Otherwise we stay

        # TO PRODUCER - Consumer resolution request:
        if observations['consumer']['o_success'][0] == 1:  # If success is False we increase resolution
            observations["producer"]["o_c_res"][2] = 1
        else:
            observations["producer"]["o_c_res"][1] = 1  # Otherwise we stay

        # To Worker - Producer GPU request:
        if observations['producer']['o_e'][2] == 1:  # If Energy is high, switch off GPU
            observations["worker"]["o_req_gpu"][0] = 1
        else:
            observations["worker"]["o_req_gpu"][1] = 1  # Otherwise stay.

        self.iterations = self.iterations + 1

        # print('next observation: ')
        # print(next_obs)
        p_obs, w_obs, c_obs = self.convert_observations(observations)

        return p_obs, w_obs, c_obs

    def convert_observations(self, observations):
        prod_obs = list()
        work_obs = list()
        cons_obs = list()

        prod_obs.append(observations["producer"]["o_e"].index(1))
        prod_obs.append(observations["producer"]["o_w_fps"].index(1))
        prod_obs.append(observations["producer"]["o_c_fps"].index(1))
        prod_obs.append(observations["producer"]["o_c_res"].index(1))
        prod_obs.append(observations["producer"]["o_fps"].index(1))
        prod_obs.append(observations["producer"]["o_res"].index(1))
        prod_obs.append(observations["producer"]["o_gpu"].index(1))
        work_obs.append(observations["worker"]["o_in_time"].index(1))
        work_obs.append(observations["worker"]["o_exec_time"].index(1))
        work_obs.append(observations["worker"]["o_fps"].index(1))
        work_obs.append(observations["worker"]["o_req_gpu"].index(1))
        work_obs.append(observations["worker"]["o_gpu"].index(1))
        cons_obs.append(observations["consumer"]["o_success"].index(1))
        cons_obs.append(observations["consumer"]["o_distance"].index(1))
        cons_obs.append(observations["consumer"]["o_fps"].index(1))
        cons_obs.append(observations["consumer"]["o_res"].index(1))

        return prod_obs, work_obs, cons_obs

        # energy_mid_value = self.s_threshold['Energy'][0]
        # if row['consumption'] < energy_mid_value:
        #     producer_energy = 'LOW'
        # elif row['consumption'] > energy_mid_value:
        #     producer_energy = 'HIGH'
        # else:
        #     producer_energy = 'MID'
        # print('ENERGY:')
        # print('threshold: ' + str(energy_mid_value))
        # print('energy consumed: ' + str(row['consumption']))
        #
        # producer_resolution = self.resolution_dict[str(row['pixel'])]
        # producer_fps = row['fps']
        # if row['GPU'] == True:
        #     worker_gpu = 'ON'
        # else:
        #     worker_gpu = 'OFF'
        # # worker_gpu = row['GPU']
        #
        # worker_intime_test = row['execution_time'] < 1000/producer_fps
        # print('WORKER in-time')
        # print('execution time: ' + str(row['execution_time']))
        # print('maximum time:' + str(1000/producer_fps))
        #
        # if not worker_intime_test:
        #     producer_worker_res = 'DECREASE'
        #     worker_intime = False
        # else:
        #     producer_worker_res = 'STAY'
        #     worker_intime = True
        #
        # latency_threshold = 1000/producer_fps  # self.s_threshold['Latency'][0]
        # # Execution time + random network latency + fps function
        # # consumer_latency_value = row['execution_time'] + self.rnd_gen.integers(-2, 2) + int(0.1*producer_fps)
        # consumer_latency_value = row['execution_time'] + self.rnd_gen.integers(-2, 2)
        # if (consumer_latency_value <= latency_threshold):  #   & (row['fps'] >= self.s_threshold['fps'][0])):  # FIXME: Added a minimum of FPS for happiness
        #     if (latency_threshold - consumer_latency_value) > 8:  # FIXME make this value a threshold
        #         consumer_latency = 'GOOD'
        #         producer_consumer_latency = 'INCREASE'
        #     else:
        #         consumer_latency = 'GOOD'
        #         producer_consumer_latency = 'STAY'
        # # elif ((consumer_latency_value <= latency_threshold) &
        # #       (row['fps'] < self.s_threshold['fps'][0])):
        # #     consumer_latency = 'BAD'
        # #     producer_consumer_latency = 'INCREASE'
        # # else:
        # #     consumer_latency = 'BAD'
        # #     producer_consumer_latency = 'DECREASE'
        # else:
        #     consumer_latency = 'BAD'
        #     producer_consumer_latency = 'DECREASE'
        #
        # print('CONSUMER LATENCY')
        # print('latency threshold: ' + str(latency_threshold))
        # print('consumer latency: ' + str(consumer_latency_value))
        #
        # resolution_threshold_idx = self.res_options.index(self.s_threshold['Resolution'][0])
        # row_idx = self.res_options.index(producer_resolution)
        # if row_idx < resolution_threshold_idx:
        #     consumer_resolution = 'BAD'
        #     producer_consumer_resolution = 'INCREASE'
        # else:
        #     consumer_resolution = 'GOOD'
        #     producer_consumer_resolution = 'STAY'
        #
        # print('CONSUMER RESOLUTION (threshold_idx < res_idx)')
        # print('resolution threshold - index: ' + str(resolution_threshold_idx))
        # print('producer resolution: ' + str(row_idx))
        #
        # next_obs = dict()
        # next_obs['producer'] = [producer_energy, producer_worker_res, producer_consumer_resolution,
        #                         producer_consumer_latency, producer_resolution, producer_fps]
        # next_obs['worker'] = [worker_intime, worker_gpu]
        # next_obs['consumer'] = [consumer_latency, consumer_resolution]

        # self.iterations = self.iterations + 1
        #
        # # print('next observation: ')
        # # print(next_obs)
        #
        # return next_obs


