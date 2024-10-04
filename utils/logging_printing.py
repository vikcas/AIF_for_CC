import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def slo_fulfillment(state_data):
    # Producer
    count_lows = 0
    for val in state_data['producer_e']:
        if val == 'LOW':
            count_lows += 1
    energy_slo = count_lows / len(state_data['producer_e'])
    # print('energy_slo: ' + str(energy_slo))

    worker_stay = 0
    for val in state_data['producer_w_res']:
        if val == 'STAY':
            worker_stay += 1
    worker_slo = worker_stay / len(state_data['producer_w_res'])
    # print('worker_stay_slo: ' + str(worker_slo))

    consumer_stay_res = 0
    for val in state_data['producer_c_res']:
        if val == 'STAY':
            consumer_stay_res += 1
    consumer_slo_res = consumer_stay_res / len(state_data['producer_c_res'])
    # print('consumer_stay_res_slo: ' + str(consumer_slo_res))

    consumer_stay_fps = 0
    for val in state_data['producer_c_fps']:
        if val == 'STAY':
            consumer_stay_fps += 1
    consumer_slo_fps = consumer_stay_fps / len(state_data['producer_c_fps'])
    # print('consumer_stay_fps_slo: ' + str(consumer_slo_fps))

    slo_status = dict()
    slo_status['producer'] = [energy_slo, worker_slo, consumer_slo_res, consumer_slo_fps]

    worker_intime = 0
    for val in state_data['worker_it']:
        if val is True:
            worker_intime += 1
    worker_intime_slo = worker_intime / len(state_data['worker_it'])
    # print('worker_intime_slo: ' + str(worker_intime_slo))

    worker_gpu_off = 0
    for val in state_data['worker_gpu']:
        if val == 'OFF':
            worker_gpu_off += 1
    worker_gpu_slo = worker_gpu_off / len(state_data['worker_gpu'])
    # print('worker_gpu_off: ' + str(worker_gpu_slo))

    slo_status['worker'] = [worker_intime_slo, worker_gpu_slo]

    consumer_latency = 0
    for val in state_data['consumer_lat']:
        if val == 'GOOD':
            consumer_latency += 1
    consumer_latency_slo = consumer_latency / len(state_data['consumer_lat'])
    # print('consumer_latency_slo: ' + str(consumer_latency_slo))

    consumer_res = 0
    for val in state_data['consumer_res']:
        if val == 'GOOD':
            consumer_res += 1
    consumer_res_slo = consumer_res / len(state_data['consumer_res'])
    # print('consumer_res_slo: ' + str(consumer_res_slo))

    slo_status['consumer'] = [consumer_latency_slo, consumer_res_slo]

    return slo_status

def slo_plots(slo_data, test_id):
    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['producer_e'])
    plt.ylabel('Energy SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_producer]slo_energy.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['producer_w_res'])
    plt.ylabel('Worker request SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_producer]slo_worker_request.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['producer_c_res'])
    plt.ylabel('Consumer request (resolution) SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_producer]slo_consumer_request_res.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['producer_c_fps'])
    plt.ylabel('Consumer request (fps) SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_producer]slo_consumer_request_fps.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['worker_it'])
    plt.ylabel('Worker in time SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_worker]slo_in_time.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['worker_gpu'])
    plt.ylabel('Worker GPU SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_worker]slo_gpu.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['consumer_lat'])
    plt.ylabel('Consumer latency SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_consumer]slo_latency.pdf', format='pdf', transparent=True)
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.plot(slo_data['consumer_res'])
    plt.ylabel('Consumer resolution SLO evolution')
    plt.tight_layout()
    plt.savefig('figures/[test-' + str(test_id) + '_consumer]slo_resolution.pdf', format='pdf', transparent=True)
    plt.close()


def state_plots(state_data, agent, test_id):
    if agent == 'producer':
        ## ENERGY
        energy_data_cat = state_data['producer_e']
        energy_data = list()
        for val in energy_data_cat:
            if val == 'LOW':
                energy_data.append(0)
            if val == 'MID':
                energy_data.append(1)
            if val == 'HIGH':
                energy_data.append(2)
        plt.figure(figsize=(12, 8))
        plt.plot(energy_data[:], marker='o')
        plt.ylabel('Energy consumption')
        plt.yticks(ticks=[0, 1, 2], labels=['LOW', 'MID', 'HIGH'])
        # plt.xticks(ticks=range(len(energy_data[1:])))
        plt.xlim(0, len(energy_data[:]))
        plt.xlabel('iterations')
        plt.title('Energy consumption (Categorical)')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]energy_consumption.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## Worker request
        plt.figure(figsize=(12, 8))
        w_res_data_cat = state_data['producer_w_res']
        w_res_data = list()
        for val in w_res_data_cat:
            if val == 'DECREASE':
                w_res_data.append(0)
            if val == 'INCREASE':
                w_res_data.append(2)
            if val == 'STAY':
                w_res_data.append(1)
        plt.plot(w_res_data[:], marker='o')
        plt.ylabel('Worker resolution request')
        plt.yticks(ticks=[0, 1, 2], labels=['DECREASE', 'STAY', 'INCREASE'])
        plt.xlim(0, len(w_res_data[:]))
        plt.xlabel('iterations')
        plt.title('Worker resolution request')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]worker_request_res.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## Consumer resolution request
        plt.figure(figsize=(12, 8))
        c_res_data_cat = state_data['producer_c_res']
        c_res_data = list()
        for val in c_res_data_cat:
            if val == 'DECREASE':
                c_res_data.append(0)
            if val == 'INCREASE':
                c_res_data.append(2)
            if val == 'STAY':
                c_res_data.append(1)
        plt.plot(c_res_data[:], marker='o')
        plt.ylabel('Consumer resolution request')
        plt.yticks(ticks=[0, 1, 2], labels=['DECREASE', 'STAY', 'INCREASE'])
        plt.xlim(0, len(c_res_data[:]))
        plt.xlabel('iterations')
        plt.title('Consumer resolution request')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]consumer_request_res.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## Consumer fps request
        plt.figure(figsize=(12, 8))
        c_fps_data_cat = state_data['producer_c_fps']
        c_fps_data = list()
        for val in c_fps_data_cat:
            if val == 'DECREASE':
                c_fps_data.append(0)
            if val == 'INCREASE':
                c_fps_data.append(2)
            if val == 'STAY':
                c_fps_data.append(1)
        plt.plot(c_fps_data[:], marker='o')
        plt.ylabel('Consumer fps request')
        plt.yticks(ticks=[0, 1, 2], labels=['DECREASE', 'STAY', 'INCREASE'])
        plt.xlim(0, len(c_fps_data[:]))
        plt.xlabel('iterations')
        plt.title('Consumer fps request')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]consumer_request_fps.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## Resolution
        plt.figure(figsize=(12, 8))
        res_data_cat = state_data['producer_res']
        res_data = list()
        for val in res_data_cat:
            if val == '120p':
                res_data.append(120)
            if val == '180p':
                res_data.append(180)
            if val == '240p':
                res_data.append(240)
            if val == '360p':
                res_data.append(360)
            if val == '480p':
                res_data.append(480)
            if val == '720p':
                res_data.append(720)
        plt.plot(res_data[:], marker='o')
        plt.ylabel('Resolution value')
        plt.yticks(ticks=[120, 180, 240, 360, 480, 720], labels=['120p', '180p', '240p', '360p', '480p', '720p'])
        plt.xlim(0, len(res_data[:]))
        plt.xlabel('iterations')
        plt.title('Image resolution evolution')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]resolution.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## FPS
        plt.figure(figsize=(12, 8))
        fps_data = state_data['producer_fps']
        plt.plot(fps_data[:], marker='o')
        plt.ylabel('FPS value')
        plt.yticks(ticks=[12, 16, 20, 26, 30], labels=['12', '16', '20', '26', '30'])
        plt.xlim(0, len(fps_data[:]))
        plt.xlabel('iterations')
        plt.title('Image FPS evolution')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]fps.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
    if agent == 'worker':
        ## In time
        plt.figure(figsize=(12, 8))
        intime_data = state_data['worker_it']
        plt.plot(intime_data[:], marker='o')
        plt.ylabel('In time')
        plt.xlabel('iterations')
        plt.yticks(ticks=[0, 1], labels=['False', 'True'])
        plt.xlim(0, len(intime_data[:]))
        plt.title('Worker in time SLO')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]in_time.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## GPU status
        plt.figure(figsize=(12, 8))
        gpu_data_cat = state_data['worker_gpu']
        gpu_data = list()
        for val in gpu_data_cat:
            if val == 'OFF':
                gpu_data.append(0)
            if val == 'ON':
                gpu_data.append(1)
        plt.plot(gpu_data[:], marker='o')
        plt.ylabel('GPU status')
        plt.yticks(ticks=[0, 1], labels=['OFF', 'ON'])
        plt.xlim(0, len(gpu_data[:]))
        plt.xlabel('iterations')
        plt.title('Worker GPU status -> preferred OFF (SLO)')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]gpu_status.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
    if agent == 'consumer':
        ## LATENCY
        plt.figure(figsize=(12, 8))
        latency_data_cat = state_data['consumer_lat']
        latency_data = list()
        for val in latency_data_cat:
            if val == 'BAD':
                latency_data.append(0)
            if val == 'GOOD':
                latency_data.append(1)
        plt.plot(latency_data[:], marker='o')
        plt.ylabel('Latency')
        plt.yticks(ticks=[0, 1], labels=['BAD', 'GOOD'])
        plt.xlim(0, len(latency_data[:]))
        plt.xlabel('iterations')
        plt.title('Consumer latency SLO')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]latency.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()
        ## Resolution
        plt.figure(figsize=(12, 8))
        resolution_data_cat = state_data['consumer_res']
        resolution_data = list()
        for val in resolution_data_cat:
            if val == 'BAD':
                resolution_data.append(0)
            if val == 'GOOD':
                resolution_data.append(1)
        plt.plot(resolution_data[:], marker='o')
        plt.ylabel('Resolution')
        plt.yticks(ticks=[0, 1], labels=['BAD', 'GOOD'])
        plt.xlim(0, len(resolution_data[:]))
        plt.xlabel('iterations')
        plt.title('Consumer resolution SLO')
        plt.tight_layout()
        plt.savefig('figures/[test-' + str(test_id) + '_{}-state]resolution.pdf'.format(agent), format='pdf', transparent=True)
        plt.close()

