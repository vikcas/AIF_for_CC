import numpy as np
import pandas as pd
from pymdp import utils
from pymdp.agent import Agent
from models import consumer, producer, worker, generative_model
import matplotlib.pyplot as plt
from utils.logs import Logger
import time

if __name__ == '__main__':
    print("start")
    start_time = time.time()
    tests_idx = range(1)

    policy_len = 2  # Next try higher learning rate on agents

    log = Logger(debug=True)

    rnd_gen = np.random.default_rng(29)

    producer_obj = producer.ProducerAgent(b_matrix_learning=True, policy_length=policy_len)
    worker_obj = worker.WorkerAgent(b_matrix_learning=True, policy_length=policy_len)
    consumer_obj = consumer.ConsumerAgent(b_matrix_learning=True, policy_length=policy_len)
    environment = generative_model.GenerativeModel(rnd_gen)

    for t_id in tests_idx:

        producer_agent = producer_obj.generate_agent()
        worker_agent = worker_obj.generate_agent()
        consumer_agent = consumer_obj.generate_agent()

        inital_params = dict()
        inital_params['fps'] = rnd_gen.choice([12, 16, 20, 26, 30])
        inital_params['resolution'] = rnd_gen.choice(['120p', '180p', '240p', '360p', '480p', '720p'])
        inital_params['GPU'] = rnd_gen.choice([False, True])

        print(inital_params)

        p_obs, w_obs, c_obs = environment.start_env(initial_params=inital_params)

        max_iter = 70

        print("Prep time: %f seconds" % (time.time() - start_time))

        for ii in range(max_iter):
            loop_time = time.time()

            log.append_obs('producer', p_obs)
            log.append_obs('worker', w_obs)
            log.append_obs('consumer', c_obs)

            qs_p = producer_agent.infer_states(p_obs)
            qs_w = worker_agent.infer_states(w_obs)
            qs_c = consumer_agent.infer_states(c_obs)

            # IMPORTANT: Learning - No update on observation only on transition
            if ii > 0:
                # producer_agent.update_A(producer_obs)
                producer_agent.update_B(qs_p)
                # worker_agent.update_A(worker_obs)
                worker_agent.update_B(qs_w)
                # consumer_agent.update_A(consumer_obs)
                consumer_agent.update_B(qs_c)

            # Producer policy inference and action selection
            q_pi_prod, G_prod, G_sub_prod = producer_agent.infer_policies()
            chosen_action_id = producer_agent.sample_action()
            producer_action = producer_obj.translate_action(chosen_action_id)

            log.append_state('producer', qs_p)
            log.append_action('producer', chosen_action_id)
            log.append_efe('producer', q_pi_prod, G_prod, G_sub_prod)

            # Worker policy inference and action selection
            q_pi_worker, G_worker, G_sub_worker = worker_agent.infer_policies()
            chosen_action_id = worker_agent.sample_action()
            worker_action = worker_obj.translate_action(chosen_action_id)

            log.append_state('worker', qs_w)
            log.append_action('worker', chosen_action_id)
            log.append_efe('worker', q_pi_worker, G_worker, G_sub_worker)

            # Consumer policy inference and action selection
            q_pi_cons, G_cons, G_sub_cons = consumer_agent.infer_policies()
            chosen_action_id = consumer_agent.sample_action()
            consumer_action = consumer_obj.translate_action(chosen_action_id)

            log.append_state('consumer', qs_c)
            log.append_action('consumer', chosen_action_id)
            log.append_efe('consumer', q_pi_cons, G_cons, G_sub_cons)

            log.compute_slos()

            producer_agent.step_time()
            worker_agent.step_time()
            consumer_agent.step_time()

            # Generate new observation by applying the action
            p_obs, w_obs, c_obs = environment.apply_action(producer_action=producer_action, worker_action=worker_action,
                                                       consumer_action=consumer_action)

            print("#### iteration: " + str(ii) + ' finished ####')
            print("Loop time was: %f seconds" % (time.time() - loop_time))

        log.print()
        environment.reset_env()
        print("--- %f seconds ---" % (time.time() - start_time))
