# -*- coding: utf-8 -*-
import json
import signal
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from envs.deliveryNetwork import DeliveryNetwork
from agents.heuGroup31 import HeuGroup31
from agents.heuGroup31_rnd import heuGroup31_rnd
from agents.heuGroup31_crowd import heuGroup31_crowd
from agents.exactVRPAgent import ExactVRPAgent
from rndInstances import rndInstances


def timeout_handler(signum, frame):   # Custom signal handler
    raise Exception('Too much time')


# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)


if __name__ == '__main__':
    np.random.seed(0)
    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )
    TIME_LIMIT = 5
    signal.alarm(TIME_LIMIT)
    fp = open("./cfg/setting_1.json", 'r')
    settings = json.load(fp)
    fp.close()

    # randInst = rndInstances(number_delivers=30, n_vehicles=3)
    # settings = randInst.run()

    env = DeliveryNetwork(settings, './cfg/delivery_info.json')

    agents = [
        HeuGroup31(env),
        # heuGroup31_rnd(env),
        # heuGroup31_crowd(env)
        ExactVRPAgent(env)
    ]

    data = {}
    tot_crowd_cost_list = {}
    tot_cost = {}
    for i, agent in enumerate(agents):
        data[agent.name] = []
        tot_crowd_cost_list[agent.name] = []
        tot_cost[agent.name]=[]

    black_list = []

    # TRAIN
    for s in range(3):
        env.prepare_crowdsourcing_scenario()
        for agent in agents:
            try:
                if agent.name in black_list:
                    continue
                agent.learn_and_save()
            except Exception as e:
                print("###")
                print(e)
                print("###")
                black_list.append(agent.name)
            else:
                # Reset the alarm
                signal.alarm(0)

    # STOP TRAIN AND START TEST
    for agent in agents:
        try:
            agent.start_test()
            if agent.name in black_list:
                continue
        except Exception as e:
            black_list.append(agent.name)
        else:
            # Reset the alarm
            signal.alarm(0)


    # TEST
    for s in range(10):
        env.prepare_crowdsourcing_scenario()
        for agent in agents:
            if agent.name in black_list:
                continue
            print("## ## ## ")
            id_deliveries_to_crowdship = agent.compute_delivery_to_crowdship(env.get_delivery())
            print("id_deliveries_to_crowdship: ", id_deliveries_to_crowdship)
            remaining_deliveries, tot_crowd_cost = env.run_crowdsourcing(id_deliveries_to_crowdship)
            print("remaining_deliveries: ", remaining_deliveries )
            print("tot_crowd_cost: ", tot_crowd_cost)
            tot_crowd_cost_list[agent.name].append(tot_crowd_cost)
            VRP_solution = agent.compute_VRP(remaining_deliveries, env.get_vehicles())
            print("VRP_solution: ", VRP_solution)
            # env.render_tour(remaining_deliveries, VRP_solution)
            try:
                obj = env.evaluate_VRP(VRP_solution)
                print(">>> obj: ", obj + tot_crowd_cost)
                data[agent.name].append(obj)
                temp = obj+tot_crowd_cost
                tot_cost[agent.name].append(temp)
            except Exception as e:
                black_list.append(agent.name)
            else:
                # Reset the alarm
                signal.alarm(0)
            print("## ## ## ")

    for ele in black_list:
        data.pop(ele, None)
        tot_crowd_cost_list.pop(ele,None)
        tot_cost.pop(ele,None)

    print("black_list:", black_list)
    print(data)

    # plt.figure()
    df = pd.DataFrame.from_dict(data)
    df.boxplot(vert=False)
    plt.xlabel('Cost')
    plt.title('The Cost variation by changing assigning crowdsourcing')
    # plt.subplots_adjust(left=0.25)
    plt.savefig('./results/final-l.png')
    plt.show()
    #
    # # plt.figure
    # # print(tot_crowd_cost_list)
    # df_tc = pd.DataFrame.from_dict(tot_crowd_cost_list)
    # df_tc.boxplot(vert=False)
    # plt.xlabel('Cost')
    # plt.title('The Crowdsourcing Cost variation by changing assigning crowdsourcing')
    # # plt.subplots_adjust(left=0.25)
    # plt.savefig('./results/final_TC_3.png')
    # plt.show()
    #
    # # print(tot_cost)
    # df_to = pd.DataFrame.from_dict(tot_cost)
    # df_to.boxplot(vert=False)
    # plt.xlabel('Cost')
    # plt.title('The Total Cost variation by changing assigning crowdsourcing')
    # # plt.subplots_adjust(left=0.25)
    # plt.savefig('./results/final_TO_3.png')
    # plt.show()

