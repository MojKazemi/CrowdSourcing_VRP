# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from envs.deliveryNetwork import DeliveryNetwork
from agents.exactVRPAgent import ExactVRPAgent
from agents.solAgent import solAgent
from agents.distPoints import distPoints
import time

if __name__ == '__main__':
    np.random.seed(0)
    log_name = "./logs/main_test_single.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )
    fp = open("./cfg/setting.json", 'r')
    settings = json.load(fp)
    fp.close()

    env = DeliveryNetwork(settings)
    delivery_info = env.get_delivery()

    # dist = distPoints(env)
    # dist_dict, arcos = dist.dist_evaluate()
    # print(dist_dict)
    # print(dist_dict[(0,2)])
    # breakpoint()
    # TODO: I have to find points that should be delivery
    # Temporary --------------------
    delivery_to_do = []
    for _, ele in delivery_info.items():
        delivery_to_do.append(ele['id'])


    agent = solAgent(env)
    env.prepare_crowdsourcing_scenario()
    print(delivery_to_do)
    id_deliveries_to_crowdship = agent.compute_delivery_to_crowdship(env.get_delivery())
    remaining_deliveries, tot_crowd_cost = env.run_crowdsourcing(id_deliveries_to_crowdship)
    for i in env.delivery_info:
        if env.delivery_info[i]['crowdsourced'] == 1:
            print('the id of crowdsourced',i)
    print("remaining_deliveries: ", remaining_deliveries.keys())
    print("tot_crowd_cost: ", tot_crowd_cost)

    start = time.time()

    VRP_solution = agent.compute_VRP(remaining_deliveries, env.get_vehicles())


    # VRP_solution = agent.main(delivery_to_do, vehicles_dict)

    end = time.time()
    comp_time = end - start

    # VRP_solution=[]
    # VRP_solution.append(cycle)
    # print(VRP_solution)
    env.render_tour(remaining_deliveries, VRP_solution)
    print(VRP_solution)
    print('Competition Time', comp_time)






    # agent = nnAgent(env)
    # agent = ExactVRPAgent(env)
    # env.prepare_crowdsourcing_scenario()

    # id_deliveries_to_crowdship = agent.compute_delivery_to_crowdship(
    #     env.get_delivery()
    # )

    # print(env.delivery_info[i]['crowdsourced'])
    ###
    # print("id_deliveries_to_crowdship: ", id_deliveries_to_crowdship)
    # remaining_deliveries, tot_crowd_cost = env.run_crowdsourcing(id_deliveries_to_crowdship)
    # for i in env.delivery_info:
    #     if env.delivery_info[i]['crowdsourced'] == 1:
    #         print('the id of crowdsourced',i)
    # print("remaining_deliveries: ", remaining_deliveries.keys())
    # print("tot_crowd_cost: ", tot_crowd_cost)
    # VRP_solution = agent.compute_VRP(remaining_deliveries, env.get_vehicles(), debug_model=True, verbose=True)
    # print("VRP_solution_exact: ", VRP_solution)
    #
    # env.render_tour(remaining_deliveries, VRP_solution)
    # obj = env.evaluate_VRP(VRP_solution)
    # print("obj: ", obj)
    # # for i in env.delivery_info:
    # #     print(env.delivery_info[i]['crowdsourced'])

