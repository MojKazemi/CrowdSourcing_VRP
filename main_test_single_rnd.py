# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from envs.deliveryNetwork import DeliveryNetwork
from agents.exactVRPAgent import ExactVRPAgent
from agents.heuGroup31_rnd import heuGroup31_rnd
from rndInstances import rndInstances
import pdb

if __name__ == '__main__':
    np.random.seed(10)
    log_name = "./logs/main_test_single.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )
    # fp = open("./cfg/setting.json", 'r')
    fp = open("./cfg/setting_1.json", 'r')
    settings = json.load(fp)
    fp.close()

    # randInst = rndInstances(number_delivers=50, n_vehicles=5)
    # settings = randInst.run()

    # env = DeliveryNetwork(settings)
    env = DeliveryNetwork(settings, data_csv='./cfg/delivery_info.json')
    # env = DeliveryNetwork(settings, data_csv='./cfg/delivery_info_rnd.json')

    agent = heuGroup31_rnd(env)
    # agent = ExactVRPAgent(env)
    env.prepare_crowdsourcing_scenario()
    it = 0
    while it<500:
        print('iteration: ',it)
        agent.learn_and_save()
        it += 1


    agent.start_test()

    id_deliveries_to_crowdship = agent.compute_delivery_to_crowdship(
        env.get_delivery()
    )
    print("id_deliveries_to_crowdship: ", id_deliveries_to_crowdship)
    remaining_deliveries, tot_crowd_cost = env.run_crowdsourcing(id_deliveries_to_crowdship)
    for i in env.delivery_info:
        if env.delivery_info[i]['crowdsourced'] == 1:
            print('the id of crowdsourced',i)
    print("remaining_deliveries: ", remaining_deliveries.keys())
    print("tot_crowd_cost: ", tot_crowd_cost)
    VRP_solution = agent.compute_VRP(remaining_deliveries, env.get_vehicles())#, debug_model=True, verbose=True)
    # print("VRP_solution_exact: ", VRP_solution)
    print("VRP_solution_heuGroup31: ", VRP_solution)
    obj = env.evaluate_VRP(VRP_solution)
    print("obj: ", obj)
    print("obj+crowd",obj+tot_crowd_cost)
    env.render_tour(remaining_deliveries, VRP_solution)
    import matplotlib.pyplot as plt
    # perform =[]
    # quant =[]
    output =[]
    compt_time=[]
    for i, perf in enumerate(agent.data_improving_quantile['performance']):
        if perf < 1000:
            output.append((agent.data_improving_quantile['numCrowd'][i],perf))
            compt_time.append(agent.data_improving_quantile['compute_time'][i])
            # perform.append(perf)
            # quant.append(agent.data_improving_quantile['quantile_crowdCost'][i])
    output.sort(key=lambda x:x[1])
    # print(output)
    plt.scatter(*zip(*output))
    plt.ylabel('Cost')
    plt.xlabel('numCrowd')
    # plt.title('Cost versus Quantile of distance')
    plt.title('Cost Versus Number of Crowdsourcing')
    plt.savefig('./results/costVERquantile_rnd_500.png')
    plt.grid()
    plt.show()
    plt.plot(compt_time)
    plt.ylabel('Time(s)')
    plt.xlabel('Number of Iteration')
    plt.title('Computation Time')
    plt.savefig('./results/comptTime_500.png')
    plt.grid(axis='both')
    plt.show()


