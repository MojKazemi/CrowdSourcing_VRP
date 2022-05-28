# -*- coding: utf-8 -*-
import time
from agents import *


class DummyAgent(Agent):

    def __init__(self, env):
        self.name = "DummyAgent"
        self.env = env

    def compute_delivery_to_crowdship(self, deliveries):
        return [i + 1 for i in range(len(deliveries))]

    def compute_VRP(self, delivery_to_do, vehicles):
        ris = []

        for i in range(vehicles):
            ris.append([])
        ris[0].append(0)
        for ele in delivery_to_do:
            ris[0].append(ele)
        ris[0].append(0)
        return ris

    def learn_and_save(self):
        time.sleep(7)
    
    def start_test(self):
        pass
