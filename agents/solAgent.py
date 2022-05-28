# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from agents import *
import gurobipy as grb
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class solAgent(Agent):

    def __init__(self, env):
        self.env = env
        self.name = 'ExactAgent'
        self.quantile = 0.5
        self.data_improving_quantile = {
            "quantile": [],
            "performace": []
        }

    def distCalculate(self, deliveries):
        #deliveries = self.env.get_delivery()
        if len(deliveries) == 0:
            return []
        points = [[0,0]]
        nodes =[0]
        self.delivery = []
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
            nodes.append(ele['id'])
        dist_matrix = spatial.distance_matrix(points, points)
        remind_dist_matrix = dist_matrix.copy()
        for i in nodes:
            remind_dist_matrix[i, i] = np.inf

        return dist_matrix, remind_dist_matrix

    def clusterDeliveries(self,deliveries, n_vehicles):
        if len(deliveries) == 0:
            return []
        if n_vehicles == 1:
            return deliveries
        points = []
        for _,ele in deliveries.item():
            points.append([ele['lat'],ele['lng']])
        standard = StandardScaler()
        standard_points = standard.fit_transform(points)
        kmeans = KMeans(n_clusters= n_vehicles)
        

    def swapPositions(self, slist, pos1, pos2):

        # Storing the two elements
        # as a pair in a tuple variable get
        getitem = slist[pos1], slist[pos2]

        # unpacking those elements
        slist[pos2], slist[pos1] = getitem

        return slist

    def solTSP(self, tour):
        cycle = [0]
        distance_matrix, remind_dist_matrix = self.distCalculate(self.env.get_delivery())
        _cost = remind_dist_matrix.copy()
        # print(distance_matrix,'\n',remind_dist_matrix)
        distSD = []
        # find NN tour in points
        print('-------------------- NN Solution --------------------------')
        srcP = 0
        for i in range(len(tour)):
            destP = np.argmin(remind_dist_matrix[srcP])
            cycle.append(destP)
            distSD.append(remind_dist_matrix[srcP].min())
            remind_dist_matrix[:, srcP] = np.inf
            srcP = destP
            i+=1
            if i == len(tour):
                cycle.append(0)
                distSD.append(distance_matrix[0,srcP])

        print('NN Tour: ',cycle)#,'\nDistance: ',distSD)
        print('2-opt is Running ...')
        # improved tour based on 2-opt solution
        improved = True
        while improved:
            best = sum(distSD)
            size = len(cycle)
            improved = False
            for i in range(size - 3):  # cycle[0:size-3]:
                for j in range(i + 2, size - 1):  # cycle[i+2:size-1]:
                    gain = _cost.item((cycle[i], cycle[i + 1])) + \
                           _cost.item((cycle[j], cycle[j + 1])) - \
                           _cost.item((cycle[i], cycle[j])) - \
                           _cost.item((cycle[i + 1], cycle[j + 1]))

                    if gain != np.inf and not np.isnan(gain) and gain > 1e-1:
                        best -= gain
                        cycle = self.swapPositions(cycle, i + 1, j)
                        # print('swap cycle: ', cycle)
                        improved = True
                        break
        # print('Final Decision: cycle: ', cycle, '\nDistance Matrix: ', distSD)
        print('------------------------------------------')
        return cycle

    def main(self, delivery_to_do, vehicles_dict):
        self.n_vehicles = len(vehicles_dict)
        self.n_deliveries = len(delivery_to_do)
        self.n_nodes = 1 + len(delivery_to_do)



        # For 1 vehicles - one tour
        self.distCalculate(self.env.get_delivery())
        cycle = self.solTSP(delivery_to_do)

        #For more than 1 vehicles
        #TODO: make optimze batch
        return cycle