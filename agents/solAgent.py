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

    def distCalculate(self, deliveries, points_group):
        #deliveries = self.env.get_delivery()
        if len(deliveries) == 0:
            return []
        points = [[0,0]]
        nodes =[0]
        for _, ele in deliveries.items():
            for i in points_group:
                if ele['id'] == i:
                    points.append([ele['lat'], ele['lng']])
                    nodes.append(ele['id'])
        dist_matrix = spatial.distance_matrix(points, points)
        remind_dist_matrix = dist_matrix.copy()
        for i in range(len(nodes)):
            remind_dist_matrix[i, i] = np.inf

        return dist_matrix, remind_dist_matrix

    def ClusterIndices(self, clustNum, labels_array,mainlist):  # numpy
        output=[]
        for i in np.where(labels_array == clustNum)[0]:
            output.append(mainlist[i])
        return output

    def clusterDeliveries(self,deliveries, n_vehicles):
        if len(deliveries) == 0:
            return []
        points = []
        delivery_points = []
        for _,ele in deliveries.items():
            points.append([ele['lat'],ele['lng']])
            delivery_points.append(ele['id'])
        standard = StandardScaler()
        standard_points = standard.fit_transform(points)
        kmeans = KMeans(n_clusters= n_vehicles)
        kmeans.fit(standard_points)
        tour={}
        for veh in range(n_vehicles):
            tour[veh] = self.ClusterIndices(veh, kmeans.labels_,delivery_points)
        return tour

    def swapPositions(self, slist, pos1, pos2):

        # Storing the two elements
        # as a pair in a tuple variable get
        getitem = slist[pos1], slist[pos2]

        # unpacking those elements
        slist[pos2], slist[pos1] = getitem

        return slist

    def solTSP(self, tour):
        cycle = [0]
        distance_matrix, remind_dist_matrix = self.distCalculate(self.env.get_delivery(), tour)
        _cost = remind_dist_matrix.copy()
        # print(distance_matrix,'\n',remind_dist_matrix)
        distSD = []
        # find NN tour in points
        print('-------------------- NN Solution --------------------------')
        srcP = 0
        tour = [0] + tour
        print('tour',tour)
        for i in range(len(tour)-1):
            destP = np.argmin(remind_dist_matrix[srcP])
            # print('s',srcP,'d',destP)
            cycle.append(tour[destP])
            distSD.append(remind_dist_matrix[srcP].min())
            remind_dist_matrix[:, srcP] = np.inf
            srcP = destP
            i+=1
            if i == len(tour)-1:
                # print('s',srcP,'d',0)
                cycle.append(0)
                distSD.append(distance_matrix[0,srcP])

        print('NN Tour: ',cycle,'\nDistance: ',distSD)
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

        points_groups = self.clusterDeliveries(self.env.get_delivery(),self.n_vehicles)
        print('points_group: ', points_groups)

        cycle =[]
        for tour in points_groups.values():
            print(tour)

            # dist_matrix = distCalculate(self.env.get_delivery(),tour)
            opt_tour = self.solTSP(tour)
            print(opt_tour)
            cycle.append(opt_tour)
        print(cycle)
        return cycle