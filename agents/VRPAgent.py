# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from agents import *
import gurobipy as grb
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class VRPAgent(Agent):

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
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            nodes.append(ele['id'])
        arcos = {(i, j) for i in nodes for j in nodes if i != j}
        dist_dict = {(i, j): np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
                     for i in nodes for j in nodes if i != j}
        print('arcs:\n', arcos, '\narcs and dist:\n', dist_dict)

        return arcos, dist_dict

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
        kmeans = KMeans(n_clusters= n_vehicles,n_init=10)
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

    def solTSP(self, tour, iteration):
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

        print('NN cycle: ',cycle,'\nDistance: ',distSD)
        print('2-opt is Running ...')
        # improved tour based on 2-opt solution
        improved = True
        ite = 0
        while improved:
            if ite < iteration:
                print('2-opt is running in iteration: ',ite)
                ite += 1
                best = sum(distSD)
                size = len(cycle)
                improved = False
                for i in range(size - 3):  # cycle[0:size-3]:
                    for j in range(i + 2, size - 1):  # cycle[i+2:size-1]:
                        gain = _cost.item((tour.index(cycle[i]), tour.index(cycle[i + 1]))) + \
                               _cost.item((tour.index(cycle[j]), tour.index(cycle[j + 1]))) - \
                               _cost.item((tour.index(cycle[i]), tour.index(cycle[j]))) - \
                               _cost.item((tour.index(cycle[i + 1]), tour.index(cycle[j + 1])))

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

        cycle =[]

        return cycle
