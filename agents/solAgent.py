# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from agents import *
from scipy import spatial
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from agents.distPoints import distPoints

class solAgent(Agent):

    def __init__(self, env):
        self.env = env
        self.name = 'ExactAgent'
        self.quantile = 0.5
        self.data_improving_quantile = {
            "quantile": [],
            "performace": []
        }
        self.Point_w_min_dist = []

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
        output=[0]
        for i in np.where(labels_array == clustNum)[0]:
            output.append(mainlist[i])
        return output

    def compute_delivery_to_crowdship(self, deliveries):
        min_samplesv = 2
        quantile = 0.5
        points = []
        self.delivery = []
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
        distance_matrix = spatial.distance_matrix(points, points)
        threshold = np.quantile(distance_matrix[0, :], quantile)
        id_to_crowdship_exact = []
        for i in range(len(distance_matrix[0, :])):
            if distance_matrix[0, i] > threshold:
                id_to_crowdship_exact.append(i)
        print('id to crowdship Exact: ', id_to_crowdship_exact)

        # quantile = round((len(points)-1)/2)
        # id_to_crowdship = np.argpartition(distance_matrix[0,:],-quantile)[-quantile:]  # is like prof. algorithm
        # print('Max dist',id_to_crowdship)
        # # DBSCAN to find outliers points of outhers
        # epsv = np.mean(distance_matrix[0,:])/((len(points)-1)/10)
        # clusters = DBSCAN(eps=epsv, min_samples=min_samplesv,  metric='euclidean').fit(points)
        # label_List, label_diversity = np.unique(clusters.labels_,return_counts=True)
        # id_to_crowdship_DBSCAN = np.argwhere(clusters.labels_ != label_List[np.argmax(label_diversity)]).flatten()
        # print('DBSCAN Removing: ',id_to_crowdship_DBSCAN)
        # plt.show()
        return id_to_crowdship_exact

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
        self.centers = kmeans.cluster_centers_
        # print('centers: ',kmeans.cluster_centers_)
        tour={}
        for veh in range(n_vehicles):
            tour[veh] = self.ClusterIndices(veh, kmeans.labels_,delivery_points)
        return tour

    def swapPositions(self, slist, pos1, pos2):
        getitem = slist[pos1], slist[pos2]
        slist[pos2], slist[pos1] = getitem
        return slist

    def nearestneieghbor(self,pointd_groups):
        VRP_Solution = []
        distSD_list = []
        _cost  = []
        print(pointd_groups)
        for tour in pointd_groups.values():
            cycle = [0]
            distance_matrix, remind_dist_matrix = self.distCalculate(self.env.get_delivery(), tour)
            print(tour)
            print(distance_matrix)
            __cost = remind_dist_matrix.copy()
            _cost.append(__cost)
            # print(distance_matrix,'\n',remind_dist_matrix)
            distSD = []
            # find NN tour in points
            print('-------------------- NN Solution --------------------------')
            srcP = 0
            for i in range(len(tour) - 1):
                destP = np.argmin(remind_dist_matrix[srcP])
                # print('s',srcP,'d',destP)
                cycle.append(tour[destP])
                distSD.append(remind_dist_matrix[srcP].min())
                remind_dist_matrix[:, srcP] = np.inf
                srcP = destP
                i += 1
                if i == len(tour) - 1:
                    if tour[-1] != 0:
                        # print('s',srcP,'d',0)
                        cycle.append(0)
                        distSD.append(distance_matrix[0, srcP])
                    else:
                        distSD[-1] = distance_matrix[0, srcP]
            VRP_Solution.append(cycle)
            distSD_list.append(distSD)

            print('NN cycle: ', cycle, '\nDistance: ', distSD)
        return VRP_Solution, distSD_list, _cost

    def swap2opt(self, tour, dist_dict, iteration):
        VRP_solution = []
        cycle_list, distSD_list, _cost = self.nearestneieghbor(tour)
        print('2-opt is Running ...')
        # improved tour based on 2-opt solution
        for cycle in cycle_list:
            improved = True
            ite = 0
            while improved and ite < iteration:
                print('2-opt algorithm is running in the iteration: ', ite)
                ite += 1
                best = sum(distSD_list[cycle_list.index(cycle)])
                size = len(cycle)
                improved = False
                for i in range(size - 3):  # cycle[0:size-3]:
                    for j in range(i + 2, size - 2):  # cycle[i+2:size-1]:
                        gain = dist_dict[(cycle[i], cycle[i+1])] + \
                               dist_dict[(cycle[j],cycle[j+1])] - \
                               dist_dict[(cycle[i], cycle[j])] - \
                               dist_dict[(cycle[i+1], cycle[j+1])]
                        if gain != np.inf and not np.isnan(gain) and gain > 1e-1:
                            best -= gain
                            cycle = self.swapPositions(cycle, i + 1, j)
                            improved = True
                            break
            VRP_solution.append(cycle)
        # print('Final Decision: cycle: ', cycle, '\nDistance Matrix: ', distSD)
        print('------------------------------------------')
        return VRP_solution

    def changepoints(self, VRP_solution, numbVehicles):
        Without_fail_tour = [i for i in range(self.n_vehicles) if i != numbVehicles] # the list of tour without the tour with error
        # dist of point with center of other cluster
        deliveries = self.env.get_delivery()
        for k in Without_fail_tour:
            center = self.centers[k]
            points = {ele['id'] : np.hypot(center[0] - ele['lat'], center[1] - ele['lng'])
                      for _, ele in deliveries.items() for i in VRP_solution[numbVehicles] if ele['id'] == i}
            sortedDist = sorted(points.items(), key=lambda x: x[1], reverse=False)
            for i in sortedDist:
                print(i)
                print(self.Point_w_min_dist)
                if self.Point_w_min_dist.count(i[0]) == 0:
                    self.Point_w_min_dist.append(i[0])
                    print(self.Point_w_min_dist)
                    break

            # self.Point_w_min_dist = points['CeVeh_id'][np.argmin(points['dist'])][1]
            try:
                VRP_solution[numbVehicles].remove(self.Point_w_min_dist[-1])
                a = VRP_solution[k].pop(-1) # remove zero from the other tour
                VRP_solution[k].append(self.Point_w_min_dist[-1])
                VRP_solution[k].append(a)
            except:
                raise Exception('Model is infeasible')

        # print(points)
        points_groups ={}
        for k in range(self.n_vehicles):
            points_groups[k] = VRP_solution[k]
        print('changed finished',points_groups)
        return points_groups

    def compute_VRP(self, delivery_to_do, vehicles_dict):
        self.n_vehicles = len(vehicles_dict)
        self.n_deliveries = len(delivery_to_do)
        self.n_nodes = 1 + len(delivery_to_do)
        dist = distPoints(self.env)
        dist_dict, arcos = dist.dist_evaluate()
        points_groups = self.clusterDeliveries(delivery_to_do, self.n_vehicles)
        print('points_group after clustering: ', points_groups)
        processed = True
        iteration = 0
        pre_cost = np.inf
        while processed and iteration < 100:
            VRP_solution = []
            print('evaluate the cost function in iteration: ',iteration)
            iteration += 1
            # for tour in points_groups.values():
            print('tour in start of while: ',points_groups)
            # dist_matrix = distCalculate(self.env.get_delivery(),tour)
            VRP_solution = self.swap2opt(points_groups, dist_dict, 500)
            print('optimum tour after 2- opt',VRP_solution)
            # VRP_solution.append(opt_tour)

            __cost, errorFlag, k = self.env.evaluate_VRP(VRP_solution)
            if not errorFlag:
                print(__cost)
                if __cost < pre_cost:
                    pre_cost = __cost
                    processed = False
                else:
                    points_groups = self.changepoints(VRP_solution, k)
            else:
                print('moji')
                points_groups = self.changepoints(VRP_solution, k)
        return VRP_solution