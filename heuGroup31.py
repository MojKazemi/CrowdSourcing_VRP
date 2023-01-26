# -*- coding: utf-8 -*-
import os
import random
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from agents import *
from scipy import spatial
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import json

class distFromFile():
    '''
    This class use for convert distance file to distance matrix and make edge of origin to destination
    '''
    def __init__(self,csvFile):
        self.csvfile = csvFile
    def exportDist(self):
        '''
        'exportDist' function make a dictionary with tuples of origin and destination as its key with distance
        between them as its value. For e.g. {(i,j):dist(i,j)} for all i and j that i \neq j
        '''
        db = pd.read_csv(self.csvfile, header= None)
        dist_dict = {(i,j): db.iloc[i,j] for i in range(db.shape[0]) for j in range(db.shape[1]) if i != j}
        return dist_dict
    def distMatrix(self):
        db = pd.read_csv(self.csvfile, header = None)
        return db.to_numpy()

class HeuGroup31(Agent):
    '''
    This Class define the heuristic algorithm for crowdsourcing-VRP
    Selecting rowdsource points based on 3 methods :
        - distance from the depot
        - Crowdsource cost of points
        - randomly select points
    Then find the best solution for VRP in 4 steps :
        - Kmean clustering
        - Nearest Neighbour algorithm
        - 2-opt algorithm
        - Swap vehicles and deliveries between the different cluster
    '''

    def __init__(self, env):
        self.env = env
        self.name = 'heugroup31'
        self.quantile = 0.5         # Initial Quantile of distance points to depot
        self.quantile_crowd = 0.5   # Initial Quantile of crowdsourcing cost
        self.numCrowd = 10          # Inital Number of crowdsourcing
        self.data_improving_quantile = {
            "performance": [],
            "quantile": [],
            # "quantile_crowdCost": [],
            "compute_time":[]
        }
        self.Point_w_min_dist = []
        self.vehicles = self.env.get_vehicles()
        dist_class = distFromFile('./cfg/distance_matrix.csv')      # The class for distance matrix
        self.distance_matrix = dist_class.distMatrix()              # The function of taking distance matrix from file
        self.dist_dict = dist_class.exportDist()
        self.delivery_info = self.env.get_delivery()
        self.conv_time_to_cost = self.env.conv_time_to_cost

    def ClusterIndices(self, clustNum, labels_array, mainlist):
        '''
        This function is used in kmean clustering for finding the label of each delivery
        '''
        output=[0]
        for i in np.where(labels_array == clustNum)[0]:
            output.append(mainlist[i])
        return output

    def compute_delivery_to_crowdship(self, deliveries):
        '''
        The function select crowdsourcing points based on the distance between points and the depot.
        Its input is all deliveries and Its output is the ID of selected crowdsourcing
        '''
        if len(deliveries) == 0:
            return []
        points = []
        self.delivery = []
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
        threshold = np.quantile(self.distance_matrix[0, :], self.quantile)
        id_to_crowdship = []
        for i in range(len(self.distance_matrix[0, :])):
            if self.distance_matrix[0, i] > threshold:
                id_to_crowdship.append(i)
        return id_to_crowdship

    def compute_delivery_to_crowdship_crowd(self, deliveries):
        '''
        The function select crowdsourcing points based on the crowdsourcing cost of each point.
        Its input is all deliveries and Its output is the ID of selected crowdsourcing
        '''
        if len(deliveries) == 0:
            return []

        points = []
        self.delivery = []
        self.crowdcost = []
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
            self.crowdcost.append(ele['crowd_cost'])
        threshold_crowdCost = np.quantile(self.crowdcost,self.quantile_crowd)
        id_to_crowdship = []
        for i in range(len(self.distance_matrix[0, :])):
            if i != 0:
            # if self.distance_matrix[0, i] > threshold:
                if self.crowdcost[i-1] < threshold_crowdCost:
                    id_to_crowdship.append(i)
        return id_to_crowdship
    def compute_delivery_to_crowdship_rnd(self, deliveries):
        '''
        The function select crowdsourcing points randomly from the list of deliveries
        and the number of crowdsource is select in learn and save function, the intial value is define by 'self.numCrowd'
        Its input is all deliveries and Its output is the ID of selected crowdsourcing
        '''
        if len(deliveries) == 0:
            return []
        self.delivery = []
        self.crowdcost = []
        indexlist =[]
        id_to_crowdship = []
        print('numcrowd',self.numCrowd)
        id_to_crowdship = random.sample(range(1,len(deliveries)+1),self.numCrowd)
        print(id_to_crowdship)
        return id_to_crowdship

    def evaluate_VRP(self, VRP_solution):
        '''
        This function calculate the cost function of crowdsourcing-VRP
        and if the time window for each point and the capacity limitation of each vehicle are not satisfied
        the errorFlag will become TRUE. The output of the function is cost of VRP and tour's number
        '''
        # USAGE COST
        usage_cost = 0
        errorFlag = False
        for k in range(self.n_vehicles):
            if len(VRP_solution[k]) > 0:
                usage_cost += self.vehicles[k]['cost']

        # TOUR COST and CHECK TIME WINDOWS
        travel_cost = 0
        for k in range(self.n_vehicles):
            tour_time = 0
            for i in range(1, len(VRP_solution[k])-1):
                tour_time += self.distance_matrix[
                    VRP_solution[k][i - 1],
                    VRP_solution[k][i],
                ]
                if tour_time > self.delivery_info[VRP_solution[k][i]]['time_window_max']:
                    # print('Too Late for Delivery: ', VRP_solution[k][i])
                    errorFlag = True
                    return travel_cost, errorFlag, k

            travel_cost += self.conv_time_to_cost * tour_time

        # CHECK VOLUME
        for k in range(self.n_vehicles):
            tot_vol_used = 0
            for i in range(1, len(VRP_solution[k]) - 1):
                tot_vol_used += self.delivery_info[VRP_solution[k][i]]['vol']

            if tot_vol_used > self.vehicles[k]['capacity']:
                # print(f"Capacity Bound Violeted {tot_vol_used}>{self.vehicles[k]['capacity']}")
                errorFlag = True
                return travel_cost, errorFlag, k

        return usage_cost + travel_cost, errorFlag, self.n_vehicles

    def clusterDeliveries(self,deliveries, n_vehicles):
        '''
        The function cluster the delivery points by KMean algorithm.
        The class of KMean is equal to the number of vehicles.
        The inputs of the function are deliveries and the number of vehicles
        The output of the function is the clusters of delivery points.
        '''
        if len(deliveries) == 0:
            return []
        points = []
        delivery_points = []
        for _,ele in deliveries.items():
            points.append([ele['lat'],ele['lng']])
            delivery_points.append(ele['id'])
        # standard = StandardScaler()
        # standard_points = standard.fit_transform(points)
        # KMean algorithm:
        kmeans = KMeans(n_clusters= n_vehicles,n_init=10)
        kmeans.fit(points)
        y_kmeans = kmeans.predict(points)
        # Taking center of each cluster that used in swap-points step
        self.centers = kmeans.cluster_centers_

        tour =[]
        for veh in range(n_vehicles):
            tour.append(self.ClusterIndices(veh, kmeans.labels_,delivery_points))
        return tour

    def swapPositions(self, slist, pos1, pos2):
        # Swap the position of two item in the list
        getitem = slist[pos1], slist[pos2]
        slist[pos2], slist[pos1] = getitem
        return slist

    def nearestneieghbor(self,pointd_groups, dist_dict):
        '''
        The function is second step of heuristic method.
        It finds the initial tour between deliveries in each cluster
        '''
        VRP_Solution = []
        distSD_list = []
        for tour in pointd_groups:#.values():
            cycle = [0]
            distSD = []
            # find NN tour in points
            srcP = 0
            if tour[-1] == 0:
                tour.pop(-1)
            for i in range(len(tour)):
                if i == len(tour) - 1:
                    cycle.append(0)
                    distSD.append(dist_dict[(srcP, 0)])
                else:
                    dist_list = [(dist_dict[(srcP, j)], j) for j in tour if j != srcP and cycle.count(j) == 0]
                    sortdist = sorted(dist_list)
                    destP = sortdist[0][1]
                    cycle.append(destP)
                    distSD.append(sortdist[0][0])
                    srcP = destP

            VRP_Solution.append(cycle)
            distSD_list.append(distSD)

            # print('NN cycle: ', cycle, '\nDistance: ', distSD)
        return VRP_Solution, distSD_list

    def two_opt(self, tour, dist_dict, iteration):
        '''
        2-opt is the third step of the heuristic algorithm.
        This function swap points based on the defined cost function.
        '''
        VRP_solution = []
        cycle_list, distSD_list = self.nearestneieghbor(tour, dist_dict)
        # bayad route jadid jaygozin ghabli dar vrp solution bokonam
        i_r=0
        for route in cycle_list:
            best_list = copy.deepcopy(cycle_list)
            best = route
            improved = True
            ite = 0
            while improved and ite < iteration:
                # print('2-opt algorithm is running in the iteration: ', ite)
                improved = False
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route)):
                        if j - i == 1: continue  # changes nothing, skip then
                        new_route = route[:]
                        new_route[i:j] = route[j - 1:i - 1:-1]  # this is the 2woptSwap
                        cycle_list[i_r] = new_route
                        cost_new_route, _,_, = self.evaluate_VRP(cycle_list)
                        best_list[i_r] = best
                        cost_best,errorflag,_ = self.evaluate_VRP(best_list)
                        if not errorflag and cost_new_route < cost_best:
                            best = new_route
                            improved = True
                route = best
                ite += 1
            i_r += 1
        return best_list

    def two_opt_dist_cost(self, tour, dist_dict, iteration):
        '''
        2-opt is the third step of the heuristic algorithm.
        This function swap points based on the distance between them.
        '''
        VRP_solution = []
        cycle_list, distSD_list = self.nearestneieghbor(tour, dist_dict)
        for cycle in cycle_list:
            improved = True
            ite = 0
            while improved and ite < iteration:
                ite += 1
                best = sum(distSD_list[cycle_list.index(cycle)])
                size = len(cycle)
                improved = False
                for i in range(size - 2):  # cycle[0:size-3]:
                    for j in range(i + 2, size - 1):  # cycle[i+2:size-1]:
                        gain = dist_dict[(cycle[i], cycle[i+1])] + \
                               dist_dict[(cycle[j],cycle[j+1])] - \
                               dist_dict[(cycle[i], cycle[j])] - \
                               dist_dict[(cycle[i+1], cycle[j+1])]
                        if gain > 0:
                            best -= gain
                            cycle = self.swapPositions(cycle, i +1, j)
                            improved = True
                            # break
            VRP_solution.append(cycle)
        # print('Final Decision: cycle: ', cycle, '\nDistance Matrix: ', distSD)
        return VRP_solution

    def changepoints(self, vrp, numbVehicles, dist_dict):
        '''
        The function is used for change points between the cluster.
        It is used when a tour can not satisfy its limitation and
        it is used for the step fourth of the heuristic algorithm.
        '''
        points_selected = []
        Without_fail_tour = [i for i in range(self.n_vehicles) if i != numbVehicles] # make list of tour without the tour that raises error
        # dist of point with center of other cluster
        deliveries = self.env.get_delivery()
        for k in Without_fail_tour:
            center = self.centers[k]
            points = {ele['id'] : np.hypot(center[0] - ele['lat'], center[1] - ele['lng'])
                      for _, ele in deliveries.items() for i in vrp[numbVehicles] if ele['id'] == i}
            sortedDist = sorted(points.items(), key=lambda x: x[1], reverse=False)
            poin_min_dist = sortedDist[0][0]
            min_dist_in_k = sortedDist[0][1]
            points_selected.append((min_dist_in_k, poin_min_dist, k))
        sorted_points = sorted(points_selected,key = lambda x:x[0])
        inital_cost = np.inf
        list_best = []
        try:
            if len(vrp[numbVehicles]) > 2:
                for item in sorted_points:
                    VRP_temp = copy.deepcopy(vrp)
                    VRP_temp[numbVehicles].remove(item[1])
                    a_b = VRP_temp[item[2]].pop(-1) # remove zero from the other tour
                    VRP_temp[item[2]].append(item[1])
                    VRP_temp[item[2]].append(a_b)
                    temp_vrp = self.two_opt_dist_cost(VRP_temp, dist_dict, 100)
                    Temp_cost, errorFlag, tourNum = self.evaluate_VRP(temp_vrp)
                    if not errorFlag:
                        if 0 < Temp_cost < inital_cost:
                            list_best.append((Temp_cost,temp_vrp,item[1]))
                            inital_cost = Temp_cost
                if len(list_best) != 0:
                    sorted_vrp = sorted(list_best, key=lambda x: x[0])
                    vrp = copy.deepcopy(sorted_vrp[0][1])
                    best_point = sorted_vrp[0][2]
            else:
                raise Exception('Model is infeasible')
        except:
            raise Exception('Model is infeasible')

        return vrp

    def swapVehicleTour(self, best_VRP_solution, best_cost, n_vehicles):
        # change the vehicles assignment to clusters
        for i in range(n_vehicles):
            for j in range(n_vehicles):
                chk_VRP = copy.deepcopy(best_VRP_solution)
                chk_VRP = self.swapPositions(chk_VRP, i, j)
                swap_cost, errorFlag, _ = self.evaluate_VRP(chk_VRP)
                if swap_cost < best_cost and not errorFlag:
                    best_cost = swap_cost
                    best_VRP_solution = copy.deepcopy(chk_VRP)
        return best_VRP_solution, best_cost

    def checkChangePoint(self, best_VRP_solution, best_cost, dist_dict, n_vehicles):
        '''
        Change points between clusters
        '''
        for k in range(n_vehicles):
            temp_vrp = self.changepoints(best_VRP_solution, k, dist_dict)
            temp_cost, errorFlag, _ = self.evaluate_VRP(temp_vrp)
            if not errorFlag and temp_cost < best_cost:
                best_cost = temp_cost
                best_VRP_solution = copy.deepcopy(temp_vrp)
        return best_VRP_solution, best_cost

    def compute_VRP(self, delivery_to_do, vehicles_dict, debug_model=True, verbose=True):

        self.n_vehicles = len(vehicles_dict)
        self.n_deliveries = len(delivery_to_do)
        self.n_nodes = 1 + len(delivery_to_do)

        points_groups = self.clusterDeliveries(delivery_to_do, self.n_vehicles)
        processed = True
        iteration = 0
        pre_cost = np.inf
        __cost = np.inf
        VRP_solution = []

        start = time.time()

        while processed and iteration < 50:
            # print('processed iteration', iteration)
            iteration += 1
            VRP_solution = self.two_opt_dist_cost(points_groups, self.dist_dict, 100)
            # VRP_solution = self.two_opt(points_groups, self.dist_dict, 100)
            __cost, errorFlag, k = self.evaluate_VRP(VRP_solution)
            if not errorFlag:
                if __cost < pre_cost:
                    pre_cost = __cost
                    processed = False
                else:
                    points_groups = self.changepoints(VRP_solution, k, self.dist_dict)
            else:
                points_groups = self.changepoints(VRP_solution, k, self.dist_dict)
                if iteration == 100:
                    print('Model infeasible')

        best_VRP_solution = copy.deepcopy(VRP_solution)
        best_cost = __cost
        # # change the vehicles assignment to clusters
        best_VRP_solution, best_cost = self.swapVehicleTour(best_VRP_solution, best_cost,self.n_vehicles)
        # # change points btw points to find better cluster
        best_VRP_solution, best_cost = self.checkChangePoint(best_VRP_solution,best_cost,self.dist_dict,self.n_vehicles)

        end = time.time()
        self.comp_time = end - start
        print('Compute Time: ',self.comp_time)
        return best_VRP_solution

    def learn_and_save(self):
        self.quantile = np.random.uniform()
        # self.quantile_crowd = np.random.uniform()
        # self.numCrowd = random.randint(0,len(self.env.get_delivery()))

        id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
        remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
        try:
            VRP_solution = self.compute_VRP(remaining_deliveries, self.env.get_vehicles())
            obj = self.env.evaluate_VRP(VRP_solution)
        except:
            obj = 1000

        self.data_improving_quantile['quantile'].append(self.quantile)
        # self.data_improving_quantile['quantile_crowdCost'].append(self.quantile_crowd)
        # self.data_improving_quantile['quantile_crowdCost'].append(self.numCrowd)
        self.data_improving_quantile['performance'].append(tot_crowd_cost + obj)
        self.data_improving_quantile['compute_time'].append(self.comp_time)


    def start_test(self):
        db = pd.DataFrame(self.data_improving_quantile)
        db.to_csv('./results/data_improving_dist.csv')
        pos_min = self.data_improving_quantile['performance'].index(min(self.data_improving_quantile['performance']))
        self.quantile = self.data_improving_quantile['quantile'][pos_min]
        # self.quantile_crowd = self.data_improving_quantile['quantile_crowdCost'][pos_min]