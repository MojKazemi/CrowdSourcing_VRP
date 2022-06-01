# -*- coding: utf-8 -*-
import os
import time
import numpy as np
from agents import *
import gurobipy as grb
from scipy import spatial
import pdb

class ExactVRPAgent(Agent):

    def __init__(self, env):
        self.env = env
        self.name = 'ExactAgent'
        self.quantile = 0.5
        self.data_improving_quantile = {
            "quantile": [],
            "performace": []
        }

    def compute_delivery_to_crowdship(self, deliveries):
        if len(deliveries) == 0:
            return []
        points = []
        self.delivery = []
        for _, ele in deliveries.items():
            points.append([ele['lat'], ele['lng']])
            self.delivery.append(ele)
        distance_matrix = spatial.distance_matrix(points, points)
        threshold = np.quantile(distance_matrix[0, :], self.quantile)
        id_to_crowdship = []
        for i in range(len(distance_matrix[0, :])):
            if distance_matrix[0, i] > threshold:
                id_to_crowdship.append(i)
        print('id to crowdship',id_to_crowdship)
        return id_to_crowdship 

    def compute_VRP(self, delivery_to_do, vehicles_dict, gap=None, time_limit=None, verbose=False, debug_model=False):

        def subtourelim(model, where):
            if where == grb.GRB.Callback.MIPSOL:
                # for i in nodes:
                #     for j in nodes:
                #         if model.cbGetSolution(model.getVarByName(f"Y[{i},{j}]")) > 0.5:
                #             print("Y[{},{}]".format(i,j))
                #print("## BGN ##")
                for k in vehicles:
                    #print("k:", k)
                    next_vector = []
                    for i in nodes:
                        for j in nodes:
                            if model.cbGetSolution(model.getVarByName(f"X[{i},{j},{k}]")) > 0.5:
                                # print("X[{},{},{}]".format(i, j, k))
                                next_vector.append(j)
                    print('vehicles: ',k,'\nnext_vector: ',next_vector)
                    # breakpoint()
                    tot_delivery = 0
                    for i in deliveries:
                        if  model.cbGetSolution(model.getVarByName(f"Z[{i},{k}]")) > 0.5:
                            tot_delivery += 1

                    # print("> next_vector: ", next_vector)
                    # find the shortest cycle in the selected edge list
                    if len(next_vector) > 1:
                        tour = subtour(next_vector)
                        print("> TOUR: ", tour, " ", len(tour), "-", tot_delivery + 2)
                        if len(tour) < tot_delivery + 2:
                            #print("> adding constr")
                            tmp = X[tour[0], tour[1], k]
                            #print(f"{tour[0]}, {tour[1]} ")
                            for i in range(2, len(tour)):
                                tmp += X[tour[i - 1], tour[i], k]
                                #print(f"{tour[i - 1]}, {tour[i]} ")
                            #print("<=", len(tour)-2)

                            model.cbLazy(
                                tmp <= len(tour) - 2
                            )            
                #print("## END ##")

        def subtour(next_vector):
            # breakpoint()
            cycle = [0]
            # TODO: improve this while:
            while True:
                try:
                    cycle.append(next_vector[cycle[-1]])
                    if cycle[-1] == cycle[0]:
                        break
                except IndexError:
                    break
            return cycle

        # Model
        self.model = grb.Model("C_VRP_TW")
        n_vehicles = len(vehicles_dict)
        vehicles = range(n_vehicles)
        n_deliveries = len(delivery_to_do)
        deliveries = range(len(delivery_to_do))
        n_nodes = 1 + len(delivery_to_do)
        nodes = range(n_nodes)


        points = [[0,0]]
        nodes_idx = [0]
        delivery_idx = []
        for key, ele in delivery_to_do.items():
            points.append([ele['lat'], ele['lng']])
            nodes_idx.append(ele['id'])
            delivery_idx.append(ele['id'])

        distance_matrix = spatial.distance_matrix(points, points)

        # distance_matrix = self.distance_matrix[np.ix_(nodes_id, nodes_id)]

        # 1 if vehicle k pass link ij
        X = self.model.addVars(
            n_deliveries + 1, n_deliveries + 1, n_vehicles,
            vtype=grb.GRB.BINARY,
            name='X'
        )
        # 1 if link ij is visitted
        Y = self.model.addVars(
            n_deliveries + 1, n_deliveries + 1,
            vtype=grb.GRB.BINARY,
            name='Y'
        )
        # 1 delivery i is done by vehicle k
        Z = self.model.addVars(
            deliveries, n_vehicles,
            vtype=grb.GRB.BINARY,
            name='Z'
        )
        H = self.model.addVars(deliveries,n_vehicles,)
        # 1 vehicle k is used
        W = self.model.addVars(
            n_vehicles,
            vtype=grb.GRB.BINARY,
            name='W'
        )
        # arrival time at i
        T = self.model.addVars(
            n_deliveries + 1,
            vtype=grb.GRB.CONTINUOUS,
            lb=0.0,
            name='T'
        )

        obj_func = grb.quicksum(
            distance_matrix[i][j] * Y[i, j]
            for i in nodes
            for j in nodes
        )
        obj_func += grb.quicksum(
            self.env.vehicles[k]['cost'] * W[k]
            for k in vehicles
        )

        self.model.setObjective(obj_func, grb.GRB.MINIMIZE)

        # CAPACITY LIMIT
        self.model.addConstrs(
            (grb.quicksum(
                delivery_to_do[delivery_idx[i]]['vol'] * Z[i, k]
                for i in deliveries
            ) <= vehicles_dict[k]['capacity'] for k in vehicles),
            name='initial_inventory'
        )

        # VISIT ALL
        self.model.addConstrs(
            (grb.quicksum( Y[i + 1, j] for j in nodes) == 1 for i in deliveries),
            name='linkZY'
        )
        # LINK Z W
        self.model.addConstrs(
            (Z[i,k] <= W[k] for i in deliveries for k in vehicles),
            name='linkZW'
        )

        # LINK Z X
        self.model.addConstrs(
            (X[i + 1, j, k] <= Z[i,k] for i in deliveries for j in nodes for k in vehicles),
            name='linkZW'
        )

        # LINK X Y
        self.model.addConstrs(
            (grb.quicksum( X[i,j,k] for k in vehicles) == Y[i, j] for i in nodes for j in nodes),
            name='linkXY'
        )

        # X IN DEPOT
        self.model.addConstrs(
            (grb.quicksum( X[0, j, k] for j in nodes) ==  W[k] for i in nodes for k in vehicles),
            name='max_vehicles'
        )
        # FLOW IN - OUT
        self.model.addConstrs(
            (grb.quicksum( X[i, j, k] for j in nodes) == grb.quicksum( X[j, i, k] for j in nodes ) for i in nodes for k in vehicles),
            name='flow'
        )

        # T def
        self.model.addConstr((T[0] == 0), name='exit time')

        self.model.addConstrs(
            (T[i + 1] >= distance_matrix[j][i + 1] + T[j] - 1000 * (1 - Y[j, i + 1]) for j in nodes for i in deliveries),
            name='initial_time'
        )

        self.model.addConstrs(
            (T[i + 1] >= delivery_to_do[delivery_idx[i]]['time_window_min']  for i in deliveries),
            name='initial_time_bound'
        )
        self.model.addConstrs(
            (T[i + 1] <= delivery_to_do[delivery_idx[i]]['time_window_max']  for i in deliveries),
            name='initial_time_bound'
        )

        # STARTING HAMILTONIAN CONSTRAINTS
        self.model.addConstrs(
            (Y[i, i] == 0  for i in nodes),
            name='no self loop'
        )

        self.model.update()
        if gap:
            self.model.setParam('MIPgap', gap)
        if time_limit:
            self.model.setParam(grb.GRB.Param.TimeLimit, time_limit)
        if verbose:
            self.model.setParam('OutputFlag', 1)
        else:
            self.model.setParam('OutputFlag', 0)
        self.model.setParam('LogFile', './logs/gurobi.log')
        if debug_model:
            self.model.write(f"./logs/{self.name}.lp")

        self.model.Params.lazyConstraints = 1
        start = time.time()
        self.model.optimize(subtourelim)
        end = time.time()
        comp_time = end - start

        if self.model.status == grb.GRB.Status.OPTIMAL:
            sol = [] 

            for k in vehicles:
                sol.append([])
                if W[k].X > 0.5:
                    print(f"W[{k}]")
                    sol[k].append(0)
                else:
                    continue
                next_dict = {}
                for i in nodes:
                    for j in nodes:
                        if X[i,j,k].X > 0.5:
                            print(f"X[{i},{j},{k}]")
                            next_dict[i] = j
                
                for i in range(len(next_dict)):
                    sol[k].append(next_dict[sol[k][-1]])
                # from order to id delivery to do.
                for i in range(len(sol[k])):
                    sol[k][i] = nodes_idx[sol[k][i]]

            return sol
        else:
            print("MODEL INFEASIBLE OR UNBOUNDED")
            print(-1, [], comp_time)
            return []

    def learn_and_save(self):
        self.quantile = np.random.uniform()

        id_deliveries_to_crowdship = self.compute_delivery_to_crowdship(self.env.get_delivery())
        remaining_deliveries, tot_crowd_cost = self.env.run_crowdsourcing(id_deliveries_to_crowdship)
        VRP_solution = self.compute_VRP(remaining_deliveries, self.env.get_vehicles())
        obj = self.env.evaluate_VRP(VRP_solution)

        self.data_improving_quantile['quantile'].append(self.quantile)
        self.data_improving_quantile['performace'].append(tot_crowd_cost + obj)
    
    def start_test(self):
        pos_min = self.data_improving_quantile['performace'].index(min(self.data_improving_quantile['performace']))
        self.quantile = self.data_improving_quantile['quantile'][pos_min]

