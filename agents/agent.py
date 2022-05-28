from abc import abstractmethod
import numpy as np
import gurobipy as grb
from scipy import spatial

class Agent():
    """Base class for agent"""

    @abstractmethod
    def __init__(self, env):
        """
        Setting contains a dictionary with the parameters
        """
        self.env = env
        self.name = 'ExactAgent'
        self.quantile = 0.5
        self.data_improving_quantile = {
            "quantile": [],
            "performace": []
        }
    
    @abstractmethod
    def compute_delivery_to_crowdship(self, deliveries):
        pass

    @abstractmethod
    def compute_VRP(self, delivery_to_do, vehicles):
        def subtourelim(model, where):
            if where == grb.GRB.Callback.MIPSOL:
                # for i in nodes:
                #     for j in nodes:
                #         if model.cbGetSolution(model.getVarByName(f"Y[{i},{j}]")) > 0.5:
                #             print("Y[{},{}]".format(i,j))
                # print("## BGN ##")
                for k in vehicles:
                    # print("k:", k)
                    next_vector = []
                    for i in nodes:
                        for j in nodes:
                            if model.cbGetSolution(model.getVarByName(f"X[{i},{j},{k}]")) > 0.5:
                                # print("X[{},{},{}]".format(i, j, k))
                                next_vector.append(j)

                    tot_delivery = 0
                    for i in deliveries:
                        if model.cbGetSolution(model.getVarByName(f"Z[{i},{k}]")) > 0.5:
                            tot_delivery += 1

                    # print("> next_vector: ", next_vector)
                    # find the shortest cycle in the selected edge list
                    if len(next_vector) > 1:
                        tour = subtour(next_vector)
                        # print("> TOUR: ", tour, " ", len(tour), "-", tot_delivery + 2)
                        if len(tour) < tot_delivery + 2:
                            # print("> adding constr")
                            tmp = X[tour[0], tour[1], k]
                            # print(f"{tour[0]}, {tour[1]} ")
                            for i in range(2, len(tour)):
                                tmp += X[tour[i - 1], tour[i], k]
                                # print(f"{tour[i - 1]}, {tour[i]} ")
                            # print("<=", len(tour)-2)

                            model.cbLazy(
                                tmp <= len(tour) - 2
                            )
                            # print("## END ##")

        def subtour(deliveries):
            cycle = [0]
            if len(deliveries) == 0:
                return []
            points = []
            self.delivery = []
            for _, ele in deliveries.items():
                points.append([ele['lat'], ele['lng']])
                self.delivery.append(ele)
            distance_matrix = spatial.distance_matrix(points, points)

            threshold = min(distance_matrix[0, :])

        # Model
        self.model = grb.Model("C_VRP_TW")
        n_vehicles = len(vehicles_dict)
        vehicles = range(n_vehicles)
        n_deliveries = len(delivery_to_do)
        deliveries = range(len(delivery_to_do))
        n_nodes = 1 + len(delivery_to_do)
        nodes = range(n_nodes)

        points = [[0, 0]]
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
        H = self.model.addVars(deliveries, n_vehicles, )
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
# TODO: is it need to change the objective function
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
            (grb.quicksum(Y[i + 1, j] for j in nodes) == 1 for i in deliveries),
            name='linkZY'
        )
        # LINK Z W
        self.model.addConstrs(
            (Z[i, k] <= W[k] for i in deliveries for k in vehicles),
            name='linkZW'
        )

        # LINK Z X
        self.model.addConstrs(
            (X[i + 1, j, k] <= Z[i, k] for i in deliveries for j in nodes for k in vehicles),
            name='linkZW'
        )

        # LINK X Y
        self.model.addConstrs(
            (grb.quicksum(X[i, j, k] for k in vehicles) == Y[i, j] for i in nodes for j in nodes),
            name='linkXY'
        )

        # X IN DEPOT
        self.model.addConstrs(
            (grb.quicksum(X[0, j, k] for j in nodes) == W[k] for i in nodes for k in vehicles),
            name='max_vehicles'
        )
        # FLOW IN - OUT
        self.model.addConstrs(
            (grb.quicksum(X[i, j, k] for j in nodes) == grb.quicksum(X[j, i, k] for j in nodes) for i in nodes for k in
             vehicles),
            name='flow'
        )

        # T def
        self.model.addConstr((T[0] == 0), name='exit time')

        self.model.addConstrs(
            (T[i + 1] >= distance_matrix[j][i + 1] + T[j] - 1000 * (1 - Y[j, i + 1]) for j in nodes for i in
             deliveries),
            name='initial_time'
        )

        self.model.addConstrs(
            (T[i + 1] >= delivery_to_do[delivery_idx[i]]['time_window_min'] for i in deliveries),
            name='initial_time_bound'
        )
        self.model.addConstrs(
            (T[i + 1] <= delivery_to_do[delivery_idx[i]]['time_window_max'] for i in deliveries),
            name='initial_time_bound'
        )

        # STARTING HAMILTONIAN CONSTRAINTS
        self.model.addConstrs(
            (Y[i, i] == 0 for i in nodes),
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
        

    @abstractmethod
    def learn_and_save(self):
        pass
    
    @abstractmethod
    def start_test(self):
        pass
