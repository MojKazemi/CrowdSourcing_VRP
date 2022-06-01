import numpy as np

class distPoints():
    def __init__(self, env):
        self.deliveries = env.get_delivery()
        self.coordinate = [[0,0]]
        self.nodes = [0]

    def dist_evaluate(self):
        for _, ele in self.deliveries.items():
                self.coordinate.append([ele['lat'], ele['lng']])
                self.nodes.append(ele['id'])
        arcos = {(i, j) for i in self.nodes for j in self.nodes if i != j}
        dist_dict = {(i, j): np.hypot(self.coordinate[i][0] - self.coordinate[j][0], self.coordinate[i][1] - self.coordinate[j][1])
                     for i in self.nodes for j in self.nodes if i != j}
        return dist_dict, arcos