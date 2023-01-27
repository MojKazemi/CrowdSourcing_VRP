'''
Get the real instances then select random delivery points of them
'''
import pandas as pd
import json
import random

class rndInstances():
    def __init__(self,  number_delivers = 20, n_vehicles=3):
        self.rnd_id_list = []
        self.numdelivers = number_delivers
        self.n_vehicles = n_vehicles
        self.delivery_info_rnd = {}

    def deliveryRndJson(self, delivery_file = './cfg/delivery_info.json'):
        with open(delivery_file) as _file:
            self.delivery_info = json.load(_file)

        for i in range(self.numdelivers):
            dup_flag = True
            rnd_id = random.choice(list(self.delivery_info.keys()))
            while dup_flag:  # because maybe we have duplicate random value
                if self.rnd_id_list.count(rnd_id) == 0:  # we don't have duplicate item in dictionary
                    dup_flag = False
                    self.rnd_id_list.append(rnd_id)
                    self.delivery_info_rnd[str(rnd_id)] = self.delivery_info[str(rnd_id)]
                else:
                    rnd_id = random.choice(list(self.delivery_info.keys()))

        i = 1
        delivery_info_rnd_new = {}

        for _keys, ele in self.delivery_info_rnd.items():
            ele['id'] = i
            delivery_info_rnd_new[i] = self.delivery_info_rnd[_keys]
            i += 1

        with open("./cfg/delivery_info_rnd.json", "w") as outfile:
            json.dump(delivery_info_rnd_new, outfile, indent=2)

    def settingRnd(self, setting_file = "./cfg/setting_1.json"):
        fp = open(setting_file, 'r')
        self.settings = json.load(fp)
        self.settings['n_deliveries'] = self.numdelivers
        self.settings['n_vehicles'] = self.n_vehicles
        return self.settings

    def distMatrixRnd(self, distance_matrix = './cfg/distance_matrix.csv'):
        self.rnd_id_list = [1] + [int(i) + 1 for i in self.rnd_id_list]
        db_new = pd.DataFrame()
        db = pd.read_csv(distance_matrix, header=None)
        for row, i in enumerate(self.rnd_id_list):
            for col, j in enumerate(self.rnd_id_list):
                db_new.loc[row + 1, col + 1] = db.iloc[i, j]
        db_new.to_csv('./cfg/dist_matrix_rnd.csv', header=None, index=None)

    def run(self, delivery_file = './cfg/delivery_info.json',
            setting_file = "./cfg/setting_1.json",
            distance_matrix = './cfg/distance_matrix.csv'):
        self.deliveryRndJson(delivery_file)
        settings = self.settingRnd(setting_file)
        self.distMatrixRnd(distance_matrix)
        return settings