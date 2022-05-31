"""
  @Time : 2022/2/14 20:26 
  @Author : Ziqi Wang
  @File : controller.py 
"""
import json
import random
from abc import abstractmethod

from src.utils import RingQueue, get_path

#
# class OPA_Controller:
#     @abstractmethod
#     def recieve_play_data(self):
#         pass
#
#     @abstractmethod
#     def make_decision(self):
#         pass
#
#
# class DDA_Controller:
#     pass
#
#
# class MDFC_Controller:
#     pass


class IdealFeatureSequence:
    def __init__(self, featr_seq_path=None):
        self.content = []
        if featr_seq_path is None:
            self.content = [0.5] * 30000
        else:
            with open(get_path(featr_seq_path), 'r') as f:
                self.content = json.load(f)
        with open(get_path('assets/time_unit.json'), 'r') as f:
            self.unit = json.load(f)
        self.mean = sum(self.content) / len(self.content)
        pass

    def get(self, s, e):
        i = int(s // self.unit)
        j = int(e // self.unit)
        return self.content[i:j]
        pass
    pass


class FCLS_KNN_Controller:
    def __init__(self, ideal_featr_seq, m=20, trials=50, k=5, sigma=0.02):
        self.ideal_featrs = ideal_featr_seq
        self.begin_time = 0
        self.knn_archive = RingQueue(m)
        self.trials = trials
        self.k = k
        self.sigma = sigma
        self.prev_f_hat = self.ideal_featrs.mean
        self.cur_f_hat = self.ideal_featrs.mean
        pass

    def make_decision(self):
        if not len(self.knn_archive):
            delta_tilde = 2
        else:
            tmp = self.knn_archive.to_list()
            tmp.sort(key=lambda x: abs(x['f_hat'] - self.prev_f_hat))
            neighbors = tmp[:min(len(tmp), self.k)]
            delta_tilde = sum(item['delta'] for item in neighbors) / len(neighbors)
        b_tilde = self.begin_time + delta_tilde

        f_hat = self.prev_f_hat
        min_eps = self.estimation(f_hat, b_tilde)
        for _ in range(self.trials):
            f_hat_prime = f_hat + random.gauss(0, self.sigma)
            eps_tilde = self.estimation(f_hat_prime, b_tilde)
            if eps_tilde < min_eps:
                f_hat = f_hat_prime
                min_eps = eps_tilde
        self.prev_f_hat = self.cur_f_hat
        self.cur_f_hat = f_hat
        return f_hat
        pass

    def estimation(self, f_hat, b_tilde):
        if not len(self.knn_archive):
            delta_tilde = 2
        else:
            tmp = self.knn_archive.to_list()
            tmp.sort(key=lambda x: abs(x['f_hat'] - f_hat))
            neighbors = tmp[:min(len(tmp), self.k)]
            delta_tilde = sum(item['delta'] for item in neighbors) / len(neighbors)
        eps_tilde = sum(abs(x-f_hat) for x in self.ideal_featrs.get(b_tilde, b_tilde+delta_tilde))
        return eps_tilde / delta_tilde

    def recieve_play_data(self, delta, no_record):
        self.begin_time += delta
        if not no_record and delta > 0.:
            self.knn_archive.push({'f_hat': self.prev_f_hat, 'delta': delta})
        self.prev_f_hat = self.cur_f_hat
        pass



