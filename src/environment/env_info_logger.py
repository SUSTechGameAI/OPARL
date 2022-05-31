"""
  @Time : 2021/9/27 20:25 
  @Author : Ziqi Wang
  @File : env_info_logger.py
"""

import json
from abc import abstractmethod


# class EnvLogger:
#     @abstractmethod
#     def on_step(self, **kwargs):
#         pass
#
#     @abstractmethod
#     def close(self):
#         pass
import numpy as np


class InfoCollector:
    ignored_keys = {'episode', 'terminal_observation', 'LevelStr'}
    save_itv = 1000

    def __init__(self, path, log_itv=-1, log_targets=None):
        self.data = []
        self.path = path
        self.stdout_itv = log_itv
        self.time_before_save = InfoCollector.save_itv
        self.stdout_ptr = 0
        self.log_targets = [] if log_targets is None else log_targets

    def on_step(self, dones, infos):
        for done, info in zip(dones, infos):
            if done:
                self.data.append({
                    key: val for key, val in info.items()
                    if key not in InfoCollector.ignored_keys
                })
                self.time_before_save -= 1
        if self.time_before_save <= 0:
            with open(f'{self.path}/ep_infos.json', 'w') as f:
                json.dump(self.data, f)
            self.time_before_save += InfoCollector.save_itv
        if self.log_targets and 0 < self.stdout_itv <= (len(self.data) - self.stdout_ptr):
            keys = set(self.data[-1].keys()) - InfoCollector.ignored_keys - {'TotalSteps', 'TimePassed', 'rewards'}

            msg = '%sTotal steps: %d%s\n' % ('-' * 16, self.data[-1]['TotalSteps'], '-' * 16)
            msg += 'Time passed: %ds\n' % self.data[-1]['TimePassed']
            for key in keys:
                values = [item[key] for item in self.data[self.stdout_ptr:]]
                values = np.array(values)
                msg += '%s: %.2f +- %.2f\n' % (key, values.mean(), values.std())
            if 'file' in self.log_targets:
                with open(f'{self.path}/mylog.txt', 'a') as f:
                    f.write(msg + '\n')
            if 'std' in self.log_targets:
                print(msg)
            self.stdout_ptr = len(self.data)
            pass

    def close(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f)


# class InfoMaker:
#     def __init__(self, name=None):
#         self.name = self.__class__.__name__ if not name else name
#         self.content = None
#
#     @abstractmethod
#     def on_step(self, **kwargs):
#         pass
#
#     def on_reset(self):
#         self.content = None
#
#     def reset(self):
#         c = self.content
#         del self.content
#         self.on_reset()
#         return self.name, c
#
#
# class DivSeqInfoMaker(InfoMaker):
#     def __init__(self, m=10, n=10):
#         assert m >= n
#         super(DivSeqInfoMaker, self).__init__('DivSeq')
#         self.content = {}
#         self.m = m
#         self.n = n
#         # self.count = 0
#
#     def on_step(self, **kwargs):
#         t = kwargs['t'] # t start from 1
#         archive = kwargs['archive']
#         seg = kwargs['seg']
#         if archive.capacity < self.n:
#             raise RuntimeError('Archive capacity (%d) must be lager than n (%d)' % (archive.capacity, self.n))
#         if t > self.m:
#             self.content[f't={t}'] = []
#
#     def on_reset(self):
#         self.content = {}

# class DivSeqLogger(EnvLogger):
#     save_itv = 1000
#
#     def __init__(self, path, n=10):
#         self.data = []
#         self.path = path
#         self.n = n
#         self.times_before_save = TrainingLogger.save_itv
#
#     def on_step(self, **kwargs):
#
#         pass
#
#     def close(self):
#         pass
