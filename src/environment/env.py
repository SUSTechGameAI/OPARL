"""
@DATE: 2021/9/10
@Author: Ziqi Wang
@File: env.py
"""

import gym
import time
import torch
import numpy as np
from math import ceil
from copy import deepcopy
from torch import tensor
from typing import Optional, Callable, List
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs
from stable_baselines3.common.vec_env.subproc_vec_env import _flatten_obs
from src.environment.env_info_logger import InfoCollector
from src.repairer.repairer import Repairer
from src.utils import RingQueue
from src.gan.gan_use import *
from config import archive_len, ctrl_sig_dup
from smb import *


class GenerationEnv(gym.Env):
    def __init__(self, rfunc=None, max_seg_num=50):
        self.rfunc = deepcopy(rfunc)
        self.mario_proxy = MarioProxy()
        self.action_space = gym.spaces.Box(-1, 1, (nz,))
        self.observation_space = gym.spaces.Box(-1, 1, (archive_len * nz + ctrl_sig_dup,))
        self.ctrl_sig_dup = ctrl_sig_dup
        self.simlt_buffer = RingQueue(2)
        self.level_archive = RingQueue(archive_len)
        self.latvec_archive = RingQueue(archive_len)
        self.max_seg_num = max_seg_num
        self.archive_len = archive_len
        self.counter = 0
        self.score = 0
        self.all_segs = []
        self.repairer = Repairer()

        self.onehot_seg = None
        self.backup_latvec = None
        self.backup_onehot_seg = None

    def receive(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

    def step(self, action: np.ndarray):
        seg = MarioLevel.from_one_hot_arr(self.onehot_seg)
        seg = self.repairer.repair(seg, time_budget=0.5)
        self.latvec_archive.push(action)
        self.all_segs.append(seg)

        self.counter += 1
        self.simlt_buffer.push(seg)
        if not self.rfunc is None and self.rfunc.require_simlt:
            simlt_res = self.__playable_test()
        else:
            simlt_res = None

        archive = None if not len(self.level_archive) else self.level_archive.to_list()
        reward = 0 if self.rfunc is None else self.rfunc(archive=archive, seg=seg, simlt_res=simlt_res)

        done = self.counter >= self.max_seg_num
        self.score += reward

        obs = self.__get_obs()
        if done:
            info = {} if self.rfunc is None else self.rfunc.reset()
            info['TotalScore'] = self.score
            info['EpLength'] = self.counter
            info['LevelStr'] = str(level_sum(self.all_segs))
        else:
            info = {}
        self.level_archive.push(seg)
        return obs, reward, done, info

    def __playable_test(self):
        test_seg = level_sum(self.simlt_buffer.to_list())
        return self.mario_proxy.simulate_game(test_seg)

    def reset(self):
        self.simlt_buffer.clear()
        self.level_archive.clear()
        self.latvec_archive.clear()
        self.all_segs.clear()
        self.latvec_archive.push(self.backup_latvec)
        self.level_archive.push(MarioLevel.from_one_hot_arr(self.backup_onehot_seg))
        self.backup_latvec, self.backup_onehot_seg = None, None
        self.score = 0
        self.counter = 0
        return self.__get_obs()

    def __get_obs(self):
        lack = self.archive_len - len(self.latvec_archive)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        if self.rfunc is None:
            return np.concatenate([
                *pad, *self.latvec_archive.to_list(),
                np.zeros([self.ctrl_sig_dup], np.float32)
            ])
        ctrl_sig = np.zeros([self.ctrl_sig_dup], np.float32)
        for term in self.rfunc.terms:
            if term.__class__.__name__ == 'Controllability':
                ctrl_sig[:] = term.tar_diff
        return np.concatenate([*pad, *self.latvec_archive.to_list(), ctrl_sig])

    def render(self, mode='human'):
        pass


class VecGenerationEnv(SubprocVecEnv):
    def __init__(
        self, env_fns: List[Callable[[], gym.Env]], start_method: Optional[str] = None,
        log_path=None, log_itv=-1, log_targets=None, device='cuda:0'
    ):
        super(VecGenerationEnv, self).__init__(env_fns, start_method)
        self.generator = get_generator(device=device)

        if log_path:
            self.logger = InfoCollector(log_path, log_itv, log_targets)
        else:
            self.logger = None
        self.total_steps = 0
        self.start_time = time.time()
        self.device = device

    def step_async(self, actions: np.ndarray) -> None:
        with torch.no_grad():
            z = torch.clamp(tensor(actions.astype(np.float32), device=self.device), -1, 1).view(-1, nz, 1, 1)
            onehot_segs = process_levels(self.generator(z))
        for remote, onehot_seg in zip(self.remotes, onehot_segs):
            remote.send(("env_method", ('receive', [], {'onehot_seg': onehot_seg})))
        for remote in self.remotes:
            remote.recv()
        for remote, action, onehot_seg in zip(self.remotes, actions, onehot_segs):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        self.total_steps += self.num_envs
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)

        envs_to_send = [i for i in range(self.num_envs) if dones[i]]
        self.send_reset_data(envs_to_send)

        if self.logger is not None:
            for i in range(self.num_envs):
                if infos[i]:
                    infos[i]['TotalSteps'] = self.total_steps
                    infos[i]['TimePassed'] = time.time() - self.start_time
            self.logger.on_step(dones, infos)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        self.send_reset_data()
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        self.send_reset_data()
        return _flatten_obs(obs, self.observation_space)

    def send_reset_data(self, env_ids=None):
        if env_ids is None:
            env_ids = [*range(self.num_envs)]
        target_remotes = self._get_target_remotes(env_ids)
        latvecs = np.random.rand(len(env_ids), nz).astype(np.float32) * 2 - 1
        with torch.no_grad():
            z = tensor(latvecs).view(-1, nz, 1, 1).to(self.device)
            onehot_segs = process_levels(self.generator(z))
        for remote, latvec, onehot_seg in zip(target_remotes, latvecs, onehot_segs):
            kwargs = {'backup_latvec': latvec, 'backup_onehot_seg': onehot_seg}
            remote.send(("env_method", ('receive', [], kwargs)))
        for remote in target_remotes:
            remote.recv()

    def close(self) -> None:
        super().close()
        if self.logger is not None:
            self.logger.close()


def make_vec_generation_env(
        num_envs, rfunc=None, log_path=None, max_seg_num=50,
        log_itv=-1, device='cuda:0', log_targets=None
    ):
    return make_vec_env(
        GenerationEnv, n_envs=num_envs, vec_env_cls=VecGenerationEnv,
        vec_env_kwargs={
            'log_path': log_path,
            'log_itv': log_itv,
            'log_targets': log_targets,
            'device': device
        },
        env_kwargs={
            'rfunc': rfunc,
            'max_seg_num': max_seg_num,
        }
    )

