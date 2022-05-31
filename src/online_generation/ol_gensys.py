# """
#   @Time : 2022/2/9 14:05
#   @Author : Ziqi Wang
#   @File : ol_gensys.py
# """
import random
from enum import Enum

import numpy as np
import torch
from smb import MarioProxy
from src.designer.use_designer import Designer
from src.gan.gan_use import get_generator, process_levels
from src.online_generation.controller import FCLS_KNN_Controller, IdealFeatureSequence
from src.repairer.repairer import Repairer
from src.gan.gan_config import nz
from src.utils import RingQueue, get_path
from config import archive_len, ctrl_sig_dup
from src.level_diffs import naive_difficulty_measure


class ResampleStrategies(Enum):
    NO_RESAMPLE = 0
    RAND = 1
    REGEN = 2


class AggregatedGenerator:
    resample_limits = 1

    def __init__(self, designer_path, booster=None, repairer=None, resample=ResampleStrategies.REGEN, booster_device='cuda:0'):
        self.designer = Designer(designer_path)
        self.booster = get_generator(device=booster_device) if booster is None else booster
        self.booster_device = booster_device
        self.repairer = Repairer() if repairer is None else repairer
        self.resample_strategy = resample
        self.simulator = MarioProxy() if resample != ResampleStrategies.NO_RESAMPLE else None
        self.archive = RingQueue(archive_len)
        self.reset()
        pass

    def decode_latvec(self, z):
        one_hot_seg = self.booster(
            torch.tensor(z, dtype=torch.float32, device=self.booster_device).view(1, nz, 1, 1)
        )
        seg = process_levels(one_hot_seg, True)[0]
        return seg

    def generate(self, previous_seg, ctrl_sig=None):
        obs = self.__get_obs(ctrl_sig)
        z = self.designer.act(obs)

        # one_hot_seg = self.booster(torch.tensor(z).view(1, nz, 1, 1))
        # seg = process_levels(one_hot_seg, True)[0]
        seg = self.decode_latvec(z)
        seg = self.repairer.repair(seg, time_budget=0.5)

        if self.simulator is None or self.__check_playable(previous_seg, seg):
            self.archive.push(z)
            return seg, False, False
        resample_fail = True
        for _ in range(AggregatedGenerator.resample_limits):
            if self.resample_strategy == ResampleStrategies.RAND:
                z = np.random.rand(nz) * 2 - 1
            elif self.resample_strategy == ResampleStrategies.REGEN:
                z = self.designer.act(obs)
            # one_hot_seg = self.booster(torch.tensor(z, dtype=torch.float32).view(1, nz, 1, 1))
            # seg = process_levels(one_hot_seg, True)[0]
            seg = self.decode_latvec(z)
            seg = self.repairer.repair(seg, time_budget=0.5)
            if self.__check_playable(previous_seg, seg):
                resample_fail = True
                break
        self.archive.push(z)
        return seg, self.resample_strategy == ResampleStrategies.RAND, resample_fail

    def __check_playable(self, previous_seg, seg):
        tmp = seg if previous_seg is None else previous_seg + seg
        return self.simulator.simulate_game(tmp)['status'] == 'WIN'

    def __get_obs(self, ctrl_sig):
        lack = archive_len - len(self.archive)
        pad = [np.zeros([nz], np.float32) for _ in range(lack)]
        duped_ctrl_sig = np.zeros([ctrl_sig_dup], np.float32)
        if not ctrl_sig is None:
            duped_ctrl_sig[:] = ctrl_sig

        return np.concatenate([*pad, *self.archive.to_list(), duped_ctrl_sig]).astype(np.float32)

    def reset(self, z=None):
        self.archive.clear()
        z0 = np.random.rand(nz) * 2 - 1 if z is None else z
        self.archive.push(z0)
        pass


class OnlineGenerationSystem:
    def __init__(self, generator, featr_path=None, disable_ctrl=False):
        self.generator = generator
        self.controller = FCLS_KNN_Controller(IdealFeatureSequence(featr_path))
        self.noisy_flag = False
        self.eps_out = 0
        self.f_hat_seq = []
        self.f_real_seq = []
        self.prev_seg = None
        self.disable_ctrl = disable_ctrl
        self.resample_fails = 0
        pass

    def step(self):
        ctrl_sig = 0 if self.disable_ctrl else self.controller.make_decision()
        seg, self.noisy_flag, resample_fail = self.generator.generate(self.prev_seg, ctrl_sig)
        if resample_fail:
            self.resample_fails += 1
        seg_diff = naive_difficulty_measure(seg=seg)
        self.eps_out += abs(ctrl_sig - seg_diff)
        self.f_hat_seq.append(ctrl_sig)
        self.f_real_seq.append(seg_diff)
        self.prev_seg = seg
        return seg

    def recieve_play_data(self, delta, no_record=False):
        self.controller.recieve_play_data(delta, no_record)
        pass

    def reset(self, z=None):
        self.generator.reset(z)
        self.f_hat_seq.clear()
        return self.generator.decode_latvec(z)
        pass


