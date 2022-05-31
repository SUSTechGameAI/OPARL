"""
  @Time : 2022/4/19 13:22 
  @Author : Ziqi Wang
  @File : ol_generator.py 
"""
import numpy as np
import torch

from src.gan.gan_config import nz
from src.gan.gan_use import process_levels
from src.repairer.repairer import Repairer
from src.utils.datastruct import RingQueue



class OnlineGenerator:
    def __init__(self, designer, generator, n, g_device='cuda:0'):
        self.designer = designer
        self.generator = generator
        self.generator.to(g_device)
        self.g_device = g_device
        self.obs_buffer = RingQueue(n)
        self.repairer = Repairer()
        for _ in range(n):
            latvec = (np.random.rand(nz) * 2 - 1).astype(np.float32)
            self.obs_buffer.push(latvec)

        z = torch.tensor(self.obs_buffer.rear(), device=self.g_device).view(-1, nz, 1, 1)
        seg = process_levels(self.generator(z), True)[0]
        self.init_seg_str = self.repairer.repair(seg, time_budget=0.5)

    def step(self):
        obs = np.concatenate(self.obs_buffer.to_list())
        latvec = self.designer.step(obs)
        self.obs_buffer.push(latvec)
        z = torch.tensor(latvec, device=self.g_device).view(-1, nz, 1, 1)
        seg = process_levels(self.generator(z), True)[0]
        seg = self.repairer.repair(seg, time_budget=0.5)
        seg_str = ('-' * seg.w + '\n') * max(0, 16 - seg.h)  + str(seg)
        return seg_str


