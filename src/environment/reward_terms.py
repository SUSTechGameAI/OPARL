"""
  @Time : 2022/2/7 10:57 
  @Author : Ziqi Wang
  @File : reward_terms.py 
"""
import random

from smb import level_sum
from src.environment.reward_func import RewardFuncTerm
from src.level_divs import ContentDivergenceMetrics
from src.level_diffs import ContentDifficultyMetrics


class Fun(RewardFuncTerm):
    default_kwargs = {
        'novelty_metric': ContentDivergenceMetrics.TilePttrKL2,
        'n': 3, 'lb': 0.26, 'ub': 0.94, 'delta': 14
    }

    def __init__(self, magnitude=1, **kwargs):
        cfgs = Fun.default_kwargs.copy()
        cfgs.update(kwargs)
        super(Fun, self).__init__(magnitude, False, **cfgs)
        self.metric = cfgs['novelty_metric'].get_func()
        self.ub = cfgs['ub']
        self.lb = cfgs['lb']
        self.n = cfgs['n']
        self.delta = cfgs['delta']

    def compute_reward(self, **kwargs):
        seg = kwargs['seg']
        archive = kwargs['archive']
        if not archive:
            return 0
        w = seg.w
        history = level_sum(archive)
        div_sum = 0.
        for k in range(self.n):
            s, e = history.w - k * self.delta - w, history.w - k * self.delta
            if s < 0:
                break
            cmp_seg = history[:, s:e]
            div_sum += self.metric(seg, cmp_seg)
        div = div_sum / self.n
        print(div, -min(0., div - self.lb, self.ub - div) ** 2)
        return -min(0., div - self.lb, self.ub - div) ** 2


class Playability(RewardFuncTerm):
    def __init__(self, magnitude=1):
        super(Playability, self).__init__(magnitude, True)

    def compute_reward(self, **kwargs):
        simlt_res = kwargs['simlt_res']
        return -1 if simlt_res['status'] != 'WIN' else 0


class Controllability(RewardFuncTerm):
    default_kwargs = {'diffculty_metric': ContentDifficultyMetrics.Naive, 'diff_range': (0, 1)}

    def __init__(self, magnitude=1, **kwargs):
        cfgs = Controllability.default_kwargs.copy()
        cfgs.update(kwargs)
        super(Controllability, self).__init__(magnitude, False, **cfgs)
        self.metric = cfgs['diffculty_metric'].get_func()
        self.dmin, self.dmax = cfgs['diff_range']
        self.tar_diff = random.uniform(self.dmin, self.dmax)
        self.sigma = (self.dmax - self.dmin) / 20

    def compute_reward(self, **kwargs):
        generate_diff = self.metric(**kwargs)
        reward = 1 - abs(generate_diff - self.tar_diff) / self.dmax

        xi = random.gauss(0, self.sigma)
        while not self.dmin <= self.tar_diff + xi <= self.dmax:
            xi = random.gauss(0, self.sigma)
        self.tar_diff = self.tar_diff + xi
        return reward

    def on_reset(self):
        self.tar_diff = random.uniform(self.dmin, self.dmax)


class FunTest(RewardFuncTerm):
    default_kwargs = {
        'novelty_metric': ContentDivergenceMetrics.TilePttrKL2,
        'n': 3, 'lb': 0.26, 'ub': 0.94, 'delta': 14
    }

    def __init__(self, magnitude=1, **kwargs):
        cfgs = FunTest.default_kwargs.copy()
        cfgs.update(kwargs)
        super(FunTest, self).__init__(magnitude, False, **cfgs)
        self.metric = cfgs['novelty_metric'].get_func()
        self.ub = cfgs['ub']
        self.lb = cfgs['lb']
        self.n = cfgs['n']
        self.delta = cfgs['delta']

    def compute_reward(self, **kwargs):
        seg = kwargs['seg']
        archive = kwargs['archive']
        if not archive:
            return 0
        w = seg.w
        history = level_sum(archive)
        div_sum = 0.
        for k in range(self.n):
            s, e = history.w - k * self.delta - w, history.w - k * self.delta
            if s < 0:
                break
            cmp_seg = history[:, s:e]
            div_sum += self.metric(seg, cmp_seg)
        div = div_sum / self.n
        return max(0., self.lb - div, div - self.ub)
