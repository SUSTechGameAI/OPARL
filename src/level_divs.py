"""
  @Time : 2022/1/1 16:55 
  @Author : Ziqi Wang
  @File : level_divs.py 
"""

import math
import numpy as np
from enum import Enum
from smb import MarioLevel
from scipy.stats import entropy


def tile_diff_rate(seg1: MarioLevel, seg2: MarioLevel):
    h, w = seg1.shape
    difference, *_ = np.where(seg1.content != seg2.content)
    return len(difference) / (h * w)

def tile_pattern_kl_div(seg1: MarioLevel, seg2: MarioLevel, w):
    eps = 1e-3
    counts1 = seg1.tile_pattern_counts(w)
    counts2 = seg2.tile_pattern_counts(w)
    all_keys = counts1.keys().__or__(counts2.keys())
    revised_counts1 = np.array([counts1.setdefault(key, 0) + eps for key in all_keys])
    revised_counts2 = np.array([counts2.setdefault(key, 0) + eps for key in all_keys])
    return entropy(revised_counts1, revised_counts2)
#
# def tile_pattern_js_div(seg1: MarioLevel, seg2: MarioLevel, w):
#     counts1 = seg1.tile_pattern_counts(w)
#     counts2 = seg2.tile_pattern_counts(w)
#     all_keys = counts1.keys().__or__(counts2.keys())
#     p = np.array([counts1.setdefault(key, 0) for key in all_keys])
#     q = np.array([counts2.setdefault(key, 0) for key in all_keys])
#     return (entropy(p, p + q, base=2) + entropy(q, p + q, base=2)) / 2
#

class ContentDivergenceMetrics(Enum):
    TileDiffRate = 0
    TilePttrKL2 = 1
    TilePttrKL3 = 2
    TilePttrKL4 = 3

    def get_func(self):
        if self.name == 'TileDiffRate':
            return tile_diff_rate
        elif self.name == 'TilePttrKL2':
            return lambda a, b: tile_pattern_kl_div(a, b, 2)
        elif self.name == 'TilePttrKL3':
            return lambda a, b: tile_pattern_kl_div(a, b, 3)
        elif self.name == 'TilePttrKL4':
            return lambda a, b: tile_pattern_kl_div(a, b, 4)

    def __str__(self):
        return self.name

