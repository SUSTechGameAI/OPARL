"""
  @Time : 2022/1/1 16:55 
  @Author : Ziqi Wang
  @File : level_divs.py 
"""

from enum import Enum


def naive_difficulty_measure(**kwargs):
    seg = kwargs['seg']
    # return (3 * seg.n_enemies + seg.w - seg.n_grounds) / seg.w
    return (seg.n_enemies + seg.w - seg.n_grounds) / seg.w


class ContentDifficultyMetrics(Enum):
    Naive = 0

    def get_func(self):
        if self.name == 'Naive':
            return naive_difficulty_measure

    def __str__(self):
        return self.name

