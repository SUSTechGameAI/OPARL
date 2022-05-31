"""
  @Time : 2022/1/4 10:03 
  @Author : Ziqi Wang
  @File : rfuncs.py 
"""

from src.environment.reward_func import RewardFunction
from src.environment.rewterms.playability import *
from src.environment.rewterms.fun import *


fun_js2 = RewardFunction(
    SlackingA_ClippedDivReward(novelty_metric=ContentDivergenceMetrics.TilePttrJS2)
)

fun_js3 = RewardFunction(
    SlackingA_ClippedDivReward(novelty_metric=ContentDivergenceMetrics.TilePttrJS3)
)

fun_js23 = RewardFunction(
    SlackingA_ClippedDivReward(novelty_metric=ContentDivergenceMetrics.TilePttrJS23)
)

