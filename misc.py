"""
  @Time : 2022/2/7 10:16 
  @Author : Ziqi Wang
  @File : misc.py 
"""

import torch
from src.gan.gan_use import get_generator
from src.utils import get_path
from src.level_diffs import ContentDifficultyMetrics
from smb import traverse_level_files, MarioLevel


def add(a, b):
    print(a + b)

if __name__ == '__main__':
    # model = get_generator()
    # torch.save(model.state_dict(), get_path('models/generator_state_dict.pth'))
    # metric = ContentDifficultyMetrics.Naive.get_func()
    # results = []
    # for lvl, _ in traverse_level_files():
    #     for s in range(lvl.w - MarioLevel.default_seg_width):
    #         seg = lvl[:, s:s+MarioLevel.default_seg_width]
    #         print(metric(seg=seg))
    #     pass
    # level = MarioLevel.from_txt('exp_data/main2/fcp/Ginseng_Baumgarten-old/lvl3.txt')[:, 560:700]
    # level.to_img('exp_data/main2/fcp/Ginseng_Baumgarten-old/lvl3.png')
    values = (2, 4)
    add(*values)
    pass

