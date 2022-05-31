"""
  @Time : 2022/2/7 14:31 
  @Author : Ziqi Wang
  @File : orilvl_statics.py 
"""
import json

from src.utils import get_path
from smb import traverse_level_files, MarioLevel


def test_difficulties(metric, save_path='./', save_name='difficulties.json'):
    results = {}
    for lvl, name in traverse_level_files():
        results[name] = []
        for s in range(lvl.w - MarioLevel.default_seg_width):
            seg = lvl[:, s:s+MarioLevel.default_seg_width]
            # print(seg.n_grounds)
            results[name].append(metric(seg=seg))
        print(name, results[name])
    with open(get_path(f'{save_path}/{save_name}'), 'w') as f:
        json.dump(results, f)
    pass


if __name__ == '__main__':
    from src.level_diffs import ContentDifficultyMetrics
    test_difficulties(
        ContentDifficultyMetrics.Naive.get_func(),
        'exp_data/orilvl_statics/intermediate',
        'naive_difficulties.json'
    )
