"""
  @Time : 2022/2/18 18:32 
  @Author : Ziqi Wang
  @File : make_tab1.py 
"""

import json
import numpy as np
from itertools import permutations
from src.utils import get_path
from smb import traverse_level_files


def compute_diversity(path):
    levels = [lvl for lvl, _ in traverse_level_files(path)]
    div_sum, n = 0., 0
    for lvl1, lvl2 in permutations(levels, 2):
        div_sum += len(np.where(lvl1.content != lvl2.content)[0]) / (lvl1.h * lvl1.w)
        n += 1
    return div_sum / n
    pass


if __name__ == '__main__':
    headings = {
        'f': '$F$', 'p': '$P$', 'fp': '$F{+}P$', 'c': '$C$',
        'fc': '$C{+}F$', 'cp': '$C{+}P$', 'fcp': '$C{+}F{+}P$'
    }

    series1 = ['f', 'p', 'fp']
    series2 = ['c', 'fc', 'cp', 'fcp']


    for key in series1:
        with open(get_path(f'exp_data/sac/{key}/test_rewards.json'), 'r') as f:
            rewards = json.load(f)
        funs = np.array([item['FunTest'] for item in rewards])
        revised_funs = funs / 50
        playability = np.array([item['Playability'] for item in rewards])
        revised_ps = -playability / 50
        with open(get_path(f'exp_data/sac/{key}/Ginseng_Baumgarten/simulation_res.json'), 'r') as f:
            sim_res = json.load(f)
        fun_ol = np.array([item['fun'] for item in sim_res])
        playability_ol = np.array([item['playability'] for item in sim_res]) / 50
        eps_all = np.array([item['eps_all'] for item in sim_res])
        d = compute_diversity(f'exp_data/sac/{key}/Ginseng_Baumgarten')
        contents = [
            headings[key],
            '%.3g $\pm$ %.3g' % (100 * revised_funs.mean(), 100 * revised_funs.std()),
            '%.3g $\pm$ %.3g' % (100 * revised_ps.mean(), 100 * revised_ps.std()),
            '$-/-$',
            '%.3g $\pm$ %.3g' % (100 * fun_ol.mean(), 100 * fun_ol.std()),
            '%.3g $\pm$ %.3g' % (100 * playability_ol.mean(), 100 * playability_ol.std()),
            '$-/-$', '$-/-$',
            '%.3g $\pm$ %.3g' % (100 * eps_all.mean(), 100 * eps_all.std()),
            '%.3f' % d
        ]
        line = ' & '.join(contents) + ' \\\\'
        print(line)

    print('\midrule[0.8pt]')

    for key in series2:
        with open(get_path(f'exp_data/sac/{key}/test_rewards.json'), 'r') as f:
            rewards = json.load(f)
        funs = np.array([item['FunTest'] for item in rewards])
        revised_funs = funs / 50
        playability = np.array([item['Playability'] for item in rewards])
        revised_ps = -playability / 50
        controllability = np.array([item['Controllability'] for item in rewards])
        revised_cs = 1-controllability / 50
        with open(get_path(f'exp_data/sac/{key}/Ginseng_Baumgarten/simulation_res.json'), 'r') as f:
            sim_res = json.load(f)
        with open(get_path(f'exp_data/sac/{key}/Ginseng_Baumgarten/simulation_res1.json'), 'r') as f:
            sim_res += json.load(f)
        fun_ol = np.array([item['fun'] for item in sim_res])
        playability_ol = np.array([item['playability'] for item in sim_res]) / 50
        eps_in = np.array([item['eps_in'] for item in sim_res])
        eps_out = np.array([item['eps_out'] for item in sim_res])
        eps_all = np.array([item['eps_all'] for item in sim_res])

        d = compute_diversity(f'exp_data/sac/{key}/Ginseng_Baumgarten')
        contents = [
            headings[key],
            '%.3g $\pm$ %.3g' % (100 * revised_funs.mean(), 100 * revised_funs.std()),
            '%.3g $\pm$ %.3g' % (100 * revised_ps.mean(), 100 * revised_ps.std()),
            '%.3g $\pm$ %.3g' % (100 * revised_cs.mean(), 100 * revised_cs.std()),
            '%.3g $\pm$ %.3g' % (100 * fun_ol.mean(), 100 * fun_ol.std()),
            '%.3g $\pm$ %.3g' % (100 * playability_ol.mean(), 100 * playability_ol.std()),
            '%.3g $\pm$ %.3g' % (100 * eps_in.mean(), 100 * eps_in.std()),
            '%.3g $\pm$ %.3g' % (100 * eps_out.mean(), 100 * eps_out.std()),
            '%.3g $\pm$ %.3g' % (100 * eps_all.mean(), 100 * eps_all.std()),
            '%.3f' % d
        ]
        line = ' & '.join(contents) + ' \\\\'
        print(line)
