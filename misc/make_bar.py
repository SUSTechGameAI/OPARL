"""
  @Time : 2022/2/18 19:33 
  @Author : Ziqi Wang
  @File : make_bar.py 
"""
import json
from itertools import permutations

import numpy as np
from matplotlib import pyplot as plt

from smb import traverse_level_files
from src.utils import get_path


def autolabel(rects, size=15, fmt='%.2f'):
    for rect in rects:
        height = rect.get_height()
        plt.text(
            rect.get_x()+rect.get_width()/2., 0.5 * height, fmt % float(height),
            ha="center", va="center", size=size, color='white'
        )

def compute_diversity(path):
    levels = [lvl for lvl, _ in traverse_level_files(path)]
    div_sum, n = 0., 0
    for lvl1, lvl2 in permutations(levels, 2):
        div_sum += len(np.where(lvl1.content != lvl2.content)[0]) / (lvl1.h * lvl1.w)
        n += 1
    return div_sum / n
    pass


musics = ['Ginseng', 'Farewell']
agents = ['Baumgarten', 'Sloane', 'Hartmann', 'Polikarpov', 'Schumann']

def get_bar_vals(music):
    res = []
    for agent_name in agents:
        if metric == 'diversity':
            d = compute_diversity(f'exp_data/sac/fcp/{music}_{agent_name}')
            res.append(d)
            continue
        with open(get_path(f'exp_data/sac/fcp/{music}_{agent_name}/simulation_res.json'), 'r') as f:
            data = json.load(f)
        if metric == 'fun':
            vals = [1 - item['fun'] for item in data]
        else:
            vals = [1 - item['eps_all'] for item in data]
        res.append(sum(vals) / len(vals))
    return res


if __name__ == '__main__':
    metric = 'overall~error'
    # metric = 'fun'
    # metric = 'diversity'

    # plt.figure(figsize=(6.4, 1.4), dpi=200, tight_layout={'w_pad': 5, 'h_pad': 5})
    plt.figure(figsize=(12.8, 2.8), dpi=200, tight_layout={'w_pad': 15, 'h_pad': 5})
    # size = 5
    x = np.arange(len(agents))

    a = get_bar_vals(musics[0])
    b = get_bar_vals(musics[1])

    total_width, n = 1.2, 3
    width = total_width / n
    x = x

    ginseng_bars = plt.bar(x - width / 2, a, width=width, label='Ginseng')
    farewell_bars = plt.bar(x + width / 2, b, width=width, label='Farewell')
    # plt.bar(x + 2 * width, c, width=width, label='c')

    if metric == 'diversity':
        autolabel(ginseng_bars, size=14, fmt='%.3f')
        autolabel(farewell_bars, size=14, fmt='%.3f')
    else:
        autolabel(ginseng_bars)
        autolabel(farewell_bars)

    if metric == 'diversity':
        plt.ylim((0, 0.1))
    elif metric == 'fun':
        plt.ylim((0, 1))
    else:
        plt.ylim((0, 1.5))

    plt.xticks(x, labels=agents, size=16)
    if metric == 'fun':
        plt.yticks(
            [0., 0.2, 0.4, 0.6, 0.8, 1.0],
            ['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'],
            size=14
        )
    else:
        plt.yticks(size=14)

    plt.title(f'Evaluation scores of ${metric}$ for different agents and musics', fontsize=20)
    if metric == 'overall~error':
        plt.legend(loc='upper center', ncol=2, fontsize=16)
        plt.plot([-0.4, len(agents) - 1 + 0.4], [1.0, 1.0], color=(0.3, 0.3, 0.3), ls='--')
    plt.show()
