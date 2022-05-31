"""
  @Time : 2022/5/27 18:46 
  @Author : Ziqi Wang
  @File : make_duration_curves.py 
"""

import json
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_path


musics = ['Ginseng', 'Farewell']
agents = ['Baumgarten', 'Sloane', 'Hartmann', 'Polikarpov', 'Schumann']
# colors6 = ['#A0DB19', '#34C55C', '#00A980', '#00898B', '#00687C', '#2F4858']
colors5 = ['#712AE8', '#ED00B3', '#FF1E7A', '#FF7E4F', '#FFC144']

def set_common_and_show():
    plt.legend(loc='upper left')
    plt.ylim((0, 375))
    plt.xlabel('Number of segments')
    plt.ylim((0, 375))
    axes = fig.get_axes()[0]
    axes.set_axisbelow(True)
    plt.tight_layout()
    plt.grid()
    plt.show()

def get_y_data(music, agent_name):
    with open(get_path(f'exp_data/sac/fcp/{music}_{agent_name}/simulation_res.json'), 'r') as f:
        values = np.array([item['ends'] for item in json.load(f)]).transpose((1, 0))
    mean = values.mean(axis=1)
    lower = mean - values.std(axis=1)
    upper = mean + values.std(axis=1)
    return mean, lower, upper


if __name__ == '__main__':
    n = 51
    x = [*range(n)]
    fig = plt.figure(figsize=(4, 3), dpi=300)
    for i, agent in enumerate(agents):
        y, yb, yt = get_y_data('Ginseng', agent)
        plt.plot(x, y[:n], color=colors5[i], label=agent)
        plt.fill_between(x, yb[:n], yt[:n], color=colors5[i], alpha=0.2)
        # with open(get_path(f'exp_data/sac/fcp/time_test/Ginseng_{agent}.json'), 'r') as f:
        #     y = json.load(f)
        # plt.plot(x, y[:n], color=colors5[i], label=agent)
    plt.title('Ginseng', size=14)
    set_common_and_show()

    plt.figure(figsize=(4, 3), dpi=300)
    for i, agent in enumerate(agents):
        y, yb, yt = get_y_data('Farewell', agent)
        plt.plot(x, y[:n], color=colors5[i], label=agent)
        plt.fill_between(x, yb[:n], yt[:n], color=colors5[i], alpha=0.2)
        # with open(get_path(f'exp_data/sac/fcp/time_test/Farewell_{agent}.json'), 'r') as f:
        #     y = json.load(f)
        # plt.plot(x, y[:n], color=colors5[i], label=agent)
    plt.title('Farewell', size=14)
    set_common_and_show()
