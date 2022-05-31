"""
  @Time : 2022/2/11 14:00 
  @Author : Ziqi Wang
  @File : demo_bar_plt.py 
"""

import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects, **text_kwargs):
    for rect in rects:
        height = rect.get_height()
        plt.text(
            rect.get_x()+rect.get_width()/2., 0.5 * height, '%.2f' % float(height),
            ha="center", va="center", **text_kwargs
        )

if __name__ == '__main__':
    # metric = '\sqrt{F}'
    # metric = 'C'
    metric = 'D'

    plt.figure(figsize=(7.2, 2.4), dpi=200)
    size = 5
    x = np.arange(size)
    a = np.random.random(size) / 3 + 0.66
    b = np.random.random(size) / 3 + 0.66
    # c = np.random.random(size)

    total_width, n = 1.2, 3
    width = total_width / n
    x = x

    dda_bars = plt.bar(x - width / 2, a, width=width, label='DDA controller')
    mi_bars = plt.bar(x + width / 2, b, width=width, label='MI controller')
    # plt.bar(x + 2 * width, c, width=width, label='c')
    autolabel(dda_bars, color='white')
    autolabel(mi_bars, color='white')
    plt.xticks(x, labels=['agent1', 'agent2', 'agent3', 'agent4', 'agent5'])
    plt.title(f'Evaluation scores of ${metric}$ for different agents')
    plt.legend()
    plt.show()
    pass
