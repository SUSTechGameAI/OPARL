"""
  @Time : 2022/2/22 13:35 
  @Author : Ziqi Wang
  @File : make_err_illustrate.py 
"""
import random

import numpy as np
from math import pi
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.figure(figsize=(19.2, 4.8))
    times = np.linspace(0, 2.5 * pi, 1000)
    f_hat = np.sin(times) * 0.7
    breaks = [377, 674]


    f_tilde = [
        [sum(f_hat[:breaks[0]]) / breaks[0] - 0.05] * breaks[0],
        [sum(f_hat[breaks[0]:breaks[1]]) / (breaks[1] - breaks[0]) + 0.02] * (breaks[1] - breaks[0]),
        [sum(f_hat[breaks[1]:]) / (1000 - breaks[1]) + 0.03] * (1000 - breaks[1]),
    ]
    plt.fill_between(times[:breaks[0]], f_tilde[0], f_hat[:breaks[0]], alpha=0, hatch='/', label='inner error')
    plt.fill_between(times[breaks[0]:breaks[1]], f_tilde[1], f_hat[breaks[0]:breaks[1]], alpha=0, hatch='/')
    plt.fill_between(times[breaks[1]:], f_tilde[2], f_hat[breaks[1]:], alpha=0, hatch='/')

    f = [
        [f_tilde[0][0] + 0.18] * breaks[0],
        [f_tilde[1][0] + 0.15] * (breaks[1] - breaks[0]),
        [f_tilde[2][0] - 0.17] * (1000 - breaks[1])
    ]
    plt.fill_between(times[:breaks[0]], f[0], f_tilde[0], alpha=0, hatch='\\', label='outer error')
    plt.fill_between(times[breaks[0]:breaks[1]], f[1], f_tilde[1], alpha=0, hatch='\\')
    plt.fill_between(times[breaks[1]:], f[2], f_tilde[2], alpha=0, hatch='\\')


    plt.fill_between(times[:breaks[0]], f[0], f_hat[:breaks[0]], color='black', alpha=0.16, label='overall error')
    plt.fill_between(times[breaks[0]:breaks[1]], f[1], f_hat[breaks[0]:breaks[1]], color='black', alpha=0.16)
    plt.fill_between(times[breaks[1]:], f[2], f_hat[breaks[1]:], color='black', alpha=0.16)

    plt.plot(times, f_hat, color='black', lw=3, label='ideal feature')
    plt.plot([times[breaks[0]], times[breaks[0]]], [-1, 1], color='gray', ls='--', lw=3)
    plt.plot([times[breaks[1]], times[breaks[1]]], [-1, 1], color='gray', ls='--', lw=3)

    plt.plot(times[:breaks[0]], f_tilde[0], color='blue', lw=3, label='target feature')
    plt.plot(times[breaks[0]:breaks[1]], f_tilde[1], color='blue', lw=3)
    plt.plot(times[breaks[1]:], f_tilde[2], color='blue', lw=3)

    plt.plot(times[:breaks[0]], f[0], color='red', lw=3, label='real feature')
    plt.plot(times[breaks[0]:breaks[1]], f[1], color='red', lw=3)
    plt.plot(times[breaks[1]:], f[2], color='red', lw=3)

    plt.ylim((-1, 1))
    ########## Eliminate ticks ##########
    plt.xticks([], [])
    plt.yticks([], [])
    #####################################

    plt.legend(ncol=2, fontsize=18)
    plt.show()
