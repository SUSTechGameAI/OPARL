"""
  @Time : 2022/2/11 13:39 
  @Author : Ziqi Wang
  @File : main.py 
"""

import time
import json

import matplotlib.pyplot as plt

from smb import MarioLevel, MarioProxy
from src.utils import get_path


if __name__ == '__main__':
    unit = 0.023219954648526078
    s_second = 44.123
    with open(get_path("assets/Ginseng_diffs.json"), 'r') as f:
        data = json.load(f)
    s = int(s_second / unit)
    # with open(get_path("assets/Ginseng_diffs_clipped.json"), 'w') as f:
    #     json.dump(data[s:], f)

    x = [i * unit for i in range(int(60/unit))]
    y = data[:int(60/unit)]
    plt.figure(figsize=(12.8, 3.2))
    plt.plot(x, y)
    plt.plot(x[s:], [v - 0.1 for v in data[s:int(60/unit)]])
    plt.show()
    # lvl = MarioLevel.from_txt('misc/hard-segment/1-1.txt')
    # proxy = MarioProxy()
    # start_time = time.time()
    # print(proxy.simulate_game(lvl)['status'])
    # print(time.time() - start_time)
    # lvl2 = MarioLevel.from_txt('misc/hard-segment/1-2.txt')
    # lvl = lvl + lvl2
    # start_time = time.time()
    # print(proxy.simulate_game(lvl)['status'])
    # print(time.time() - start_time)

    # for i in range(5):
    #     lvl = MarioLevel.from_txt(f'misc/lvl-players-demos/sample{i}.txt')[:, :84]
    #     lvl.to_img(f'misc/lvl-players-demos/agent{i}.png')
