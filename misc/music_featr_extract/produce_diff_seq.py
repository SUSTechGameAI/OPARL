"""
  @Time : 2022/2/18 11:57 
  @Author : Ziqi Wang
  @File : produce_diff_seq.py 
"""
import json
import matplotlib.pyplot as plt
from src.utils import get_path

def mapping_energy_diff(energys):
    diffs = []
    for energy in energys:
        clipped = max(-2.5, min(energy, 0))
        diff = (clipped + 2.5) / 2.5
        diffs.append(diff)
    return diffs


if __name__ == '__main__':
    # with open(get_path('misc/music_featr_extract/Ginseng_energy_curve.json'), 'r') as f:
    #     energys_ginseng = json.load(f)
    # diffs_ginseng = mapping_energy_diff(energys_ginseng)
    # with open(get_path('misc/music_featr_extract/Ginseng_diffs.json'), 'w') as f:
    #     json.dump(diffs_ginseng, f)
    #
    # with open(get_path('misc/music_featr_extract/Farewell_energy_curve.json'), 'r') as f:
    #     energys_farewell = json.load(f)
    # diffs_farewell = mapping_energy_diff(energys_farewell)
    # with open(get_path('misc/music_featr_extract/Farewell_diffs.json'), 'w') as f:
    #     json.dump(diffs_farewell, f)
    # with open(get_path('misc/music_featr_extract/blended_energy_curve.json'), 'r') as f:
    #     energys_farewell = json.load(f)
    # diffs_farewell = mapping_energy_diff(energys_farewell)
    # with open(get_path('misc/music_featr_extract/blended_diffs.json'), 'w') as f:
    #     json.dump(diffs_farewell, f)

    with open(get_path('assets/blended_diffs.json'), 'r') as f:
        diffs_blended = json.load(f)

    # plt.plot([i * 0.023 for i in range(len(diffs_ginseng))], diffs_ginseng)
    # plt.plot([i * 0.023 for i in range(len(diffs_farewell))], diffs_farewell)
    plt.figure(figsize=(12.8, 3.2))
    plt.ylim((0, 1))
    plt.plot([i * 0.023219954648526078 for i in range(3200)], diffs_blended[:3200], lw=3)
    plt.ylabel('Difficulty', size=18)
    plt.text(1600 * 0.023219954648526078, 0.06, 'Real-Time', ha="center", va="center", size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    # plt.legend(fontsize=16)
    plt.show()

    # with open(get_path('assets/Ginseng_diffs.json'), 'r') as f:
    #     diffs_ginseng = json.load(f)
    #
    # with open(get_path('assets/Farewell_diffs.json'), 'r') as f:
    #     diffs_farewell = json.load(f)
    # print(len(diffs_ginseng) * 0.023219954648526078)
    #
    # # plt.plot([i * 0.023 for i in range(len(diffs_ginseng))], diffs_ginseng)
    # # plt.plot([i * 0.023 for i in range(len(diffs_farewell))], diffs_farewell)
    # plt.figure(figsize=(12.8, 3.2))
    # plt.ylim((0, 1))
    # plt.plot([i * 0.023219954648526078 for i in range(3200)], diffs_ginseng[:3200], lw=3, label='Ginseng')
    # plt.plot([i * 0.023219954648526078 for i in range(3200)], diffs_farewell[:3200], lw=3, label='Farewell')
    # plt.ylabel('Difficulty', size=18)
    # plt.text(1600 * 0.023219954648526078, 0.06, 'Real-Time', ha="center", va="center", size=18)
    # plt.xticks(size=15)
    # plt.yticks(size=15)
    # plt.legend(fontsize=16)
    # plt.show()
