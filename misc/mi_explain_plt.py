"""
  @Time : 2022/2/11 14:42 
  @Author : Ziqi Wang
  @File : mi_explain_plt.py
"""
import json

import librosa.display
import matplotlib.pyplot as plt
import pygame

from smb import MarioLevel
from src.level_diffs import naive_difficulty_measure
from src.utils import get_path

if __name__ == '__main__':
    # y1, sr1 = librosa.load(
    #     get_path('misc/music_featr_extract/Ginseng-EnV.flac'),
    #     duration=110
    # )
    # plt.figure(figsize=(16.8, 2.4))
    # librosa.display.waveshow(y1, color='forestgreen')
    # plt.grid()
    # plt.xticks([], [])
    # plt.yticks([], [])
    # plt.show()

    trial_id = 12
    agent_name = 'Hartmann'
    time_unit = 0.023219954648526078
    plt.figure(figsize=(14.4, 2.8))
    t_start = 49
    t_end = 57
    i_start = int(t_start // time_unit)
    i_end = int((t_end + 1) // time_unit)
    with open(get_path('misc/music_featr_extract/Ginseng_diffs.json'), 'r') as f:
        tar_diffs = json.load(f)
    plt.plot(
        [i * time_unit for i in range(i_start, i_end)], tar_diffs[i_start:i_end],
        color='black', lw=3, aa=True, label='ideal difficulty'
    )

    with open(get_path(f'exp_data/sac/fcp/Ginseng_{agent_name}/simulation_res.json'), 'r') as f:
        data = json.load(f)[trial_id]
    level = MarioLevel.from_txt(f'exp_data/sac/fcp/Ginseng_{agent_name}/lvl0.txt')
    ends = data['ends']

    s = 0
    while ends[s] < t_start:
        s += 1
    e = s
    while ends[e] < t_end:
        e += 1
        if ends[e] < t_end:
            diff = naive_difficulty_measure(seg=level[:, (e-1)*28: e*28])
            if e == s+1:
                plt.plot([ends[e-1], ends[e]], [diff, diff], color='red', lw=2, label='real difficulty')
            else:
                plt.plot([ends[e-1], ends[e]], [diff, diff], color='red', lw=2)

    for i in range(s, e):
        plt.plot([ends[i], ends[i]], [0, 1], ls='--', color='grey')
    print(s, e)
    plt.ylim((0, 1))
    # plt.ylabel('Difficulty', size=18)
    # plt.ylabel('Difficulty', size=18)
    plt.xticks(size=15)
    plt.yticks(size=15)
    # plt.legend(loc='lower right', fontsize=16)
    plt.show()
    # #
    # # img = level[:, s*28: (e-1)*28].to_img(None)
    # # canvas = pygame.Surface((img.get_width() + (e-s-2) * 5, img.get_height()))
    # # canvas.fill('white')
    # # for i in range(e-s-1):
    # #     canvas.blit(img.subsurface((i * 28 * 16, 0, 28*16, 14*16)), (i * (28 * 16 + 5), 0))
    # # pygame.image.save(canvas, get_path(f'misc/mi_explain_plt/{agent_name}_{trial_id}__{s}-{e-1}.png'))
    # #
