"""
  @Time : 2022/2/18 10:29 
  @Author : Ziqi Wang
  @File : test_librosa.py 
"""
import json

import librosa
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_path

if __name__ == '__main__':
    # # y, sr = librosa.load(librosa.ex('trumpet'))
    # y1, sr1 = librosa.load(
    #     get_path('misc/music_featr_extract/Ginseng-EnV.flac'),
    #     # offset=2, duration=10
    # )
    # print(len(y1))
    # librosa.feature.rms(y=y1)
    # S1, phase1 = librosa.magphase(librosa.stft(y1))
    # rms1 = librosa.feature.rms(S=S1)[0]
    # # fig, ax = plt.subplots(nrows=2)
    # times1 = librosa.times_like(rms1)
    # log_rms1 = np.log10(np.array(rms1))
    #
    # smooth_log_rms1 = [
    #     sum(log_rms1[max(0, c-50): min(len(log_rms1)-1, c+50)]) / 100
    #     for c in range(len(log_rms1))
    # ]
    # with open(get_path('misc/music_featr_extract/time_unit.json'), 'w') as f:
    #     json.dump(times1[1], f)
    # print(times1[1], 1 / times1[1], 2 / times1[2])
    # print(times1)
    #
    #
    # plt.figure(figsize=(6.4, 2.4), dpi=200)
    # plt.plot(times1, smooth_log_rms1)

    y2, sr2 = librosa.load(
        get_path('assets/blended.wav'),
        # offset=2, duration=60
    )
    print(len(y2))
    librosa.feature.rms(y=y2)
    S2, phase2 = librosa.magphase(librosa.stft(y2))
    rms2 = librosa.feature.rms(S=S2)[0]
    # fig, ax = plt.subplots(nrows=2)
    times2 = librosa.times_like(rms2)
    log_rms2 = np.log10(np.array(rms2))
    smooth_log_rms2 = [
        sum(log_rms2[max(0, c-50): min(len(log_rms2)-1, c+50)]) / 100
        for c in range(len(log_rms2))
    ]
    print(times2)
    # print(len(y) // 2048, len(rms))

    plt.figure(figsize=(12.8, 3.2))
    plt.plot(times2, smooth_log_rms2)
    plt.grid()
    plt.show()

    # with open(get_path('misc/music_featr_extract/Farewell_energy_curve.json'), 'w') as f:
    #     json.dump(smooth_log_rms1, f)
    with open(get_path('misc/music_featr_extract/blended_energy_curve.json'), 'w') as f:
        json.dump(smooth_log_rms2, f)
