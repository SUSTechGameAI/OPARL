"""
  @Time : 2022/8/24 14:22 
  @Author : Ziqi Wang
  @File : map_features.py 
"""
import json

import librosa
import numpy as np

from src.utils import get_path


def extract_music_feature(fpath):
    y, sr = librosa.load(get_path(fpath))
    librosa.feature.rms(y=y)
    S, _ = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)[0]
    return rms

def mapping_feature(music_features):
    log10_rms = np.log10(music_features)
    smooth_log_rms = [
        sum(log10_rms[max(0, c-50): min(len(log10_rms)-1, c+50)]) / 100
        for c in range(len(log10_rms))
    ]
    diffs = []
    for x in smooth_log_rms:
        clipped = max(-2.5, min(x, 0))
        diff = (clipped + 2.5) / 2.5
        diffs.append(diff)
    return diffs


if __name__ == '__main__':
    music_path = 'assets/blended.wav'
    lvl_features = mapping_feature(extract_music_feature(music_path))
    clean_name = ''.join(music_path.split('.')[:-1])
    with open(get_path(f'{clean_name}_diffs.json'), 'w') as f:
        json.dump(lvl_features, f)
        pass
    pass
