"""
  @Time : 2022/2/24 18:46 
  @Author : Ziqi Wang
  @File : tmp.py 
"""
import os

from src.utils import get_path

if __name__ == '__main__':
    folders = ['f', 'c', 'p', 'fc', 'fp', 'cp', 'fcp']
    for fd in folders:
        path = get_path(f'exp_data/sac/{fd}/Ginseng_Baumgarten')
        for i in range(30):
            os.rename(path + f'/lvl{i}.txt', path + f'/lvl{i+70}.txt')
            pass

