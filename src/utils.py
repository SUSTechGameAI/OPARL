"""
  @Time : 2021/9/9 10:08 
  @Author : Ziqi Wang
  @File : utils.py 
"""

import os
import pygame.draw
import pygame as pg
from math import ceil
from root import PRJROOT


def image_row_cat(imgs, space=16, path=None):
    n = len(imgs)
    concated = pg.Surface(
        (max(img.get_width() for img in imgs),
        sum(img.get_height() for img in imgs) + space * (n - 1)),
    )
    concated.set_colorkey((255, 255, 255))
    concated.fill((255, 255, 255, 0))
    y = 0
    for img in imgs:
        concated.blit(img, (0, y))
        y += (img.get_height() + space)
    if path:
        pg.image.save(concated, path)
    return concated

def auto_dire(path=None, name='trial'):
    dire_id = 0
    prefix = PRJROOT if path is None else get_path(path) + '/'
    tar = f'{prefix}'
    while os.path.exists(tar):
        tar = f'{prefix}{name}{dire_id}'
        dire_id += 1
    os.makedirs(tar)
    return tar

def tab_per_line(string, n=1):
    lines = string.split('\n')
    tabed = []
    for line in lines:
        tabed.append('\t' * n + line)
    return '\n'.join(tabed)

def draw_level_sheet(levels, highlights=None, n_cols=8, pad=8, save_path=None):
    img_list = [lvl.to_img(None) for lvl in levels]
    n_rows = len(img_list) // n_cols if len(img_list) % n_cols == 0 else ceil(len(img_list) / n_cols)
    imgw, imgh = img_list[0].get_size()
    if type(pad) == int:
        h_pad, v_pad = pad, pad
    else:
        h_pad, v_pad = pad
    cw, ch = n_cols * imgw + n_cols * (h_pad + 1), n_rows * imgh + n_rows * (v_pad + 1)
    canvas = pygame.Surface((cw, ch))
    canvas.fill((255, 255, 255))
    for n in range(len(img_list)):
        img = img_list[n]
        i, j = n % n_cols, n // n_cols
        canvas.blit(img, (i * (imgw + h_pad) + h_pad, j * (imgh + v_pad) + v_pad))
    if highlights:
        for n in highlights:
            i, j = n % n_cols, n // n_cols
            line_width = max(4, min(v_pad, h_pad))
            tar_rect = (
                i * (imgw + v_pad) + line_width // 2, j * (imgh + h_pad) + line_width // 2,
                imgw + line_width, imgh + line_width
            )
            pygame.draw.rect(canvas, 'red', tar_rect, width=pad)
    if save_path:
        abs_path = PRJROOT + save_path
        pygame.image.save(canvas, abs_path)
    return canvas

def batched_iter(arr, n):
    for s in range(0, len(arr), n):
        e = min(s + n, len(arr))
        n = e - s
        yield arr[s:e], n

def get_path(path):
    """ if is absolute path or working path(./, .\\), return {path}, else return {PRJROOT + path} """
    if os.path.isabs(path) or path in {'./', '.\\'}:
        return path
    else:
        return PRJROOT + path


class RingQueue:
    def __init__(self, capacity):
        self.main = []
        self.p = 0
        self.capacity = capacity

    def push(self, item):
        if len(self.main) < self.capacity:
            self.main.append(item)
        else:
            self.main[self.p] = item
        self.p = (self.p + 1) % self.capacity

    def front(self):
        if len(self.main) < self.capacity:
            return self.main[-1]
        else:
            return self.main[(self.p - 1) % self.capacity]

    def rear(self):
        if len(self.main) < self.capacity:
            return self.main[0]
        else:
            return self.main[self.p]

    def to_list(self):
        return self.main[self.p:] + self.main[:self.p]

    def clear(self):
        self.main.clear()
        self.p = 0

    def __len__(self):
        return len(self.main)


def a_clip(v, g, r=1.0):
    return min(r, 1 - abs(v - g) / g)


if __name__ == '__main__':
    pass

