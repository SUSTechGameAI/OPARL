"""
  @Time : 2022/1/27 15:20 
  @Author : Ziqi Wang
  @File : gan_use.py 
"""
import json
import numpy as np
import torch
from smb import MarioLevel
from src.gan.gan_models import Generator
from src.gan.gan_config import nz
from src.utils import get_path


def sample_latvec(n=1, device='cpu'):
    return torch.rand(n, nz, 1, 1, device=device) * 2 - 1


def process_levels(raw_tensor_lvls, to_lvl_obj=False):
    H, W = MarioLevel.height, MarioLevel.default_seg_width
    res = []
    for single in raw_tensor_lvls:
        lvl = single[:, :H, :W].detach().cpu().numpy()
        if to_lvl_obj:
            lvl = MarioLevel.from_one_hot_arr(lvl)
        res.append(lvl)
    return res


def get_generator(path='models/generator.pth', device='cpu'):
    safe_path = get_path(path)
    generator = torch.load(safe_path, map_location=device)
    generator.requires_grad_(False)
    generator.eval()
    return generator
    pass


if __name__ == '__main__':
    # z = sample_latvec(20)
    # generator = get_generator()
    # level_onehots = generator(z)
    # levels = process_levels(level_onehots, True)
    # latvecs = z.numpy()
    # z.squeeze()
    # for i, (latvec, level) in enumerate(zip(z.tolist(), levels)):
    #     with open(get_path(f'exp_data/gan_samples/latvec_{i}.json'), 'w')as f:
    #         json.dump(latvec, f)
    #     level.to_img(f'exp_data/gan_samples/level_{i}.png')

    with open(get_path('assets/start_latvec.json'), 'r')as f:
        data = json.load(f)
    print(np.array(data).squeeze())

