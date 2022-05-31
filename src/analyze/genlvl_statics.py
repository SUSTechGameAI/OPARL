"""
  @Time : 2022/2/8 11:20 
  @Author : Ziqi Wang
  @File : genlvl_statics.py 
"""
import json
import os

from smb import MarioLevel
# from src.designer.use_designer import Designer
from src.designer.use_designer import Designer
from src.environment.env import make_vec_generation_env
from stable_baselines3.ppo import PPO

from src.environment.reward_func import RewardFunction
from src.environment.reward_terms import Fun, FunTest, Controllability, Playability
from src.utils import get_path


def generate_levels(src_path, additional_folder='', save_img=False, n=1, l=10, n_parallel=1):
    # designer = Designer(src_path + '/actor.pth')
    designer = PPO.load(get_path(src_path + '/designer.zip'))
    env = make_vec_generation_env(n_parallel, max_seg_num=l)
    levels = []
    obs = env.reset()
    while len(levels) < n:
        actions, _ = designer.predict(obs)
        next_obs, _, dones, infos = env.step(actions)
        del obs
        obs = next_obs
        for done, info in zip(dones, infos):
            if done:
                level = MarioLevel(info['LevelStr'])
                levels.append(level)
    add_path = '' if additional_folder == '' else '/' + additional_folder
    os.makedirs(get_path(f'{src_path}{add_path}'), exist_ok=True)
    for i in range(n):
        level = levels[i]
        level.save(f'{src_path}{add_path}/sample{i}.txt')
        if save_img:
            level.to_img(f'{src_path}{add_path}/sample{i}.png')
    pass


def test_reward(src_path, rfunc, n=30, l=50, n_parallel=5):
    # designer = PPO.load(get_path(src_path + '/designer.zip'))
    designer = Designer(get_path(src_path + '/actor.pth'))
    env = make_vec_generation_env(n_parallel, rfunc, max_seg_num=l)
    obs = env.reset()
    res = []
    if os.path.exists(get_path(src_path + '/test_rewards.json')):
        with open(get_path(src_path + '/test_rewards.json'), 'r') as f:
            res = json.load(f)
    while len(res) < n:
        actions = designer.act(obs)
        # actions, _ = designer.predict(obs)
        next_obs, _, dones, infos = env.step(actions)
        del obs
        obs = next_obs
        for done, info in zip(dones, infos):
            if done:
                res.append({
                    key: info[key] for key in info.keys() if key in
                    {'Fun', 'FunTest', 'Controllability', 'Playability'}
                })
                print(res[-1])
    os.makedirs(get_path(f'{src_path}'), exist_ok=True)
    with open(get_path(src_path + '/test_rewards.json'), 'w') as f:
        json.dump(res, f)
    pass


if __name__ == '__main__':
    test_reward('exp_data/sac/f', RewardFunction(FunTest(), Playability()), n=100)
    test_reward('exp_data/sac/p', RewardFunction(FunTest(), Playability()), n=100)
    test_reward('exp_data/sac/fp', RewardFunction(FunTest(), Playability()), n=100)
    test_reward('exp_data/sac/c', RewardFunction(FunTest(), Playability(), Controllability()), n=100)
    test_reward('exp_data/sac/fc', RewardFunction(FunTest(), Playability(), Controllability()), n=100)
    test_reward('exp_data/sac/cp', RewardFunction(FunTest(), Playability(), Controllability()), n=100)
    test_reward('exp_data/sac/fcp', RewardFunction(FunTest(), Playability(), Controllability()), n=100)
    # generate_levels('exp_data/f', 'samples', True, n=5, n_parallel=5)
