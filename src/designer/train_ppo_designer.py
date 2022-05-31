"""
  @Time : 2021/10/9 19:02 
  @Author : Ziqi Wang
  @File : train.py
"""

import importlib
import os
from stable_baselines3 import PPO
from src.environment.env import make_vec_generation_env
from src.environment.checkpoint import MyCheckpointCallback
from src.utils import auto_dire
from stable_baselines3.common.logger import configure
from root import PRJROOT


def set_parser(parser):
    parser.add_argument(
        '--n_envs', type=int, default=5,
        help='Number of parallel environments.'
    )
    parser.add_argument(
        '--max_seg_num', type=int, default=50,
        help='Maximum nubmer of segments to generate in the generation enviroment.'
    )
    parser.add_argument(
        '--total_steps', type=int, default=int(1e6),
        help='Total time steps (frames) for training PPO designer.'
    )
    parser.add_argument(
        '--n_steps', type=int, default=500,
        help='n_steps parameter for SB3.PPO'
    )
    parser.add_argument(
        '--n_epochs', type=int, default=50,
        help='n_epochs parameter for SB3.PPO'
    )
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='batch_size parameter for SB3.PPO'
    )
    # parser.add_argument(
    #     '--use_gsde', action='store_true',
    #     help='If add this argument, PPO Algorithm won\'t use gSDE'
    # )
    parser.add_argument(
        '--gae_lambda', type=float, default=0.85,
        help='gamma parameter for SB3.PPO'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.7,
        help='gamma parameter for SB3.PPO'
    )
    # parser.add_argument(
    #     '--gsde_sample_freq', type=int, default=20,
    #     help='sde_sample_freq parameter for SB3.PPO'
    # )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for training the PPO agent.'
    )
    parser.add_argument(
        '--rfunc_name', type=str, default='default',
        help='Name of the file where the reward function located. '
             'The file must be put in the \'src.reward_functions\' package.'
    )
    parser.add_argument(
        '--res_path', type=str, default='',
        help='Path relateed to \'/exp_data\'to save the training log. '
             'If not specified, a new folder named exp{id} will be created.'
    )
    parser.add_argument('--log_itv', type=int, default=1)
    parser.add_argument(
        '--sb_loggers', default=['stdout', 'log'], nargs='*',
        help='SB3 loggers to use, should be a list of items within:'
             '{"stdout", "csv", "log", "tensorboard", "json"}'
    )
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='check points to save deisigner, specified by the number of time steps.'
    )
    # print(parser)

def train_designer(cfgs):
    if not cfgs.res_path:
        cfgs.res_path = auto_dire(PRJROOT + 'exp_data')
    else:
        cfgs.res_path = PRJROOT + 'exp_data/' + cfgs.res_path
        try:
            os.makedirs(cfgs.res_path)
        except FileExistsError:
            pass
    # print(cfgs.check_points)
    rfunc = (
        importlib.import_module(f'src.rewfuncs.{cfgs.rfunc_name}')
        .__getattribute__('rfunc')
    )
    with open(cfgs.res_path + '/run_config.txt', 'w') as f:
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(cfgs).items()
            if key not in {'rfunc_name', 'res_path', 'entry', 'check_points', 'sb_loggers'}
        ]

        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))

    env = make_vec_generation_env(
        cfgs.n_envs, rfunc, cfgs.res_path, cfgs.max_seg_num,
        log_itv=50, device=cfgs.device, log_targets='file'
    )
    designer = PPO(
        "MlpPolicy", env, verbose=1, tensorboard_log=cfgs.res_path, gamma=cfgs.gamma,
        n_steps=cfgs.n_steps, n_epochs=cfgs.n_epochs, batch_size=cfgs.batch_size, gae_lambda=cfgs.gae_lambda,
        use_sde=False, device=cfgs.device, policy_kwargs={'net_arch': [128, 128, 128]}
    )
    designer.set_logger(configure(cfgs.res_path, cfgs.sb_loggers))

    kwargs_for_learn = {'total_timesteps': cfgs.total_steps, 'log_interval': cfgs.log_itv, 'tb_log_name': 'tb_log'}
    if cfgs.check_points:
        kwargs_for_learn['callback'] = MyCheckpointCallback(cfgs.res_path, cfgs.check_points)
    designer.learn(**kwargs_for_learn)
    designer.save(cfgs.res_path + "/designer")
