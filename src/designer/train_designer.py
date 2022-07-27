"""
  @Time : 2021/11/29 20:03 
  @Author : Ziqi Wang
  @File : train_sac_designer.py 
"""

import os
import importlib
from root import PRJROOT
from src.rl_algo.sac import SAC_Model, SAC_Trainer
from src.rl_algo.ac_model import SquashedGaussianMLPActor, MLPQFunction
from src.environment.env import make_vec_generation_env
from src.rl_algo.replay_memory import ReplayMem
from src.utils import auto_dire
from src.gan.gan_config import nz
from config import archive_len, ctrl_sig_dup


def set_parser(parser):
    parser.add_argument(
        '--total_steps', type=int, default=int(1e5),
        help='Total time steps (frames) for training SAC designer.'
    )
    parser.add_argument(
        '--n_envs', type=int, default=5,
        help='Number of parallel environments.'
    )
    parser.add_argument(
        '--max_seg_num', type=int, default=50,
        help='Maximum nubmer of segments to generate in one episode'
    )
    parser.add_argument('--gamma', type=float, default=0.7, help='Gamma parameter of RL')
    parser.add_argument('--tar_entropy', type=float, default=-nz, help='Target entropy parameter of SAC')
    parser.add_argument('--tau', type=float, default=0.005, help='Tau parameter of SAC')
    parser.add_argument('--update_itv', type=int, default=100, help='Interval (in unit of time steps) of updating SAC model')
    parser.add_argument('--update_repeats', type=int, default=10, help='Repeatly update the SAC model for how many times in one update iteration')
    parser.add_argument('--batch_size', type=int, default=384, help='Batch size of training SAC model')
    parser.add_argument('--mem_size', type=int, default=int(1e6), help='Capacity of replay memory for SAC training')
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        help='Device for training the SAC agent.'
    )
    parser.add_argument(
        '--rfunc_name', type=str, default='default',
        help='Name of the file where the reward function located. '
             'The file must be put in the \'src.rewfuncs\' package.'
    )
    parser.add_argument(
        '--res_path', type=str, default='',
        help='Path relateed to \'/exp_data\'to save the training log. '
             'If not specified, a new folder named exp{id} will be created.'
    )
    parser.add_argument('--log_itv', type=int, default=100, help='Interval (in unit of episode) of logging')
    parser.add_argument(
        '--check_points', type=int, nargs='+',
        help='check points to save deisigner, specified by the number of time steps.'
    )

def train_designer(cfgs):
    # torch.set_default_dtype(torch.float)
    if not cfgs.res_path:
        cfgs.res_path = auto_dire(PRJROOT + 'exp_data')
    else:
        cfgs.res_path = PRJROOT + 'exp_data/' + cfgs.res_path
        os.makedirs(cfgs.res_path, exist_ok=True)

    rfunc = (
        importlib.import_module(f'src.rewfuncs.{cfgs.rfunc_name}')
        .__getattribute__('rfunc')
    )
    with open(cfgs.res_path + '/run_config.txt', 'w') as f:
        f.write('---------SAC---------\n')
        args_strlines = [
            f'{key}={val}\n' for key, val in vars(cfgs).items()
            if key not in {'rfunc_name', 'res_path', 'entry', 'check_points'}
        ]
        f.writelines(args_strlines)
        f.write('-' * 50 + '\n')
        f.write(str(rfunc))

    env = make_vec_generation_env(
        cfgs.n_envs, rfunc, cfgs.res_path, cfgs.max_seg_num, log_itv=cfgs.log_itv,
        device=cfgs.device, log_targets=['file', 'std']
    )

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    designer = SAC_Model(
        lambda: SquashedGaussianMLPActor(obs_dim, act_dim, [256, 256, 256]),
        lambda: MLPQFunction(obs_dim, act_dim, [256, 256, 256]),
        gamma=cfgs.gamma, tar_entropy=cfgs.tar_entropy, tau=cfgs.tau, device=cfgs.device
    )
    d_trainer = SAC_Trainer(
        env, cfgs.total_steps, update_itv=cfgs.update_itv, update_repeats=cfgs.update_repeats,
        update_start=cfgs.batch_size, batch_size=cfgs.batch_size, rep_mem=ReplayMem(cfgs.mem_size),
        save_path=cfgs.res_path, check_points=cfgs.check_points, no_termination=True
    )
    d_trainer.train(designer)

