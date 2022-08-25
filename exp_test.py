
import argparse

from smb import MarioJavaAgents
from src.analyze.olgen_analyze import test_and_save
from src.designer import train_designer, train_ppo_designer
from src.gan import train_gan
from src.repairer import cnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--designer_path', type=str, default='exp_data/sac/fcp')
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--disable_ctrl', type=bool, default=False)
    parser.add_argument('--music_fname', type=str, default='Ginseng')
    parser.add_argument('--agent_name', type=str, default='Baumgarten')

    args = parser.parse_args()
    test_and_save(
        args.designer_path, args.music_name, MarioJavaAgents.__getitem__(args.agent_name),
        n_trials=args.n_trials, disable_ctrl=args.disable_ctrl
    )


