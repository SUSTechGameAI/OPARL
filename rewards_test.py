
import argparse

from src.analyze.genlvl_statics import test_reward
from src.environment.reward_func import RewardFunction
from src.environment.reward_terms import FunTest, Playability, Controllability


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--designer_path', type=str, default='exp_data/sac/fcp')
    parser.add_argument('--n_segs', type=int, default=50)
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--n_parallel', type=int, default=5)
    parser.add_argument('--no_controllability', action='store_true')

    args = parser.parse_args()
    rterms = [FunTest(), Playability()]
    if not args.no_controllability:
        rterms.append(Controllability())
    test_reward(
        args.designer_path, RewardFunction(*rterms), l=args.n_segs,
        n=args.n_trials, n_parallel=args.n_parallel
    )

