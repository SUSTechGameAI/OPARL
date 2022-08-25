from smb import *
from src.online_generation.ol_gen_game import MarioOnlineGenGame


if __name__ == '__main__':
    proxy = MarioProxy()
    level = MarioLevel.from_txt('levels/flat.txt')
    # proxy.play_game(level)
    # level = MarioLevel.from_txt('exp_data/main/fp/Ginseng_Baumgarten/lvl17.txt')
    # proxy.simulate_game(level, render=True, fps=30)
    game = MarioOnlineGenGame('exp_data/main/both')
    game.play(lives=10)


