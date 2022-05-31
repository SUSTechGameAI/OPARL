"""
  @Time : 2022/4/19 15:18 
  @Author : Ziqi Wang
  @File : ol_gen_game.py 
"""

import json
import time

import jpype
import numpy as np
import pygame as pg
import multiprocessing as mp
from root import PRJROOT
from config import JVMPath
from smb import MarioLevel
# from src.designer.designer import Designer
# from src.gan.gan_use import get_generator
# from src.ol_gen.ol_generator import OnlineGenerator
from src.online_generation.ol_gensys import *
from src.utils import get_path


def _ol_gen_worker(remote, parent_remote):
    parent_remote.close()
    # designer = Designer(f'{d_path}/actor.pth')
    # with open(get_path(f'{d_path}/kwargs.json'), 'r') as f:
    #     n = json.load(f)['hist_len']
    # generator = get_generator() if g_path == '' else get_generator(get_path(g_path))
    # ol_generator = OnlineGenerator(designer, generator, n, g_device)
    print('start')
    ol_gensys = OnlineGenerationSystem(
        AggregatedGenerator('exp_data/sac/fp/actor.pth'),
        f'assets/blended_diffs.json', True
    )
    with open(get_path('assets/start_latvec.json'), 'r') as f:
        z = json.load(f)
    seg = ol_gensys.reset(np.array(z).squeeze())
    # print(str(seg))
    remote.send(str(seg))
    # remote.send(ol_gensys.step())
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(str(ol_gensys.step()))
            elif cmd == "recv":
                ol_gensys.recieve_play_data(*data)
                remote.send(None)
            elif cmd == "close":
                remote.close()
                break
        except EOFError:
            break
    pass


class MarioOnlineGenGame:
    def __init__(self, g_device='cuda:0'):
        if not jpype.isJVMStarted():
            jpype.startJVM(
                jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
                f"-Djava.class.path={PRJROOT}Mario-AI-Framework.jar", '-Xmx4g'
            )
        self.g_device = g_device
        self.ol_gen_remote, self.process = None, None

    def play(self, agent=None, max_length=20):
        self.__init_ol_gen_remote()
        seg_str = self.ol_gen_remote.recv()
        print(seg_str)
        # seg_str = str(MarioLevel.from_txt('levels/original/mario-1-1.txt'))
        if agent is None:
            game = jpype.JClass("MarioOnlineGenGame")(jpype.JString(seg_str))
        else:
            game =  jpype.JClass("MarioOnlineGenGame")(
                jpype.JString(seg_str), jpype.JClass(str(agent))(), jpype.JBoolean(True)
            )
        clk = pg.time.Clock()
        finish = False
        n_seg = 1
        # self.ol_gen_remote.send(('step', None))
        # frame_counter = 0
        generating = False
        start = time.time()
        while not finish:
            finish = bool(game.gameStep())
            if n_seg < max_length and not generating and int(game.getTileDistantToExit()) < MarioLevel.default_seg_width:
                # send play duration and starting to generate the next segment in another thread
                self.ol_gen_remote.send(('recv', (time.time() - start, n_seg < 2)))
                self.ol_gen_remote.recv()
                self.ol_gen_remote.send(('step', None))
                # frame_counter = 0
                generating = True
                start = time.time()
            if n_seg < max_length and int(game.getTileDistantToExit()) < 10:
                # append the generated segment
                seg_str = self.ol_gen_remote.recv()
                game.appendSegment(jpype.JString(seg_str))
                n_seg += 1
                generating = False
            # frame_counter += 1
            # if int(game.getMarioTileX) > MarioLevel.default_seg_width and int(game.getMarioTileX):
            #     pass
            clk.tick(30)
            pass
        self.close()
        pass
    #
    # def test(self, agent, featr):
    #     pass

    def __init_ol_gen_remote(self):
        forkserver_available = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)
        self.ol_gen_remote, work_remote = ctx.Pipe()
        args = (work_remote, self.ol_gen_remote)
        self.process = ctx.Process(target=_ol_gen_worker, args=args, daemon=True)
        self.process.start()

    def close(self):
        self.ol_gen_remote.send(('close', None))
        self.process.join()

