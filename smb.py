"""
  @Time : 2021/9/8 17:05 
  @Author : Ziqi Wang
  @File : smb.py
"""
import glob
from math import ceil

import jpype
import numpy as np
import pygame as pg
from enum import Enum
from typing import Union, Dict
from itertools import product
from jpype import JString
from root import PRJROOT
from config import JVMPath
from src.utils import get_path

# ! ! This file must be placed at the project root directory ! !


class MarioLevel:
    tex_size = 16
    height = 14
    default_seg_width = 28
    mapping = {
        'i-c': ('X', 'S', '-', '?', 'Q', 'E', '<', '>', '[', ']', 'o'),
        'c-i': {'X': 0, 'S': 1, '-': 2, '?': 3, 'Q': 4, 'E': 5, '<': 6,
        '>': 7, '[': 8, ']': 9, 'o': 10}
    }
    empty_tiles = {'-', 'E', 'o'}
    num_tile_types = len(mapping['i-c'])
    pipe_charset = {'<', '>', '[', ']'}
    pipe_intset = {6, 7, 8, 9}
    textures = [
        pg.image.load(PRJROOT + f'assets/tile-{i}.png')
        for i in range(num_tile_types)
    ]

    def __init__(self, content):
        if isinstance(content, np.ndarray):
            self.content = content
        else:
            tmp = [list(line) for line in content.split('\n')]
            while not tmp[-1]:
                tmp.pop()
            self.content = np.array(tmp)
        self.h, self.w = self.content.shape
        self.__tile_pttr_cnts = {}
        self.attr_dict = {}

    def to_num_arr(self):
        res = np.zeros((self.h, self.w), int)
        for i, j in product(range(self.h), range(self.w)):
            char = self.content[i, j]
            res[i, j] = MarioLevel.mapping['c-i'][char]
        return res

    def to_img(self, save_path='render.png') -> pg.Surface:
        tex_size = MarioLevel.tex_size
        num_lvl = self.to_num_arr()
        img = pg.Surface((self.w * tex_size, self.h * tex_size))
        img.fill((150, 150, 255))

        for i, j in product(range(self.h), range(self.w)):
            tile_id = num_lvl[i, j]
            if tile_id == 2:
                continue
            img.blit(
                MarioLevel.textures[tile_id],
                (j * tex_size, i * tex_size, tex_size, tex_size)
            )
        if save_path:
            safe_path = get_path(save_path)
            pg.image.save(img, safe_path)
        return img

    def save(self, fpath):
        safe_path = get_path(fpath)
        with open(safe_path, 'w') as f:
            f.write(str(self))

    def tile_pattern_counts(self, w=2):
        if not w in self.__tile_pttr_cnts.keys():
            counts = {}
            for i, j in product(range(self.h - w + 1), range(self.w - w + 1)):
                key = ''.join(self.content[i+x][j+y] for x, y in product(range(w), range(w)))
                count = counts.setdefault(key, 0)
                counts[key] = count + 1
            self.__tile_pttr_cnts[w] = counts
        return self.__tile_pttr_cnts[w]

    def tile_pattern_distribution(self, w=2):
        counts = self.tile_pattern_counts(w)
        C = (self.h - w + 1) * (self.w - w + 1)
        return {key: val / C for key, val in counts.items()}

    def __getattr__(self, item):
        if item == 'shape':
            return self.content.shape
        elif item == 'h':
            return self.content.shape[0]
        elif item == 'w':
            return self.content.shape[1]
        elif item not in self.attr_dict.keys():
            if item == 'n_grounds':
                ground_map1 = [0 if item in MarioLevel.empty_tiles else 1 for item in self.content[-1]]
                # print(self.content[-1], empty_map1)
                ground_map2 = [0 if item in MarioLevel.empty_tiles else 1 for item in self.content[-2]]
                res = len([i for i in range(self.w) if ground_map1[i] + ground_map2[i] > 0])
                self.attr_dict['n_grounds'] = res
            elif item == 'n_enemies':
                self.attr_dict['n_enemies'] = str(self).count('E')
            elif item == 'n_coins':
                self.attr_dict['n_coins'] = str(self).count('o')
            elif item == 'n_questions':
                self.attr_dict['n_questions'] = str(self).count('Q')
        return self.attr_dict[item]

    def __str__(self):
        lines = [''.join(line) + '\n' for line in self.content]
        return ''.join(lines)

    def __add__(self, other):
        concated = np.concatenate([self.content, other.content], axis=1)
        return MarioLevel(concated)

    def __getitem__(self, item):
        return MarioLevel(self.content[item])

    @staticmethod
    def from_num_arr(num_arr):
        h, w = num_arr.shape
        res = np.empty((h, w), str)
        for i, j in product(range(h), range(w)):
            tile_id = num_arr[i, j]
            res[i, j] = MarioLevel.mapping['i-c'][int(tile_id)]
        return MarioLevel(res)

    @staticmethod
    def from_txt(fpath):
        safe_path = get_path(fpath)
        with open(safe_path, 'r') as f:
            return MarioLevel(f.read())

    @staticmethod
    def from_one_hot_arr(one_hot_arr: np.ndarray):
        num_lvl = one_hot_arr.argmax(axis=0)
        return MarioLevel.from_num_arr(num_lvl)


class MarioJavaAgents(Enum):
    Baumgarten = 'agents.robinBaumgarten'
    Sloane = 'agents.andySloane'
    Hartmann = 'agents.glennHartmann'
    Michal = 'agents.michal'
    Polikarpov = 'agents.sergeyPolikarpov'
    Schumann = 'agents.spencerSchumann'
    Ellingsen = 'agents.trondEllingsen'

    def __str__(self):
        return self.value + '.Agent'


class MarioProxy:
    def __init__(self):
        if not jpype.isJVMStarted():
            jpype.startJVM(
                jpype.getDefaultJVMPath() if JVMPath is None else JVMPath,
                f"-Djava.class.path={PRJROOT}Mario-AI-Framework.jar", '-Xmx1g'
            )
            """
                -Xmx{size} set the heap size.
            """
        self.__proxy = jpype.JClass("MarioProxy")()

    @staticmethod
    def __extract_res(jresult):
        # Refers to Mario-AI-Framework.engine.core.MarioResult, add more entries if need be.
        return {
            'status': str(jresult.getGameStatus().toString()),
            'completing-ratio': float(jresult.getCompletionPercentage()),
            '#kills': int(jresult.getKillsTotal()),
            '#kills-by-fire': int(jresult.getKillsByFire()),
            '#kills-by-stomp': int(jresult.getKillsByStomp()),
            '#kills-by-shell': int(jresult.getKillsByShell()),
            'trace': [
                [float(item.getMarioX()), float(item.getMarioY())]
                for item in jresult.getAgentEvents()
            ],
            'JAgentEvents': jresult.getAgentEvents()
        }

    def play_game(self, level: Union[str, MarioLevel]):
        if type(level) == str:
            jfilepath = JString(level)
            jresult = self.__proxy.playGameFromTxt(jfilepath)
        else:
            jresult = self.__proxy.playGame(JString(str(level)))
        return MarioProxy.__extract_res(jresult)

    @staticmethod
    def save_rep(path, JAgentEvents):
        # tmp = jpype.JClass("agents.replay.ReplayAgent")()
        print(type(JAgentEvents))
        jpype.JClass("agents.replay.ReplayUtils").saveReplay(JString(get_path(path)), JAgentEvents)

    def replay(self, level, filepath):
        # replay_agent = jpype.JClass("agents.replay.ReplayUtils").repAgentFromFile(get_path('levels/train/mario-1-1.rep'))
        if type(level) == MarioLevel:
            self.__proxy.replay(JString(str(level)), JString(filepath))
        else:
            self.__proxy.replay(JString(level), JString(filepath))

    def simulate_game(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Baumgarten,
        render: bool=False, fps: int=0
    ) -> Dict:
        """
        Run simulation with an agent for a given level
        :param level: if type is str, must be path of a valid level file.
        :param agent: type of the agent.
        :param render: render or not.
        :param fps: frame per seconds.
        :return: dictionary of the results.
        """
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_txt(level)

        real_time_limit_ms = level.w * 80 + 200 if (not render and fps == 0) else level.w * 300
        jresult = self.__proxy.simulateGame(JString(str(level)), jagent, render, real_time_limit_ms, fps)
        return MarioProxy.__extract_res(jresult)

    def simulate_with_reset(self,
        level: Union[str, MarioLevel],
        agent: MarioJavaAgents=MarioJavaAgents.Baumgarten,
        k: float=5., b: int=200
    ) -> Dict:
        # start_time = time.perf_counter()
        ts = MarioLevel.tex_size
        jagent = jpype.JClass(str(agent))()
        if type(level) == str:
            level = MarioLevel.from_txt(level)
        reached_tile = 0
        res = {'restarts': [], 'trace': []}
        dx = 0
        while reached_tile < level.w - 1:
            jresult = self.__proxy.simulateWithRealTimeSuspension(JString(str(level[:, reached_tile:])), jagent, k, b)
            pyresult = MarioProxy.__extract_res(jresult)

            reached = pyresult['trace'][-1][0]
            reached_tile += ceil(reached / ts)
            if pyresult['status'] != 'WIN':
                res['restarts'].append(reached_tile)
            res['trace'] += [[dx + item[0], item[1]] for item in pyresult['trace']]
            dx = reached_tile * ts
        return res

def level_sum(lvls) -> MarioLevel:
    if type(lvls[0]) == MarioLevel:
        concated_content = np.concatenate([l.content for l in lvls], axis=1)
    else:
        concated_content = np.concatenate([l for l in lvls], axis=1)
    return MarioLevel(concated_content)

def traverse_level_files(path='levels/train'):
    for lvl_path in glob.glob(get_path(f'{path}\\*.txt')):
        lvl = MarioLevel.from_txt(lvl_path)
        name = lvl_path.split('\\')[-1][:-4]
        yield lvl, name


if __name__ == '__main__':
    simulator = MarioProxy()
    lvl = MarioLevel.from_txt('levels/train/mario-1-1.txt')
    simulator.play_game(lvl)
