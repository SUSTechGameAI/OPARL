"""
  @Time : 2022/2/18 15:24 
  @Author : Ziqi Wang
  @File : olgen_analyze.py 
"""
import json
import os
import time
from copy import deepcopy
from math import ceil

import numpy as np

from smb import MarioJavaAgents, MarioProxy, MarioLevel, level_sum
from src.environment.reward_terms import FunTest
from src.online_generation.ol_gensys import OnlineGenerationSystem, AggregatedGenerator
from src.utils import RingQueue, get_path


def test_olgen_joint(olgen_sys, agent=MarioJavaAgents.Baumgarten, n_segs=50):
    # TODO: save all level!
    start_time = time.time()

    simulator = MarioProxy()
    olgen_sys.reset()
    level = olgen_sys.step()
    # olgen_sys.recieve_play_data(0)
    time_passed = 0
    seg_archive = RingQueue(2)
    seg_archive.push(level)
    fun_evaluator = FunTest()
    fun = 0.
    ends = [0]

    for i in range(1, n_segs):
        seg = olgen_sys.step()
        level += seg
        trace = simulator.simulate_with_reset(level, agent)['trace']

        p = len(trace) - 1
        while trace[p][0] > 16 * i * MarioLevel.default_seg_width:
            p -= 1
        delta = p / 30 - time_passed
        time_passed = p / 30
        olgen_sys.recieve_play_data(delta)
        fun = fun + fun_evaluator.compute_reward(seg=seg, archive=seg_archive.to_list())
        end = ends[-1] + delta
        ends.append(end)

    res = simulator.simulate_with_reset(level, agent)
    trace = res['trace']
    ends.append(len(trace) / 30)
    res = {'fun': (-fun / n_segs) ** 0.5}

    ideal_featr_seq = olgen_sys.controller.ideal_featrs
    eps_in = 0.
    eps_out = 0.
    eps_all = 0.
    T = 0
    print(len(ends), len(olgen_sys.f_hat_seq), len(olgen_sys.f_real_seq))
    print(ends, olgen_sys.f_hat_seq, olgen_sys.f_real_seq, sep='\n')
    for i in range(len(olgen_sys.f_hat_seq)):
        for idea_f in ideal_featr_seq.get(ends[i], ends[i+1]):
            T += 1
            eps_in += abs(idea_f - olgen_sys.f_hat_seq[i])
            eps_out += abs(olgen_sys.f_hat_seq[i] - olgen_sys.f_real_seq[i])
            eps_all += abs(idea_f - olgen_sys.f_real_seq[i])
    res['eps_in'] = eps_in / T
    res['eps_out'] = eps_out / T
    res['eps_all'] = eps_all / T
    print((time.time() - start_time) / 60)
    return level, res

def test_olgen_sep(olgen_sys, agent=MarioJavaAgents.Baumgarten, n_segs=50, z=None):
    start_time = time.time()

    simulator = MarioProxy()
    olgen_sys.reset(z)
    all_segs = []
    seg = olgen_sys.step()
    all_segs.append(seg)

    seg_archive = RingQueue(2)
    seg_archive.push(seg)
    fun_evaluator = FunTest()
    fun = 0.
    ends = [0.]
    for i in range(1, n_segs):
        seg = olgen_sys.step()
        trace = simulator.simulate_with_reset(all_segs[-1], agent)['trace']
        delta = len(trace) / 30
        olgen_sys.recieve_play_data(delta)
        all_segs.append(seg)
        fun = fun + fun_evaluator.compute_reward(seg=seg, archive=seg_archive.to_list())
        seg_archive.push(seg)
        end = ends[-1] + delta
        ends.append(end)

    trace = simulator.simulate_with_reset(all_segs[-1], agent)['trace']
    end = ends[-1] + len(trace) / 30
    ends.append(end)
    res = {'fun': fun / (n_segs-1)}

    ideal_featr_seq = olgen_sys.controller.ideal_featrs
    eps_in = 0.
    eps_out = 0.
    eps_all = 0.
    T = 0

    for i in range(len(olgen_sys.f_hat_seq)):
        for idea_f in ideal_featr_seq.get(ends[i], ends[i+1]):
            T += 1
            eps_in += abs(idea_f - olgen_sys.f_hat_seq[i])
            eps_out += abs(olgen_sys.f_hat_seq[i] - olgen_sys.f_real_seq[i])
            eps_all += abs(idea_f - olgen_sys.f_real_seq[i])
    res['eps_in'] = eps_in / T
    res['eps_out'] = eps_out / T
    res['eps_all'] = eps_all / T
    res['ends'] = ends
    res['playability'] = olgen_sys.resample_fails
    print(time.time() - start_time)
    return level_sum(all_segs), res

def test_and_save(designer_path, music_name, agent, n_segs=50, n_trials=30, z=None, disable_ctrl=False):
    results = []
    path = f'{designer_path}/{music_name}_{agent.name}'
    os.makedirs(get_path(path), exist_ok=True)
    for i in range(n_trials):
        level, res = test_olgen_sep(
            OnlineGenerationSystem(
                AggregatedGenerator(designer_path + '/actor.pth'),
                f'assets/{music_name}_diffs.json',
                disable_ctrl
            ),
            agent, n_segs, z
        )

        results.append(res)
        level.save(f'{path}/lvl{i}.txt')
        with open(get_path(f'{path}/simulation_res.json'), 'w') as f:
            json.dump(results, f)


def test_time(music, agent, latvec):
    _, res = test_olgen_sep(
        OnlineGenerationSystem(
            AggregatedGenerator('exp_data/sac/fcp/actor.pth'),
            f'assets/{music}_diffs.json'
        ),
        agent, z=latvec
    )
    with open(get_path(f'exp_data/sac/fcp/time_test/{music}_{agent.name}.json'), 'w') as f:
        json.dump(res['ends'], f)


if __name__ == '__main__':
    z = np.random.rand(20)

    test_time('Ginseng', MarioJavaAgents.Baumgarten, z)
    test_time('Ginseng', MarioJavaAgents.Sloane, z)
    test_time('Ginseng', MarioJavaAgents.Schumann, z)
    test_time('Ginseng', MarioJavaAgents.Hartmann, z)
    test_time('Ginseng', MarioJavaAgents.Polikarpov, z)

    test_time('Farewell', MarioJavaAgents.Baumgarten, z)
    test_time('Farewell', MarioJavaAgents.Sloane, z)
    test_time('Farewell', MarioJavaAgents.Schumann, z)
    test_time('Farewell', MarioJavaAgents.Hartmann, z)
    test_time('Farewell', MarioJavaAgents.Polikarpov, z)
