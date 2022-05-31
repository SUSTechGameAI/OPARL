#python train.py ppo_designer --n_envs 50 --rfunc_name 'f' --res_path 'main/f';
#python train.py ppo_designer --n_envs 50 --rfunc_name 'c' --res_path 'main/c';
#python train.py ppo_designer --n_envs 50 --rfunc_name 'p' --res_path 'main/p';
#python train.py ppo_designer --n_envs 50 --rfunc_name 'fc' --res_path 'main/fc';
#python train.py ppo_designer --n_envs 50 --rfunc_name 'fp' --res_path 'main/fp';
#python train.py ppo_designer --n_envs 50 --rfunc_name 'cp' --res_path 'main/cp';
#python train.py ppo_designer --n_envs 50 --rfunc_name 'fcp' --res_path 'main/fcp';

python exp_test.py &

python exp_test.py --designer_path 'exp_data/main/f' --disable_ctrl True &
python exp_test.py --designer_path 'exp_data/main/p' --disable_ctrl True &
python exp_test.py --designer_path 'exp_data/main/fp' --disable_ctrl True &

python exp_test.py --designer_path 'exp_data/main/c'&
python exp_test.py --designer_path 'exp_data/main/fc'&
python exp_test.py --designer_path 'exp_data/main/cp'&

python exp_test.py --music_name 'Farewell' &
python exp_test.py --music_name 'Ginseng' --agent_name 'Sloane' &
python exp_test.py --music_name 'Farewell' --agent_name 'Sloane' &
python exp_test.py --music_name 'Ginseng' --agent_name 'Hartmann' &
python exp_test.py --music_name 'Farewell' --agent_name 'Hartmann' &
python exp_test.py --music_name 'Ginseng' --agent_name 'Michal' &
python exp_test.py --music_name 'Farewell' --agent_name 'Michal' &
python exp_test.py --music_name 'Ginseng' --agent_name 'Polikarpov' &
python exp_test.py --music_name 'Farewell' --agent_name 'Polikarpov' &
python exp_test.py --music_name 'Ginseng' --agent_name 'Schumann' &
python exp_test.py --music_name 'Farewell' --agent_name 'Schumann' ;
