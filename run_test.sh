#python exp_test.py &

#python exp_test.py --designer_path 'exp_data/main/f' --disable_ctrl True &
#python exp_test.py --designer_path 'exp_data/main/p' --disable_ctrl True &
#python exp_test.py --designer_path 'exp_data/main/fp' --disable_ctrl True &
#
#python exp_test.py --designer_path 'exp_data/main/c'&
#python exp_test.py --designer_path 'exp_data/main/fc'&
#python exp_test.py --designer_path 'exp_data/main/cp'&

#python exp_test.py --music_name 'Farewell' &
#python exp_test.py --music_name 'Ginseng' --agent_name 'Sloane' &
#python exp_test.py --music_name 'Farewell' --agent_name 'Sloane' &
#python exp_test.py --music_name 'Ginseng' --agent_name 'Hartmann' &
#python exp_test.py --music_name 'Farewell' --agent_name 'Hartmann' &
#python exp_test.py --music_name 'Ginseng' --agent_name 'Michal' &
#python exp_test.py --music_name 'Farewell' --agent_name 'Michal' &
#python exp_test.py --music_name 'Ginseng' --agent_name 'Polikarpov' &
#python exp_test.py --music_name 'Farewell' --agent_name 'Polikarpov' &
#python exp_test.py --music_name 'Ginseng' --agent_name 'Schumann' &
#python exp_test.py --music_name 'Farewell' --agent_name 'Schumann' ;

python rewards_test.py --designer_path 'exp_data/sac/f' --n_parallel 50 --n_trials 100 --no_controllability;
python rewards_test.py --designer_path 'exp_data/sac/p' --n_parallel 50 --n_trials 100 --no_controllability;
python rewards_test.py --designer_path 'exp_data/sac/fp' --n_parallel 50 --n_trials 100 --no_controllability;
python rewards_test.py --designer_path 'exp_data/sac/c' --n_parallel 50 --n_trials 100;
python rewards_test.py --designer_path 'exp_data/sac/fc' --n_parallel 50 --n_trials 100;
python rewards_test.py --designer_path 'exp_data/sac/cp' --n_parallel 50 --n_trials 100;
python rewards_test.py --designer_path 'exp_data/sac/fcp' --n_parallel 50 --n_trials 100;

