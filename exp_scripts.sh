#################### BENCHMARK-1 Traffic ####################
# Training for all learning based methods  (1~5: baselines, 10 is our method)
# 1. (RL_r) RL baseline with a manually designed reward
python train_rl_1_car.py -e exp1_rl_raw --train_rl --epochs 400000 --num_workers 8
# 2. (RL_s) RL baseline with STL robustness score as reward
python train_rl_1_car.py -e exp1_rl_stl --train_rl --epochs 400000 --num_workers 8 --stl_reward
# 3. (RL_a) RL baseline with STL accuracy as reward
python train_rl_1_car.py -e exp1_rl_acc --train_rl --epochs 400000 --num_workers 8 --acc_reward
# 4. (MBPO) Model-based RL baseline via policy optimization 
cd mbrl_bsl && python train_bsl.py algorithm=mbpo overrides=mbpo_car seed=1007 device=cuda:0 overrides.num_sac_updates_per_step=10 suffix=_upd10 && cd -
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
cd mbrl_bsl && python train_bsl.py algorithm=pets overrides=pets_car seed=1007 device=cuda:0 suffix=_bsl && cd -
# 10. (Ours) Our proposed method STL_NPC
python run_stl_1_car.py -e e1_traffic --lr 5e-4 

# Testing for all methods (1~9: baselines, 10 and 11 are our methods)
#TODO you need to change the model path accordingly
#TODO for non-learning methods, just give them any existing log dir file
#TODO if you want to see visualizations, just remove the `--no_viz` flag
# 1. (RL_r) RL baseline with a manually designed reward
python run_stl_1_car.py --test --rl -R exp1_rl_raw/models/model_last.zip --no_viz
# 2. (RL_s) RL baseline with STL robustness score as reward
python run_stl_1_car.py --test --rl --rl_stl -R exp1_rl_stl/models/model_last.zip --no_viz
# 3. (RL_a) RL baseline with STL accuracy as reward
python run_stl_1_car.py --test --rl --rl_acc -R exp1_rl_acc/models/model_last.zip --no_viz
# 4. (MBPO) Model-based RL baseline via policy optimization 
python run_stl_1_car.py --test --mbpo -R gmmdd-hhmmss_car_mbpo_1007_upd10/sac.pth --no_viz
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
python run_stl_1_car.py --test --pets -R gmmdd-hhmmss_car_pets_1007_bsl/model.pth --no_viz
# 6. (CEM) Cross-entropy method baseline
python run_stl_1_car.py --test -P e1_traffic/models/model_49000.ckpt --cem --no_viz
# 7. (MPC) Model-predictive control baseline
python run_stl_1_car.py --test -P e1_traffic/models/model_49000.ckpt --mpc --no_viz
# 8. (STL_m) STL planner baseline using Mixed-integer Linear Programming (MiLP)
python run_stl_1_car.py --test -P e1_traffic/models/model_49000.ckpt --plan --no_viz
# 9. (STL_g) STL planner baseline using gradient-based method
python run_stl_1_car.py --test -P e1_traffic/models/model_49000.ckpt --grad --grad_steps 50  --no_viz
# 10. (Ours) Our proposed method STL_NPC
python run_stl_1_car.py --test -P e1_traffic/models/model_49000.ckpt --no_viz
# 11. (Ours_f) Our propoased method STL_NPC with the backup policy
python run_stl_1_car.py --test -P e1_traffic/models/model_49000.ckpt --no_viz --finetune




#################### BENCHMARK-2 Maze Game ####################
# Training for all learning based methods  (1~5: baselines, 10 is our method)
# 1. (RL_r) RL baseline with a manually designed reward
python train_rl_2_maze.py -e exp2_rl_raw --train_rl --epochs 400000 --num_workers 8
# 2. (RL_s) RL baseline with STL robustness score as reward
python train_rl_2_maze.py -e exp2_rl_stl --train_rl --epochs 400000 --num_workers 8 --stl_reward
# 3. (RL_a) RL baseline with STL accuracy as reward
python train_rl_2_maze.py -e exp2_rl_acc --train_rl --epochs 400000 --num_workers 8 --acc_reward
# 4. (MBPO) Model-based RL baseline via policy optimization 
cd mbrl_bsl && python train_bsl.py algorithm=mbpo overrides=mbpo_maze seed=1007 device=cuda:0 overrides.num_sac_updates_per_step=10 suffix=_upd10 && cd -
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
cd mbrl_bsl && python train_bsl.py algorithm=pets overrides=pets_maze seed=1007 device=cuda:0 suffix=_bsl && cd -
# 10. (Ours) Our proposed method STL_NPC
python run_stl_2_maze.py -e e2_game --lr 3e-4

######TODO######

# Testing for all methods (1~9: baselines, 10 and 11 are our methods)
# 1. (RL_r) RL baseline with a manually designed reward
python run_stl_2_maze.py --test --rl -R exp2_rl_raw/models/model_last.zip --no_viz
# 2. (RL_s) RL baseline with STL robustness score as reward
python run_stl_2_maze.py --test --rl --rl_stl -R exp2_rl_stl/models/model_last.zip --no_viz
# 3. (RL_a) RL baseline with STL accuracy as reward
python run_stl_2_maze.py --test --rl --rl_acc -R exp2_rl_acc/models/model_last.zip --no_viz
# 4. (MBPO) Model-based RL baseline via policy optimization 
python run_stl_2_maze.py --test --mbpo -R gmmdd-hhmmss_maze_mbpo_1007_upd10/sac.pth --no_viz
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
python run_stl_2_maze.py --test --pets -R gmmdd-hhmmss_maze_pets_1007_bsl/model.pth --no_viz
# 6. (CEM) Cross-entropy method baseline
python run_stl_2_maze.py --test -P e2_game/models/model_49900.ckpt --cem --no_viz
# 7. (MPC) Model-predictive control baseline
python run_stl_2_maze.py --test -P e2_game/models/model_49900.ckpt --mpc --no_viz
# 8. (STL_m) STL planner baseline using Mixed-integer Linear Programming (MiLP)
python run_stl_2_maze.py --test -P e2_game/models/model_49900.ckpt --plan --no_viz
# 9. (STL_g) STL planner baseline using gradient-based method
python run_stl_2_maze.py --test -P e2_game/models/model_49900.ckpt --grad --grad_steps 50 --no_viz
# 10. (Ours) Our proposed method STL_NPC
python run_stl_2_maze.py --test -P e2_game/models/model_49900.ckpt --no_viz
# 11. (Ours_f) Our propoased method STL_NPC with the backup policy
python run_stl_2_maze.py --test -P e2_game/models/model_49900.ckpt --no_viz --finetune




#################### BENCHMARK-3 Safe Ship Control ####################
# Training for all learning based methods  (1~5: baselines, 10 is our method)
# 1. (RL_r) RL baseline with a manually designed reward
python train_rl_3_ship_safe.py -e exp3_rl_raw --train_rl --epochs 400000 --num_workers 8
# 2. (RL_s) RL baseline with STL robustness score as reward
python train_rl_3_ship_safe.py -e exp3_rl_stl --train_rl --epochs 400000 --num_workers 8 --stl_reward
# 3. (RL_a) RL baseline with STL accuracy as reward
python train_rl_3_ship_safe.py -e exp3_rl_acc --train_rl --epochs 400000 --num_workers 8 --acc_reward
# 4. (MBPO) Model-based RL baseline via policy optimization 
cd mbrl_bsl && python train_bsl.py algorithm=mbpo overrides=mbpo_ship1 seed=1007 device=cuda:0 overrides.num_sac_updates_per_step=10 suffix=_upd10 && cd -
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
cd mbrl_bsl && python train_bsl.py algorithm=pets overrides=pets_ship1 seed=1007 device=cuda:0 suffix=_bsl && cd -
# 10. (Ours) Our proposed method STL_NPC
python run_stl_3_ship_safe.py -e e3_ship_safe --lr 3e-4

# Testing for all methods (1~9: baselines, 10 and 11 are our methods)
# 1. (RL_r) RL baseline with a manually designed reward
python run_stl_3_ship_safe.py --test --rl -R exp3_rl_raw/models/model_last.zip --no_viz --num_trials 50
# 2. (RL_s) RL baseline with STL robustness score as reward
python run_stl_3_ship_safe.py --test --rl --rl_stl -R exp3_rl_stl/models/model_last.zip --no_viz --num_trials 50
# 3. (RL_a) RL baseline with STL accuracy as reward
python run_stl_3_ship_safe.py --test --rl --rl_acc -R exp3_rl_acc/models/model_last.zip --no_viz --num_trials 50
# 4. (MBPO) Model-based RL baseline via policy optimization 
python run_stl_3_ship_safe.py --test --mbpo -R gmmdd-hhmmss_ship1_mbpo_1007_upd10/sac.pth --no_viz --num_trials 50
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
python run_stl_3_ship_safe.py --test --pets -R gmmdd-hhmmss_ship1_pets_1007_bsl/model.pth --no_viz --num_trials 50
# 6. (CEM) Cross-entropy method baseline
python run_stl_3_ship_safe.py --test -P e3_ship_safe/models/model_45000.ckpt --cem --no_viz --num_trials 50
# 7. (MPC) Model-predictive control baseline
python run_stl_3_ship_safe.py --test -P e3_ship_safe/models/model_45000.ckpt --mpc --no_viz --num_trials 50
# 8. (STL_m) STL planner baseline using Mixed-integer Linear Programming (MiLP)
python run_stl_3_ship_safe.py --test -P e3_ship_safe/models/model_45000.ckpt --plan --no_viz --num_trials 50
# 9. (STL_g) STL planner baseline using gradient-based method
python run_stl_3_ship_safe.py --test -P e3_ship_safe/models/model_45000.ckpt --grad --no_viz --num_trials 50
# 10. (Ours) Our proposed method STL_NPC
python run_stl_3_ship_safe.py --test -P e3_ship_safe/models/model_45000.ckpt --no_viz --num_trials 50
# 11. (Ours_f) Our propoased method STL_NPC with the backup policy
python run_stl_3_ship_safe.py --test -P e3_ship_safe/models/model_45000.ckpt --no_viz --finetune --num_trials 50




#################### BENCHMARK-4 Ship Tracking Control ####################
# Training for all learning based methods  (1~5: baselines, 10 is our method)
# 1. (RL_r) RL baseline with a manually designed reward
python train_rl_4_ship_track.py -e exp4_rl_raw --train_rl --epochs 400000 --num_workers 8
# 2. (RL_s) RL baseline with STL robustness score as reward
python train_rl_4_ship_track.py -e exp4_rl_stl --train_rl --epochs 400000 --num_workers 8 --stl_reward
# 3. (RL_a) RL baseline with STL accuracy as reward
python train_rl_4_ship_track.py -e exp4_rl_acc --train_rl --epochs 400000 --num_workers 8 --acc_reward
# 4. (MBPO) Model-based RL baseline via policy optimization 
cd mbrl_bsl && python train_bsl.py algorithm=mbpo overrides=mbpo_ship2 seed=1007 device=cuda:0 overrides.num_sac_updates_per_step=10 suffix=_upd10 && cd -
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
cd mbrl_bsl && python train_bsl.py algorithm=pets overrides=pets_ship2 seed=1007 device=cuda:0 suffix=_bsl && cd -
# 10. (Ours) Our proposed method STL_NPC
python run_stl_4_ship_track.py -e e4_ship_track --lr 3e-4

# Testing for all methods (1~9: baselines, 10 and 11 are our methods)
# 1. (RL_r) RL baseline with a manually designed reward
python run_stl_4_ship_track.py --test --rl -R exp4_rl_raw/models/model_last.zip --no_viz --num_trials 20
# 2. (RL_s) RL baseline with STL robustness score as reward
python run_stl_4_ship_track.py --test --rl --rl_stl -R exp4_rl_stl/models/model_last.zip --no_viz --num_trials 20
# 3. (RL_a) RL baseline with STL accuracy as reward
python run_stl_4_ship_track.py --test --rl --rl_acc -R exp4_rl_acc/models/model_last.zip --no_viz --num_trials 20
# 4. (MBPO) Model-based RL baseline via policy optimization 
python run_stl_4_ship_track.py --test --mbpo -R gmmdd-hhmmss_ship2_mbpo_1007_upd10/sac.pth --no_viz --num_trials 20
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
python run_stl_4_ship_track.py --test --pets -R gmmdd-hhmmss_ship2_pets_1007_bsl/model.pth --no_viz --num_trials 20
# 6. (CEM) Cross-entropy method baseline
python run_stl_4_ship_track.py --test -P e4_ship_track/models/model_49000.ckpt --cem --no_viz --num_trials 20
# 7. (MPC) Model-predictive control baseline
python run_stl_4_ship_track.py --test -P e4_ship_track/models/model_49000.ckpt --mpc --no_viz --num_trials 20
# 8. (STL_m) STL planner baseline using Mixed-integer Linear Programming (MiLP)
python run_stl_4_ship_track.py --test -P e4_ship_track/models/model_49000.ckpt --plan --no_viz --num_trials 20
# 9. (STL_g) STL planner baseline using gradient-based method
python run_stl_4_ship_track.py --test -P e4_ship_track/models/model_49000.ckpt --grad --grad_steps 50 --no_viz --num_trials 20
# 10. (Ours) Our proposed method STL_NPC
python run_stl_4_ship_track.py --test -P e4_ship_track/models/model_49000.ckpt --no_viz --num_trials 20
# 11. (Ours_f) Our propoased method STL_NPC with the backup policy
python run_stl_4_ship_track.py --test -P e4_ship_track/models/model_49000.ckpt --no_viz --finetune --num_trials 20




#################### BENCHMARK-5 Robot Navigation ####################
# Training for all learning based methods  (1~5: baselines, 10 is our method)
# 1. (RL_r) RL baseline with a manually designed reward
python train_rl_5_rover.py -e exp5_rl_raw --train_rl --epochs 2000000 --num_workers 8
# 2. (RL_s) RL baseline with STL robustness score as reward
python train_rl_5_rover.py -e exp5_rl_stl --train_rl --epochs 2000000 --num_workers 8 --stl_reward
# 3. (RL_a) RL baseline with STL accuracy as reward
python train_rl_5_rover.py -e exp5_rl_acc --train_rl --epochs 2000000 --num_workers 8 --acc_reward
# 4. (MBPO) Model-based RL baseline via policy optimization 
cd mbrl_bsl && python train_bsl.py algorithm=mbpo overrides=mbpo_rover seed=1007 device=cuda:0 overrides.num_sac_updates_per_step=10 suffix=_upd10 && cd -
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
cd mbrl_bsl && python train_bsl.py algorithm=pets overrides=pets_rover seed=1007 device=cuda:0 suffix=_bsl && cd -
# 10. (Ours) Our proposed method STL_NPC
python run_stl_5_rover.py -e e5_rover --lr 1e-4

# Testing for all methods (1~9: baselines, 10 and 11 are our methods)
# 1. (RL_r) RL baseline with a manually designed reward
python run_stl_5_rover.py --test --rl -R exp5_rl_raw/models/model_last.zip --no_viz --mpc_update_freq 10 --multi_test
# 2. (RL_s) RL baseline with STL robustness score as reward
python run_stl_5_rover.py --test --rl --rl_stl -R exp5_rl_stl/models/model_last.zip --no_viz --mpc_update_freq 10 --multi_test
# 3. (RL_a) RL baseline with STL accuracy as reward
python run_stl_5_rover.py --test --rl --rl_acc -R exp5_rl_acc/models/model_last.zip --no_viz --mpc_update_freq 10 --multi_test
# 4. (MBPO) Model-based RL baseline via policy optimization 
python run_stl_5_rover.py --test --mbpo -R gmmdd-hhmmss_rover_mbpo_1007_upd10/sac.pth --no_viz --mpc_update_freq 10 --multi_test
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
python run_stl_5_rover.py --test --pets -R gmmdd-hhmmss_rover_pets_1007_bsl/model.pth --no_viz --mpc_update_freq 10 --multi_test
# 6. (CEM) Cross-entropy method baseline
python run_stl_5_rover.py --test -P e5_rover/models/model_250000.ckpt --cem --no_viz --mpc_update_freq 10 --multi_test
# 7. (MPC) Model-predictive control baseline
python run_stl_5_rover.py --test -P e5_rover/models/model_250000.ckpt --mpc --no_viz --mpc_update_freq 10 --multi_test
# 8. (STL_m) STL planner baseline using Mixed-integer Linear Programming (MiLP)
python run_stl_5_rover.py --test -P e5_rover/models/model_250000.ckpt --plan --no_viz --mpc_update_freq 10 --multi_test
# 9. (STL_g) STL planner baseline using gradient-based method
python run_stl_5_rover.py --test -P e5_rover/models/model_250000.ckpt --grad --grad_steps 50 --no_viz --mpc_update_freq 10 --multi_test
# 10. (Ours) Our proposed method STL_NPC
python run_stl_5_rover.py --test -P e5_rover/models/model_250000.ckpt --no_viz --mpc_update_freq 10 --multi_test
# 11. (Ours_f) Our propoased method STL_NPC with the backup policy
python run_stl_5_rover.py --test -P e5_rover/models/model_250000.ckpt --no_viz --finetune --mpc_update_freq 10 --multi_test




#################### BENCHMARK-6 Robot Manipulation ####################
# Training for all learning based methods  (1~5: baselines, 10 is our method)
# 1. (RL_r) RL baseline with a manually designed reward
python train_rl_6_panda.py -e exp6_rl_raw --train_rl --epochs 500000 --num_workers 8
# 2. (RL_s) RL baseline with STL robustness score as reward
python train_rl_6_panda.py -e exp6_rl_stl --train_rl --epochs 500000 --num_workers 8 --stl_reward
# 3. (RL_a) RL baseline with STL accuracy as reward
python train_rl_6_panda.py -e exp6_rl_acc --train_rl --epochs 500000 --num_workers 8 --acc_reward
# 4. (MBPO) Model-based RL baseline via policy optimization 
cd mbrl_bsl && python train_bsl.py algorithm=mbpo overrides=mbpo_panda seed=1007 device=cuda:0 overrides.num_sac_updates_per_step=10 suffix=_upd10 && cd -
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
cd mbrl_bsl && python train_bsl.py algorithm=pets overrides=pets_panda seed=1007 device=cuda:0 suffix=_bsl && cd -
# 10. (Ours) Our proposed method STL_NPC
python run_stl_6_panda.py -e e6_arm --lr 1e-4

# Testing for all methods (1~9: baselines, 10 and 11 are our methods)
# 1. (RL_r) RL baseline with a manually designed reward
python run_stl_6_panda.py --test --rl -R exp6_rl_raw/models/model_last.zip --no_viz
# 2. (RL_s) RL baseline with STL robustness score as reward
python run_stl_6_panda.py --test --rl --rl_stl -R exp6_rl_stl/models/model_last.zip --no_viz
# 3. (RL_a) RL baseline with STL accuracy as reward
python run_stl_6_panda.py --test --rl --rl_acc -R exp6_rl_acc/models/model_last.zip --no_viz
# 4. (MBPO) Model-based RL baseline via policy optimization 
python run_stl_6_panda.py --test --mbpo -R gmmdd-hhmmss_panda_mbpo_1007_upd10/sac.pth --no_viz
# 5. (PETS) Model-based RL baseline via probabilistic ensembles with trajectory sampling
python run_stl_6_panda.py --test --pets -R gmmdd-hhmmss_panda_pets_1007_bsl/model.pth --no_viz
# 6. (CEM) Cross-entropy method baseline
python run_stl_6_panda.py --test -P e6_arm/models/model_49000.ckpt --cem --no_viz
# 7. (MPC) Model-predictive control baseline
python run_stl_6_panda.py --test -P e6_arm/models/model_49000.ckpt --mpc --no_viz
# 8. (STL_m) STL planner baseline using Mixed-integer Linear Programming (MiLP)
python run_stl_6_panda.py --test -P e6_arm/models/model_49000.ckpt --plan --no_viz
# 9. (STL_g) STL planner baseline using gradient-based method
python run_stl_6_panda.py --test -P e6_arm/models/model_49000.ckpt --grad --no_viz
# 10. (Ours) Our proposed method STL_NPC
python run_stl_6_panda.py --test -P e6_arm/models/model_49000.ckpt --no_viz --mpc_update_freq 3
# 11. (Ours_f) Our propoased method STL_NPC with the backup policy
python run_stl_6_panda.py --test -P e6_arm/models/model_49000.ckpt --no_viz --finetune --mpc_update_freq 3