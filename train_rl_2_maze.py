import argparse
import time
from train_rl_base import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=3e-4)
    add("--nt", type=int, default=25)
    add("--dt", type=float, default=0.2)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)   

    add("--canvas_w", type=float, default=5)
    add("--dy_min", type=float, default=1.5)
    add("--dy_max", type=float, default=2.5)
    add("--goal_w", type=float, default=0.8)
    add("--obs_h", type=float, default=0.5)
    add("--obs_w_min", type=float, default=2.0)
    add("--obs_w_max", type=float, default=4)
    add("--v_min", type=float, default=-5)
    add("--v_max", type=float, default=5)
    add("--only_two", action='store_true', default=False)
    add("--sim_freq", type=int, default=1)
    add("--extra", action='store_true', default=False)
    add("--vy", type=float, default=1.0)
    add("--obs_ratio", type=float, default=1.5)
    add("--y_level", type=float, default=2.5)

    # new-tricks
    add("--hiddens", type=int, nargs="+", default=[64, 64, 64])
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=1.0)
    add("--update_init_freq", type=int, default=-1)
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    # new framework specific
    add("--mode", type=str, choices=["car", "maze", "ship", "rover", "panda"], default="maze")
    add("--train_rl", action='store_true', default=False)
    add("--num_workers", type=int, default=None)
    add("--stl_reward", action='store_true', default=False)
    add("--acc_reward", action='store_true', default=False)
    add("--c_val", type=float, default=0.5)

    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)
    args = parser.parse_args()
    t1=time.time()
    main(args)
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))