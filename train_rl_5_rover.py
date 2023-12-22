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
    add("--epochs", type=int, default=250000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=10)
    add("--dt", type=float, default=0.2)
    add("--print_freq", type=int, default=500)
    add("--viz_freq", type=int, default=5000)
    add("--save_freq", type=int, default=10000)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)   

    add("--sim_freq", type=int, default=1)
    add("--rover_vmax", type=float, default=10.0)
    add("--astro_vmax", type=float, default=0.0)
    add("--rover_vmin", type=float, default=0.0)
    add("--astro_vmin", type=float, default=0.0)
    add("--close_thres", type=float, default=0.8)
    add("--battery_decay", type=float, default=1.0)
    add("--battery_charge", type=float, default=5.0)
    add("--obs_w", type=float, default=3.0)
    add("--ego_turn", action="store_true", default=False)
    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--no_obs", action="store_true", default=False)
    add("--one_obs", action="store_true", default=False)
    add("--limited", action="store_true", default=False)
    add("--if_cond", action='store_true', default=False)
    add("--nominal", action='store_true', default=False)
    add("--dist_w", type=float, default=0.01)
    add("--no_acc_mask", action='store_true', default=False)
    add("--seq_reach", action='store_true', default=False)
    add("--together_ratio", type=float, default=0.2)
    add("--until_emergency", action='store_true', default=False)
    add("--list_and", action='store_true', default=False)
    add("--mpc_update_freq", type=int, default=1)
    add("--seg_gain", type=float, default=1.0)
    add("--hold_t", type=int, default=3)
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=0.05)
    add("--update_init_freq", type=int, default=500)

    # new-tricks
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    # new framework specific
    add("--mode", type=str, choices=["car", "maze", "ship1", "ship2", "rover", "panda"], default="rover")
    add("--train_rl", action='store_true', default=False)
    add("--num_workers", type=int, default=None)
    add("--stl_reward", action='store_true', default=False)
    add("--acc_reward", action='store_true', default=False)
    add("--c_val", type=float, default=0.5)
    
    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)

    args = parser.parse_args()
    args.no_acc_mask = True
    args.no_tanh = True
    args.norm_ap = True
    args.hard_soft_step = True

    t1=time.time()
    main(args)
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))