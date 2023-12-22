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
    add("--lr", type=float, default=5e-4)
    add("--nt", type=int, default=25)
    add("--dt", type=float, default=0.1)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=100.0)
    add("--sim", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)
    add("--amax", type=float, default=10)
    add("--stop_x", type=float, default=-1.0)
    add("--v_loss", type=float, default=0.1)
    add("--phase_t", type=float, default=8.0)
    add("--phase_red", type=float, default=4.0)
    add("--traffic_x", type=float, default=-1.0)
    add("--sim_freq", type=int, default=5)
    add("--stop_t", type=float, default=1.0)
    add("--vmax", type=float, default=10.0)
    add("--s_loss", type=float, default=0.1)
    add("--inter_x", type=float, default=0.0)

    add("--test", action='store_true', default=False)
    add("--triggered", action='store_true', default=False)
    add('--heading', action='store_true', default=False)

    add("--safe_thres", type=float, default=1.0)
    add("--xo_max", type=float, default=10.0)

    add('--mock', action='store_true', default=False)
    add('--no_tri_mock', action='store_true', default=False)
    add('--hybrid', action='store_true', default=False)
    add('--bloat_dist', type=float, default=1.0)
    add('--no_viz', action='store_true', default=False)

    # new-tricks
    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=1.0)
    add("--update_init_freq", type=int, default=-1)
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    # new framework specific
    add("--mode", type=str, choices=["car", "maze", "ship", "rover", "panda"], default="car")
    add("--train_rl", action='store_true', default=False)
    add("--num_workers", type=int, default=None)
    add("--stl_reward", action='store_true', default=False)
    add("--acc_reward", action='store_true', default=False)

    add("--mpc", action="store_true", default=False)
    add("--plan", action="store_true", default=False)
    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)
    add("--rl", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)
    add("--c_val", type=float, default=0.5)
    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)
    args = parser.parse_args()
    args.triggered=True
    args.heading=True
    
    t1=time.time()
    main(args)
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))