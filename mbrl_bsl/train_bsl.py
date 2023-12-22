import hydra
import numpy as np
import omegaconf
import torch
from datetime import datetime

import os
import sys
import mbrl.algorithms.mbpo as mbpo
import mbrl.algorithms.pets as pets
import mbrl.algorithms.planet as planet
import mbrl.util.env
import gym
from omegaconf import OmegaConf

for sys_pr in ["./","../","../"]:
    sys.path.append(sys_pr)
    sys.path.append(sys_pr + "envs")

from car_env import car_term_fn, CarEnv
from maze_env import maze_term_fn, MazeEnv
from ship_env import ship_term_fn, ShipEnv
from rover_env import rover_term_fn, RoverEnv
from panda_env import panda_term_fn, PandaEnv
import time

class MockArgs:
    pass

def proc_env(env, cfg):
    env = gym.wrappers.TimeLimit(
            env, max_episode_steps=cfg.overrides.trial_length
        )
    if cfg.seed is not None:
        env.seed(cfg.seed)
        env.observation_space.seed(cfg.seed + 1)
        env.action_space.seed(cfg.seed + 2)
    return env

class Logger(object):
    def __init__(self):
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
        self._terminal = sys.stdout
        self.is_created=False
        self.buffer = []

    def create_log(self, log_path):
        self.log = open(log_path + "/log.txt", "a", 1)
        self.is_created = True
        for message in self.buffer:
            self.log.write(message)

    def write(self, message, only_file=False):
        if not only_file:
            self._terminal.write(message)
        if self.is_created:
            self.log.write(message)
        else:
            self.buffer.append(message)

    def flush(self):
        pass

def get_env(cfg, full_exp_dir, is_test=False, pets=False):
    # setup environment
    args = MockArgs()
    if not is_test:
        args.exp_dir_full = full_exp_dir
        args.write_csv = True
        args.mode = cfg.overrides.env
        env_name = cfg.overrides.env
    else:
        args.mode = cfg
        env_name = cfg

    if env_name == "car":
        args.num_samples = 50000
        args.nt = 25
        args.dt = 0.1
        args.amax = 10.0
        args.vmax = 10.0
        args.safe_thres = 1.0
        args.stop_x = -1.0
        args.traffic_x = -1.0
        args.stop_t = 1.0
        args.phase_t = 8.0
        args.phase_red = 4.0
        args.heading = True
        args.triggered = True
        args.mock = False
        args.no_tri_mock = False
        args.v_loss = 0.1
        args.s_loss = 0.1
        args.xo_max = 10.0
        args.hybrid = False
        args.test = False
        args.stl_reward = True
        args.acc_reward = False
        args.smoothing_factor = 100.0
        env = CarEnv(args=args)
        if not is_test:
            test_env = CarEnv(args=args)
        term_fn = car_term_fn
    elif env_name == "maze":
        args.num_samples = 50000
        args.nt = 25
        args.dt = 0.2
        args.smoothing_factor = 500.0
        args.canvas_w = 5
        args.dy_min = 1.5
        args.dy_max = 2.5
        args.goal_w = 0.8
        args.obs_h = 0.5
        args.obs_w_min = 2.0
        args.obs_w_max = 4.0
        args.v_min = -5.0
        args.v_max = 5.0
        args.vy = 1.0
        args.obs_ratio = 1.5
        args.y_level = 2.5
        args.test = False
        args.stl_reward = True
        args.acc_reward = False
        args.c_val = 0.5
        env = MazeEnv(args=args)
        if not is_test:
            test_env = MazeEnv(args=args)
        term_fn = maze_term_fn
    elif env_name == "ship1":
        args.num_samples = 50000
        args.nt = 20
        args.dt = 0.15
        args.smoothing_factor = 5.0
        args.stl_sim_steps = 2
        args.num_sim_steps = 2
        args.n_obs = 2
        args.obs_rmax = 1.0
        args.river_width = 4.0
        args.range_x = 15.0
        args.thrust_max = 0.5
        args.delta_max = 3.0
        args.s_phimax = 0.5
        args.s_umin = 3.0
        args.s_umax = 5.0
        args.s_vmax = 0.3
        args.s_rmax = 0.5
        args.canvas_h = 4.0
        args.canvas_w = 15.0
        args.stl_reward = True
        args.acc_reward = False
        args.c_val = 0.5
        env = ShipEnv(args=args)
        if not is_test:
            test_env = ShipEnv(args=args)
        term_fn = ship_term_fn
    elif env_name == "ship2":
        args.num_samples = 50000
        args.nt = 20
        args.dt = 0.15
        args.smoothing_factor = 500.0
        args.stl_sim_steps = 2
        args.num_sim_steps = 2
        args.n_obs = 1
        args.obs_rmin = 0.6
        args.obs_rmax = 1.2
        args.river_width = 4.0
        args.range_x = 15.0
        args.thrust_max = 0.5
        args.delta_max = 3.0
        args.s_phimax = 0.5
        args.s_umin = 3.0
        args.s_umax = 5.0
        args.s_vmax = 0.3
        args.s_rmax = 0.5
        args.canvas_h = 4.0
        args.canvas_w = 15.0

        args.track_thres = 0.3
        args.tmax = 25
        args.obs_ymin = -0.0
        args.obs_ymax = 0.0
        args.obs_xmin = -1.0
        args.obs_xmax = 8.0

        args.bloat_d = 0.0
        args.origin_sampling = False
        args.origin_sampling2 = False
        args.origin_sampling3 = True
        args.dist_w = 0.0
        args.obs_specific = True
        args.stl_reward = True
        args.acc_reward = False
        args.c_val = 0.5
        env = ShipEnv(args=args)
        if not is_test:
            test_env = ShipEnv(args=args)
        term_fn = ship_term_fn
    elif env_name == "rover":
        args.num_samples = 50000
        args.nt = 10
        args.dt = 0.2
        args.smoothing_factor = 500.0

        args.rover_vmax = 10.0
        args.astro_vmax = 0.0
        args.rover_vmin = 0.0
        args.astro_vmin = 0.0
        args.close_thres = 0.8
        args.battery_decay = 1.0
        args.battery_charge = 5.0
        args.obs_w = 3.0
        args.dist_w = 0.01
        args.seg_gain = 1.0
        args.hold_t = 3
        args.tanh_ratio = 0.05
        args.stl_reward = True
        args.acc_reward = False
        args.c_val = 0.5
        
        args.no_acc_mask = True
        args.no_tanh = True
        args.norm_ap = True
        args.hard_soft_step = True

        env = RoverEnv(args=args)
        if not is_test:
            test_env = RoverEnv(args=args)
        term_fn = rover_term_fn
    elif env_name == "panda" or env_name == "arm":
        args.num_samples = 50000
        args.nt = 10
        args.dt = 0.1
        args.smoothing_factor = 500.0

        args.tanh_ratio = 0.05
        args.stl_reward = True
        args.acc_reward = False
        args.c_val = 0.5
        args.u_max = 4

        env = PandaEnv(args=args)
        if not is_test:
            test_env = PandaEnv(args=args)
        term_fn = panda_term_fn
    else:
        print("cannot find env: %s"%(env_name))
        raise NotImplementedError

    reward_fn = env.reward_fn_torch
    if not is_test:
        if pets:
            return env, None, term_fn, reward_fn
        else:
            return env, test_env, term_fn
    else:
        if pets:
            return env, args, term_fn, reward_fn
        else:
            return env, args

@hydra.main(config_path="mbrl/examples/conf", config_name="main")
def main(cfg: omegaconf.DictConfig):
    
    # stdout logger
    logger = Logger()

    # full_exp_dir = os.getcwd()
    full_exp_dir = "g%s_%s_%s%s"%(logger._timestr, cfg.algorithm.name, cfg.seed, cfg.suffix)
    full_exp_dir = os.getcwd()
    # full_exp_dir = "../../"

    logger.create_log(full_exp_dir)
    sys.stdout = logger
    logger.write("python " + " ".join(sys.argv) + "\n", only_file=True)
    with open("%s/cfg.yaml"%(full_exp_dir), "w") as fp:
        omegaconf.OmegaConf.save(config=cfg, f=fp.name)

    # random seed
    print("Set random seed to %d"%(cfg.seed))
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log_dir_bak = os.getcwd()
    # log_dir_bak = full_exp_dir
    # os.chdir(hydra.utils.get_original_cwd())

    if cfg.algorithm.name == "pets":
        env, _, term_fn, reward_fn = get_env(cfg, full_exp_dir, pets=True)
    else:
        env, test_env, term_fn = get_env(cfg, full_exp_dir)

    # set random seed
    env = proc_env(env, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env = proc_env(test_env, cfg)
    # reward_fn = None
    
    os.chdir(log_dir_bak)
    # PETS: Deep reinforcement learning in a handful of trials using probabilistic dynamics models
    # MBPO: Model-based policy optimization
    # PlaNet: Learning latent dynamics for planning from pixels
    if cfg.algorithm.name == "pets":
        pets.train(env, term_fn, reward_fn, cfg)
    elif cfg.algorithm.name == "mbpo":
        mbpo.train(env, test_env, term_fn, cfg)


if __name__ == "__main__":
    main()