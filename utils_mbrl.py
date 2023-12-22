import os
import sys
import numpy as np
import mbrl.planning
import torch
from mbrl.third_party.pytorch_sac_pranz24.sac import SAC

from mbrl_bsl.train_bsl import MockArgs

sys.path.append("envs")

from car_env import CarEnv
from maze_env import MazeEnv
from rover_env import RoverEnv
from ship_env import ShipEnv

from mbrl_bsl.train_bsl import get_env
import omegaconf

# for mbrl (mbpo, pets)
def get_mbrl_u(obs, running, policy, mbpo=False):
    if mbpo:
        obs = obs.cuda().float()
        u = policy.select_action(obs.cpu(), batched=True, evaluate=True, numpy=False).cpu()
    else:
        u = torch.from_numpy(policy.act(obs.cpu())).float()
    return u


def findname(mbpo_path):
    for key in ["car", "maze", "ship1", "ship2", "rover", "arm", "panda"]:
        if key in mbpo_path:
            return key

def get_mbrl_models(mbrl_path, args, is_mbpo):
    if is_mbpo:
        return get_mbpo_models(mbrl_path, args)
    else:
        return get_pets_models(mbrl_path, args)


def get_mbpo_models(mbpo_path, args):    
    env, env_args = get_env(findname(mbpo_path), None, is_test=True)

    num_inputs = env.observation_space.shape[0]
    action_space = env.action_space

    sac_args = MockArgs()
    sac_args.gamma = 0.99
    sac_args.tau = 0.005
    sac_args.alpha = 0.2
    sac_args.policy = "Gaussian"
    sac_args.target_update_interval = 4
    sac_args.automatic_entropy_tuning = True
    sac_args.target_entropy = -0.05
    sac_args.hidden_size = 512
    sac_args.lr = 0.0003
    sac_args.batch_size = 256
    sac_args.device="cuda:0"

    agent = SAC(num_inputs, action_space, sac_args)
    agent.load_checkpoint(ckpt_path=mbpo_path, evaluate=True, map_location=sac_args.device)

    # usage
    '''
    action = agent.act(agent_obs, sample=sac_samples_action, batched=True)
    '''

    return agent

def get_pets_models(pets_path, args):
    env, env_args, termination_fn, reward_fn = get_env(findname(pets_path), None, is_test=True, pets=True)

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    device = "cuda:0"
    rng = np.random.default_rng(seed=args.seed)
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(args.seed)

    cfg = omegaconf.OmegaConf.load('%s/cfg.yaml'%(os.path.dirname(pets_path)))
    cfg.algorithm.agent.planning_horizon = 3
    cfg.algorithm.num_particles = 3
    cfg.device = device

    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    dynamics_model.load(os.path.dirname(pets_path), gpus=cfg.device)

    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )
    return agent