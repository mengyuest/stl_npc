from lib_stl_core import *
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from matplotlib.collections import PatchCollection
import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, to_torch, build_relu_nn, build_relu_nn1
from envs.base_env import BaseEnv
from envs.car_env import CarEnv
from envs.maze_env import MazeEnv
from envs.ship_env import ShipEnv
from envs.rover_env import RoverEnv
from envs.panda_env import PandaEnv


plt.rcParams.update({'font.size': 12})

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv

import csv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0, args=None, eta=None):
        super(CustomCallback, self).__init__(verbose)
        self.args = args
        self.eta = eta
        self.csvfile = open('%s/monitor_full.csv'%(args.exp_dir_full), 'w', newline='')
        self.csvwriter = csv.writer(self.csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
   
    def _on_step(self):
        args = self.args
        epi = (self.n_calls-1) // args.nt
        eta = self.eta
        triggered = (self.n_calls-1) % args.nt == 0
        if triggered:
            eta.update()
            r_rs = self.model.env.env_method("get_rewards")
            r_rs = np.array(r_rs, dtype=np.float32)
            r_avg = np.mean(r_rs[:, 0])
            rs_avg = np.mean(r_rs[:, 1])
            racc_avg = np.mean(r_rs[:, 2])
            self.csvwriter.writerow([epi, r_avg, rs_avg, racc_avg, eta.elapsed()])

        if triggered and epi % args.print_freq == 0:
            x, y = ts2xy(load_results(args.exp_dir_full), "timesteps")
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])
            else:
                mean_reward = 0.0
            print("%s RL epi:%07d reward:%.2f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1],
                epi, mean_reward, eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                ))
        if triggered:
            if epi % 100 == 0:
                self.model.save("%s/model_last"%(args.model_dir))
            if epi % ((args.epochs // args.num_workers)//5) == 0:
                self.model.save("%s/model_%05d"%(args.model_dir, epi))
        return True


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        T = args.nt
        IO_DIMS = {"car":[7, 1*T], "maze":[9, 1*T], "ship1":[12, 2*T], "ship2":[10, 2*T], "rover":[8, 2*T], "panda":[10, 7*T]}
        self.net = build_relu_nn1(IO_DIMS[args.mode], args.hiddens, activation_fn=nn.ReLU)
    
    def clip_u(self, x, amin, amax):
        if self.args.no_tanh:
            return torch.clip(x, amin, amax)
        else:
            return torch.tanh(x) * (amax - amin) / 2 + (amax  + amin) / 2

    def forward(self, x):  
        args = self.args   
        N = x.shape[0]
        T = args.nt
        u = self.net(x).reshape(N, T, -1)
        if self.args.mode == "car":
            u = u[..., 0]
            uu = self.clip_u(u, -10.0, 10.0)
        elif self.args.mode == "maze":
            u = u[..., 0]
            uu = self.clip_u(u, -40.0, 40.0)
        elif self.args.mode in ["ship1", "ship2"]: 
            u0 = self.clip_u(u[..., 0], -args.thrust_max, args.thrust_max)
            u1 = self.clip_u(u[..., 1], -args.delta_max, args.delta_max)
            uu = torch.stack([u0, u1], dim=-1)
        elif self.args.mode == "rover":
            u0 = self.clip_u(u[..., 0], 0, 1)
            u1 = self.clip_u(u[..., 1], -np.pi, np.pi)
            uu = torch.stack([u0, u1], dim=-1)
        elif self.args.mode == "panda":
            uu = self.clip_u(u, -args.u_max, args.u_max)
        else:
            raise NotImplementError
        return uu

def run_test(net, env):
    return


def make_env(env_name, args, seed_i, seed, logdir):
    def _f():
        env = env_name(args)
        env.seed(seed)
        env.pid=seed_i
        if seed_i==0:
            return Monitor(env, logdir)
        else:
            return env
    return _f


def main(args):
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq, args.num_workers)
    
    # TODO RL case
    if args.train_rl:
        env_dict = {"car": CarEnv, "maze": MazeEnv, "ship1": ShipEnv, "ship2": ShipEnv, "rover": RoverEnv, "panda": PandaEnv}
        if args.num_workers != None:
            seeds = [args.seed + seed_i for seed_i in range(args.num_workers)]
            envs = [make_env(env_dict[args.mode], args, seed_i, seed, args.exp_dir_full) for seed_i, seed in enumerate(seeds)] 
            env = SubprocVecEnv(envs)
        else:
            env = env_dict[args.mode](args)
            env.seed(args.seed)
            env.pid = 0
            env = Monitor(env, args.exp_dir_full)
        callback = CustomCallback(args=args, eta=eta)

        print("Now train the policy ...")
        model = SAC("MlpPolicy", env, verbose=0, seed=args.seed, policy_kwargs={"net_arch":{"pi":args.hiddens, "qf":[256, 256]}})
        model.learn(total_timesteps=args.epochs*args.nt, callback=callback) #()

        print("Now evaluate ...")
        vec_env = model.get_env()
        obs = vec_env.reset()
        for i in range(1000):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.env_method(method_name='my_render')
        return

    # model setup
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    
    # optimizer setup
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # env setup
    env_dict = {"car": CarEnv, "maze": MazeEnv, "ship1": ShipEnv, "ship2": ShipEnv, "rover": RoverEnv, "panda": PandaEnv}
    env = env_dict[args.mode](args)
    
    stl = env.generate_stl()
    if args.test:
        env.test(net, env)
    else:
        csvfile = open('%s/monitor_full.csv'%(args.exp_dir_full), 'w', newline='')
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        csvfile_val = open('%s/monitor_full_val.csv'%(args.exp_dir_full), 'w', newline='')
        csvwriter_val = csv.writer(csvfile_val, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        x_init = env.init_x(args.num_samples).float().cuda()
        x_init_val = env.init_x(5000).float().cuda()
        env.print_stl()

        for epi in range(args.epochs):
            eta.update()
            if args.update_init_freq >0 and epi % args.update_init_freq == 0 and epi!=0:
                x_init = env.init_x(args.num_samples).float().cuda()
            x0 = x_init.detach()
            u = net(x0)
            seg = env.dynamics(x0, u, include_first=True)

            seg_aug = env.transform(seg)

            score = stl(seg_aug, args.smoothing_factor)[:, :1]
            score_avg = torch.mean(score)
            acc = (stl(seg_aug, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
            acc_avg = torch.mean(acc)
            _n, _t, _k = seg.shape
            all_states = to_np(seg.reshape(_n*_t, -1))
            reward = np.mean(env.generate_reward_batch(all_states)) * _t
            acc_reward = (acc_avg * 100).item()
            stl_reward = (score_avg).item()
            dist_loss = env.generate_heur_loss(acc, seg_aug)

            loss = torch.mean(nn.ReLU()(args.c_val-score)) + dist_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            csvwriter.writerow([epi, reward, stl_reward, acc_reward, eta.elapsed()])
            csvfile.flush()
            
            if epi % args.print_freq == 0:
                u_val = net(x_init_val.detach())        
                seg_val = env.dynamics(x_init_val, u_val, include_first=True)
                seg_val_aug = env.transform(seg_val)
                score_val = stl(seg_val_aug, args.smoothing_factor)[:, :1]
                score_avg_val = torch.mean(score_val)

                acc_val = (stl(seg_val_aug, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_avg_val = torch.mean(acc_val)

                all_states_val = to_np(seg_val.reshape(seg_val.shape[0] *_t, -1))
                reward_val = np.mean(env.generate_reward_batch(all_states_val)) * _t
                acc_reward_val = (acc_avg_val * 100).item()
                stl_reward_val = (score_avg_val).item()

                csvwriter_val.writerow([epi, reward_val, stl_reward_val, acc_reward_val, eta.elapsed()])
                csvfile_val.flush()

                print("%s|%03d  loss:%.3f acc:%.3f dist:%.3f acc_val:%.3f R:%.2f R':%.2f R'':%.2f dT:%s T:%s ETA:%s" % (
                    args.exp_dir_full.split("/")[-1], epi, loss.item(), acc_avg.item(),
                    dist_loss.item(), acc_avg_val.item(), reward, stl_reward, acc_reward, eta.interval_str(), eta.elapsed_str(), eta.eta_str()))
            
            # Save models
            if epi % args.save_freq == 0:
                torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))

            if epi == args.epochs-1 or epi % 100 == 0:
                torch.save(net.state_dict(), "%s/model_last.ckpt"%(args.model_dir))

            if epi % args.viz_freq == 0 or epi == args.epochs - 1:
                env.visualize(x_init, seg, acc, epi)