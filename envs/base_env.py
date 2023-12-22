from abc import (
    ABC,
    abstractmethod,
    abstractproperty,
)
from gym import Env
import numpy as np
import torch
from utils import to_np, to_torch
import csv

class BaseEnv(Env):
    def __init__(self, args):
        super(BaseEnv, self).__init__()
        self.args = args
        self.pid = None
        self.sample_idx = 0
        # TODO obs space and action space
        self.reward_list = []
        self.stl_reward_list = []
        self.acc_reward_list = []
        self.history = []
        if hasattr(args, "write_csv") and args.write_csv:
            self.epi = 0
            self.csvfile = open('%s/monitor_full.csv'%(args.exp_dir_full), 'w', newline='')
            self.csvwriter = csv.writer(self.csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        self.reward_fn = self.generate_reward_batch_fn()
        self.reward_fn_torch = self.wrap_reward_fn_torch(self.reward_fn)
    
    @abstractmethod
    def next_state(self, x, u):
        pass

    # @abstractmethod
    def dynamics(self, x0, u, include_first=False):
        args = self.args
        t = u.shape[1]
        x = x0.clone()
        segs = []
        if include_first:
            segs.append(x)
        for ti in range(t):
            new_x = self.next_state(x, u[:, ti])
            segs.append(new_x)
            x = new_x
        return torch.stack(segs, dim=1)

    @abstractmethod
    def init_x_cycle(self):
        pass
    
    @abstractmethod
    def init_x(self):
        pass
    
    @abstractmethod
    def generate_stl(self):
        pass 
    
    @abstractmethod
    def generate_heur_loss(self):
        pass
    
    @abstractmethod
    def visualize(self):
        pass

    def transform(self, seg):
        # this is used for some case when there is a need to first augment the state trajectory
        # for example, for the panda env environment
        return seg

    #@abstractmethod
    def step(self):
        pass
    
    def write_to_csv(self, env_steps):
        r_rs = self.get_rewards()
        r_rs = np.array(r_rs, dtype=np.float32)
        r_avg = np.mean(r_rs[0])
        rs_avg = np.mean(r_rs[1])
        racc_avg = np.mean(r_rs[2])
        self.csvwriter.writerow([self.epi, env_steps, r_avg, rs_avg, racc_avg])
        self.csvfile.flush()
        print("epi:%06d step:%06d r:%.3f %.3f %.3f"%(self.epi, env_steps, r_avg, rs_avg, racc_avg))
        self.epi += 1

    #@abstractmethod
    # def reset(self):
        # pass
    def reset(self):
        N = self.args.num_samples
        if self.sample_idx % N == 0:
            self.x0 = self.init_x(N)
            self.indices = torch.randperm(N)
        self.state = to_np(self.x0[self.indices[self.sample_idx % N]])
        self.sample_idx += 1
        self.t = 0
        if len(self.history)>self.args.nt:
            segs_np = np.stack(self.history, axis=0)
            segs = to_torch(segs_np[None, :])
            seg_aug = self.transform(segs)
            seg_aug_np = to_np(seg_aug)
            # print(seg_aug_np.shape)
            # exit()
            self.reward_list.append(np.sum(self.generate_reward_batch(seg_aug_np.squeeze())))
            self.stl_reward_list.append(self.stl_reward(seg_aug)[0, 0])
            self.acc_reward_list.append(self.acc_reward(seg_aug)[0, 0])
        self.history = [np.array(self.state)]
        return self.state
    
    def get_rewards(self):
        if len(self.reward_list)==0:
            return 0, 0, 0
        else:
            return self.reward_list[-1], self.stl_reward_list[-1], self.acc_reward_list[-1]

    def generate_reward_batch(self, state): # (n, 7)
        return self.reward_fn(None, state)

    def wrap_reward_fn_torch(self, reward_fn):
        def reward_fn_torch(act, state):
            act_np = act.detach().cpu().numpy()
            state_np = state.detach().cpu().numpy()
            reward_np = reward_fn(act_np, state_np)
            return torch.from_numpy(reward_np).float()[:, None].to(state.device)
        return reward_fn_torch

    @abstractmethod
    def generate_reward_batch_fn(self):
        pass

    #@abstractmethod
    def generate_reward(self, state):
        if self.args.stl_reward or self.args.acc_reward:
            last_one = (self.t+1) >= self.args.nt
            if last_one:
                segs = to_torch(np.stack(self.history, axis=0)[None, :])
                segs_aug = self.transform(segs)
                if self.args.stl_reward:
                    return self.stl_reward(segs_aug)[0, 0]
                elif self.args.acc_reward:
                    return self.acc_reward(segs_aug)[0, 0]
                else:
                    raise NotImplementError
            else:
                return np.zeros_like(0)
        else:
            return self.generate_reward_batch(state[None, :])[0]

    def stl_reward(self, segs):
        score = self.stl(segs, self.args.smoothing_factor)[:, :1]
        reward = to_np(score)
        return reward
    
    def acc_reward(self, segs):
        score = (self.stl(segs, self.args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        reward = 100 * to_np(score)
        return reward

    def print_stl(self):
        print(self.stl)
        self.stl.update_format("word")
        print(self.stl)

    def my_render(self):
        if self.pid==0:
            self.render(None)
    
    def test(self):
        for trial_i in range(self.num_trials):
            obs = self.test_reset()
            trajs = [self.test_state()]
            for ti in range(self.nt):
                u = solve(obs)
                obs, reward, done, di = self.test_step(u)
                trajs.append(self.test_state())
        
        # save metrics result