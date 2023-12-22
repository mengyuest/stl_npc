import numpy as np
from gym import Env, logger, spaces

import envs.base_env as base_env
from lib_stl_core import *
import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, \
            check_pts_collision, check_seg_collision, soft_step, to_torch, pts_in_poly, seg_int_poly, build_relu_nn, soft_step_hard

from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.size': 12})

def maze_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.any(torch.abs(next_obs[:, :]) > 100, dim=-1)
    done = done[:, None]
    return done

class MazeEnv(base_env.BaseEnv):
    def __init__(self, args):
        super(MazeEnv, self).__init__(args)
        self.action_space = spaces.Box(np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([0, args.v_min, -args.y_level,  0, args.obs_w_min, -1, 0, args.obs_w_min, -1], dtype=np.float32), 
            np.array([args.canvas_w, args.v_max, 0,  args.canvas_w, args.obs_w_max, args.canvas_w, args.canvas_w, args.obs_w_max, args.canvas_w], dtype=np.float32), dtype=np.float32)
        self.generate_stl()
        
    def next_state(self, x, u):
        args = self.args
        new_x = torch.zeros_like(x)
        new_x[:, 0] = torch.clip(x[:, 0] + x[:, 1] * args.dt, 0.1, args.canvas_w-0.1)
        new_x[:, 1] = torch.clip(x[:, 1] + u * args.dt, args.v_min, args.v_max)
        new_x[:, 2] = x[:, 2] + args.vy * args.dt
        new_x[:, 3:] = x[:, 3:]
        return new_x

    def init_x_cycle(self, N):
        args = self.args
        n = args.num_samples
        x = uniform_tensor(0, args.canvas_w, (n, 1))
        vx = uniform_tensor(args.v_min, args.v_max, (n, 1))
        dy = uniform_tensor(-args.y_level, 0, (n, 1))
        dx0, L0, G0 = self.generate_seg(n)
        dx1, L1, G1 = self.generate_seg(n)  

        # make sure feasible
        # is_G0 = (G0>=0).float()
        lamda = uniform_tensor(0, 1, (n, 1)) 
        remain_y = -dy - args.obs_h
        remain_n = (remain_y / args.vy // args.dt) - 1
        left_most = - vx * args.dt + remain_n * args.v_min * args.dt 
        right_most = - vx * args.dt + remain_n * args.v_max * args.dt
        poss_0 = torch.clip(0 + left_most, 0, args.canvas_w)
        poss_1 = torch.clip(dx0 + right_most, 0, args.canvas_w)
        poss_2 = torch.clip(dx0 + L0 + left_most, 0, args.canvas_w)
        poss_3 = torch.clip(args.canvas_w + right_most, 0, args.canvas_w)
        
        poss_pass_region_01 = poss_0 * lamda + poss_1 * (1-lamda)
        poss_pass_region_23 = poss_2 * lamda + poss_3 * (1-lamda)
        poss_pass_idx = rand_choice_tensor([0, 1], (n, 1))
        poss_pass_idx[torch.where(torch.logical_and(poss_0==poss_1, poss_0==0))[0]] = 1
        poss_pass_idx[torch.where(dx0==0)]=1
        poss_pass_idx[torch.where(torch.logical_and(poss_2==poss_3, poss_2==args.canvas_w))[0]] = 0
        poss_pass_idx[torch.where(dx0==args.canvas_w-L0)]=0
        poss_pass_region = poss_pass_region_01 * (poss_pass_idx==0).float() + poss_pass_region_23 * (poss_pass_idx==1).float()
    
        poss_0_strict = torch.clip(G0 - args.goal_w/2 + left_most, 0, args.canvas_w)
        poss_1_strict = torch.clip(G0 + args.goal_w/2 + right_most, 0, args.canvas_w)
        poss_goal_region = lamda * poss_0_strict + (1-lamda) * poss_1_strict

        poss_x = poss_pass_region *(G0<0).float() + poss_goal_region * (G0>0).float()
        x = poss_x
        x_init = torch.cat([x, vx, dy, dx0, L0, G0, dx1, L1, G1], dim=-1).cuda()
        return x_init

    def init_x(self, N):
        args = self.args
        total_sum = 0
        n_try = 0
        x_list = []
        while total_sum <= args.num_samples:
            n_try+=1
            x = self.init_x_cycle(N)
            collide = torch.logical_and(x[..., 2]>=-args.obs_h, torch.logical_and(x[..., 0]>=x[...,3], x[..., 0]<=x[..., 3]+x[...,4]))
            collide_next = torch.logical_and(x[..., 2]+args.vy*args.dt>=-args.obs_h, 
                                            torch.logical_and(x[..., 0]+x[...,1]*args.dt>=x[...,3], x[..., 0]+x[...,1]*args.dt<=x[..., 3]+x[...,4]))
            safe_idx = torch.where(torch.logical_and(collide==False, collide_next==False))[0]
            x = x[safe_idx]
            total_sum += x.shape[0]
            x_list.append(x)
        x = torch.cat(x_list, dim=0)[:args.num_samples]
        return x
    
    def generate_seg(self, n):
        args = self.args
        L0 = uniform_tensor(args.obs_w_min, args.obs_w_max, (n, 1))
        dx0 = args.canvas_w * 0.5 + uniform_tensor(-1, 1, (n, 1)) * (args.canvas_w - L0) * 0.5 * args.obs_ratio
        dx0 = dx0 - L0 * 0.5  # find the left-most point
        magnetic = 0.1
        dx0[torch.where(dx0<magnetic)[0]] = 0
        dx0[torch.where(args.canvas_w-L0-dx0<magnetic)[0]] = (args.canvas_w-L0)[torch.where(args.canvas_w-L0-dx0<magnetic)[0]]
        G0_0 = dx0 * uniform_tensor(0.3, 0.7, (n, 1))
        G0_0[torch.where(torch.logical_or(G0_0<args.goal_w/2, dx0-G0_0<args.goal_w/2))[0]] = -1
        G0_1 = (args.canvas_w - L0 - dx0) * uniform_tensor(0.3, 0.7, (n, 1)) + dx0 + L0
        G0_1[torch.where(torch.logical_or(args.canvas_w-G0_1<args.goal_w/2, G0_1-L0-dx0<args.goal_w/2))[0]] = -1
        G0_2 = uniform_tensor(-1, -1, (n, 1))
        m0 = rand_choice_tensor([0, 1, 2], (n, 1))
        G0 = (m0==0).float()*G0_0 + (m0==1).float()*G0_1 + (m0==2).float()*G0_2
        return dx0, L0, G0

    def generate_stl(self):
        args = self.args
        avoid0 = Always(0, args.nt, Imply(
            AP(lambda x: -(x[..., 2]+args.obs_h)*(x[..., 2]), comment="In0"), 
            AP(lambda x: (x[..., 3]+x[..., 4]-x[...,0]) *(x[..., 3]-x[...,0]), comment="Avoid0")))
    
        avoid1 = Always(0, args.nt, Imply(
            AP(lambda x: -(x[..., 2]+args.obs_h-args.y_level)*(x[..., 2]-args.y_level), comment="In1"), 
            AP(lambda x: (x[..., 3+3]+x[..., 4+3]-x[...,0]) *(x[..., 3+3]-x[...,0]), comment="Avoid1")))

        r = args.goal_w/2
        reach0 = Imply(AP(lambda x:x[..., 5], comment="IsG0"),
            Eventually(0, args.nt, AP(lambda x: r**2-(x[...,0]-x[...,5])**2-(x[...,2]+args.obs_h/2)**2, comment="GOAL0")))
        
        reach1 = Imply(AP(lambda x:x[..., 5+3], comment="IsG1"),
            Eventually(0, args.nt, AP(lambda x: r**2-(x[...,0]-x[...,5+3])**2-(x[...,2]-args.y_level+args.obs_h/2)**2, comment="GOAL1")))
        
        stl = ListAnd([avoid0, avoid1, reach0, reach1])
        self.stl = stl
        return stl
    
    def generate_heur_loss(self, acc, seg):
        return torch.mean(seg) * 0.0
    
    def visualize(self, x_init, seg, acc, epi):
        args = self.args
        seg_np = to_np(seg)
        acc_np = to_np(acc)
        t_len = seg_np.shape[1]
        N = args.num_samples
        
        plt.figure(figsize=(12, 12))
        col = 6
        row = 6
        for i in range(row):
            for j in range(col):
                idx = i * col + j
                ax = plt.subplot(row, col, idx+1)

                # plot obstacles
                rect0 = Rectangle([seg_np[idx, 0, 3], -args.obs_h], seg_np[idx, 0, 4], args.obs_h, color="brown", zorder=5)
                ax.add_patch(rect0)

                rect1 = Rectangle([seg_np[idx, 0, 3+3], args.y_level-args.obs_h], seg_np[idx, 0, 4+3], args.obs_h, color="brown", zorder=5)
                ax.add_patch(rect1)

                # plot goal
                r = args.goal_w / 2
                if seg_np[idx, 0, 5]>=0:
                    ell0 = Ellipse([seg_np[idx, 0, 5], -args.obs_h/2], 2*r, 2*r, color="orange", zorder=8)
                    ax.add_patch(ell0)
                
                if seg_np[idx, 0, 5+3]>=0:
                    ell1 = Ellipse([seg_np[idx, 0, 5+3], args.y_level-args.obs_h/2], 2*r, 2*r, color="orange", zorder=8)
                    ax.add_patch(ell1)

                # plot trajectories
                ymin = -args.dy_max * 1
                ymax = args.dy_max * 2
                rect3 = Rectangle([0, ymin], args.canvas_w, ymax-ymin, color="green" if acc_np[idx]>0.5 else "red", zorder=100, alpha=0.05)
                ax.add_patch(rect3)

                ax.axvline(x=0, ymin=0, ymax=ymax, linestyle="--", color="gray")
                ax.axvline(x=args.canvas_w, ymin=0, ymax=ymax, linestyle="--", color="gray")
                ax.plot(seg_np[idx,:,0], seg_np[idx,:,2], color="green" if acc_np[idx]>0.5 else "red", linewidth=2, alpha=0.5, zorder=10)
                ax.axis("scaled")
                ax.set_xlim(0, args.canvas_w)
                ax.set_ylim(ymin, ymax)
    
        figname="%s/iter_%05d.png"%(args.viz_dir, epi)
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def step(self, action):
        args = self.args
        amax = 40
        action = np.clip(action * amax, -amax, amax)
        action = torch.from_numpy(action).unsqueeze(0)
        state = torch.from_numpy(self.state).unsqueeze(0)
        next_state = self.next_state(state, action)
        self.state = to_np(next_state)[0]
        self.history.append(np.array(self.state))
        reward = self.generate_reward(self.state)
        self.t += 1
        terminated = self.t >= args.nt
        return self.state, reward, terminated, {}

    def generate_reward_batch_fn(self):
        # collide-0
        # collide-1
        # reach-0
        # reach-1
        args = self.args
        def reward_fn(act, state):
            reward = np.zeros((state.shape[0],))
            collision_penalty = -100
            reach_reward = 100

            is_collide0 = np.logical_and(
                -(state[:, 2] + args.obs_h)*(state[:, 2]) > 0, 
                -(state[:, 3] + state[:, 4] - state[:, 0])*(state[:, 3]-state[:, 0]) >0
            )

            is_collide1 = np.logical_and(
                -(state[:, 2] + args.obs_h - args.y_level) * (state[:, 2] - args.y_level) > 0,
                -(state[:, 6] + state[:, 7] - state[:, 0]) * (state[:, 6] - state[:, 0]) > 0
            )

            is_collide2 = np.logical_or(state[:, 0]<0, state[:,0]>args.canvas_w)

            r = args.goal_w / 2

            is_reach0 = (state[:,0]-state[:,5])**2 + (state[:,2]+args.obs_h/2)**2 < r ** 2
            is_reach1 = (state[:,0]-state[:,8])**2 + (state[:,2]-args.y_level+args.obs_h/2)**2 < r ** 2

            reward[is_collide0] += collision_penalty
            reward[is_collide1] += collision_penalty
            reward[is_collide2] += collision_penalty
            reward[is_reach0] += reach_reward
            reward[is_reach1] += reach_reward

            return reward
        return reward_fn
    
    def render(self, mode):
        args = self.args
        state = self.state
        ti = self.t
        nt = self.args.nt
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        first_time = True

        # plot obstacles
        rect0 = Rectangle([state[3], -args.obs_h], state[4], args.obs_h, color="brown", label="Obstacle", zorder=5)
        ax.add_patch(rect0)
        rect1 = Rectangle([state[3+3], args.y_level-args.obs_h], state[4+3], args.obs_h, color="brown", zorder=5)
        ax.add_patch(rect1)
        r = args.goal_w / 2
        # plot goals
        if state[5] >= 0:
            ell0 = Ellipse([state[5], args.y_level-args.obs_h/2], 2*r, 2*r, color="orange", label="Goal", zorder=8)
            ax.add_patch(ell0)
        if state[5+3] >= 0:
            ell1 = Ellipse([state[5+3], args.y_level-args.obs_h/2], 2*r, 2*r, color="orange", zorder=8)
            ax.add_patch(ell1)

        # plot trajectory
        ymin = state[2] - args.y_level
        ymax = state[2] + 3 * args.y_level
        ax.axvline(x=0, ymin=ymin, ymax=ymax, linestyle="--", color="gray")
        ax.axvline(x=args.canvas_w, ymin=ymin, ymax=ymax, linestyle="--", color="gray", label="Boundary")
        ax.plot([xx[0] for xx in self.history], [xx[2] for xx in self.history], color="green", linewidth=4, alpha=0.5, zorder=10, label="Trajectory")
        ax.axis("scaled")
        ax.set_xlim(0, args.canvas_w)
        ax.set_ylim(ymin, ymax)
        ax.legend(loc="upper right")
        
        plt.title("Simulation (%04d/%04d)"%(ti, nt))
        plt.savefig("%s/rl_sim_%05d_%02d.png"%(args.viz_dir, self.sample_idx, self.t), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    # TODO test function
    def test(self):
        pass