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
import os

def panda_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.any(torch.abs(next_obs[:, :]) > 100, dim=-1)
    done = done[:, None]
    return done

DOF = 7
goal_r = 0.1
obs_x, obs_y, obs_z = 0.3, 0.3, 0.5
obs_r = 0.2
thmin = np.array([-166,-101,-166,-176,-166,-1,-166]) / 180 * np.pi
thmax = np.array([166,101,166,-4,166,215,166]) / 180 * np.pi

class PandaEnv(base_env.BaseEnv):
    def __init__(self, args):
        super(PandaEnv, self).__init__(args)
        self.action_space = spaces.Box(np.array([-1.0]*DOF, dtype=np.float32), np.array([1.0]*DOF, dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-np.pi*2] * DOF + [-1.5] * 3, dtype=np.float32), 
            np.array([np.pi*2] * DOF + [1.5] * 3, dtype=np.float32), dtype=np.float32)
        self.generate_stl()

        import pytorch_kinematics as pk
        if os.path.isfile("model_description/panda.urdf")==False:
            self.chain = pk.build_serial_chain_from_urdf(open("../../model_description/panda.urdf").read(), "panda_link8")
            self.chain_cpu = pk.build_serial_chain_from_urdf(open("../../model_description/panda.urdf").read(), "panda_link8")
        else:
            self.chain = pk.build_serial_chain_from_urdf(open("model_description/panda.urdf").read(), "panda_link8")
            self.chain_cpu = pk.build_serial_chain_from_urdf(open("model_description/panda.urdf").read(), "panda_link8")
        self.chain.to(device="cuda")
    
    def next_state(self, x, u):
        new_x = torch.zeros_like(x)
        for i in range(DOF):
            new_x[:, i] = x[:, i] + u[:, i] * self.args.dt
        new_x[:, DOF:DOF+3] = x[:, DOF:DOF+3]
        return new_x

    def init_x(self, N):
        return self.initialize_x(N)

    def initialize_x(self, N):
        total_sum = 0
        n_try = 0
        x_list = []
        while total_sum <= N:
            n_try+=1
            x = self.initialize_x_cycle(N)
            xyz = self.endpoint(x)
            collide0 = self.check_collision(xyz, obs_r)
            collide1 = self.check_collision(x[..., DOF:DOF+3], obs_r)
            safe_idx = torch.where(torch.logical_and(collide0==False, collide1==False))
            x = x[safe_idx]
            total_sum += x.shape[0]
            x_list.append(x)
        x = torch.cat(x_list, dim=0)[:N]
        return x
    
    def initialize_x_cycle(self, N):
        n = N
        q = uniform_tensor(0, 1, (n, 7))
        q_min = torch.from_numpy(thmin).float() 
        q_max = torch.from_numpy(thmax).float() 
        q = q * (q_max-q_min) + q_min
        
        xx = uniform_tensor(-0.5, 0.5, (n, 1))
        yy = uniform_tensor(-0.5, 0.5, (n, 1))
        zz = uniform_tensor(0.2, 0.8, (n, 1))

        x_init = torch.cat([q, xx,yy,zz], dim=-1).cuda()
        return x_init
    
    def endpoint(self, s, all_points=False):
        shape = s.shape
        s_2d = s.reshape(-1, s.shape[-1])
        M = self.chain.forward_kinematics(s_2d[..., :DOF], end_only=not all_points)
        if all_points:
            res = torch.stack([M[mk].get_matrix()[..., :3, 3] for mk in M], dim=0)
            return res.reshape([len(M),]+list(s.shape[:-1]) + [3,] )
        else:  # end_only=True
            res = M.get_matrix()[..., :3, 3]
            return res.reshape(list(s.shape[:-1]) + [3,] )
    
    def endpoint_cpu(self, s, all_points=False):
        shape = s.shape
        s_2d = s.reshape(-1, s.shape[-1])
        M = self.chain_cpu.forward_kinematics(s_2d[..., :DOF], end_only=not all_points)
        if all_points:
            res = torch.stack([M[mk].get_matrix()[..., :3, 3] for mk in M], dim=0)
            return res.reshape([len(M),]+list(s.shape[:-1]) + [3,] )
        else:  # end_only=True
            res = M.get_matrix()[..., :3, 3]
            return res.reshape(list(s.shape[:-1]) + [3,] )
        
    def check_collision(self, xyz, r):
        collide = r**2 >= (xyz[..., 0]-obs_x)**2+(xyz[..., 1]-obs_y)**2+(xyz[..., 2]-obs_z)**2
        return collide

    def dist_to_goal_vec(self, xyz0, xyz1):
        dist = torch.norm(xyz0-xyz1, dim=-1)
        return dist

    def generate_stl(self):
        args = self.args
        obs_vec = torch.tensor([obs_x, obs_y, obs_z]).cuda()
        reach_goal = Eventually(0, args.nt, Always(0, args.nt, AP(lambda x_aug: goal_r - self.dist_to_goal_vec(x_aug[..., DOF+3:DOF+6], x_aug[..., DOF:DOF+3]), comment="REACH1")))
        avoid = Always(0, args.nt, AP(lambda x_aug: - obs_r + self.dist_to_goal_vec(x_aug[..., DOF+3:DOF+6], obs_vec), comment="AVOID"))
        angle_cons_fn=lambda i: Always(0, args.nt, AP(lambda x: (thmax[i]-x[..., i])*(x[...,i]-thmin[i]), comment="q%d"%(i)))
        angle_cons_list=[]
        for i in range(DOF):
            angle_cons_list.append(angle_cons_fn(i))
        stl = ListAnd([reach_goal, avoid] + angle_cons_list)
        self.stl = stl
        return stl
    
    def transform(self, seg):
        seg_ee = self.endpoint(seg)
        seg_aug = torch.cat([seg, seg_ee], dim=-1)
        return seg_aug

    def generate_heur_loss(self, acc, seg):
        seg_ee = self.endpoint(seg)
        dist = self.dist_to_goal_vec(seg_ee, seg[..., DOF:DOF+3])
        return torch.mean(dist) *1

    def visualize(self, x_init, seg, acc, epi):
        args = self.args
        seg_np = to_np(seg)
        acc_np = to_np(acc)
        t_len = seg_np.shape[1]
        N = args.num_samples

        plt.figure(figsize=(12, 12))

        seg_ee = self.endpoint(seg, all_points=False)
        seg_aug = torch.cat([seg, seg_ee], dim=-1)
        seg_aug_np = to_np(seg_aug)
        edpt = seg_aug_np[..., DOF+3:DOF+6]
        nx = 5
        ny = 5

        for i in range(nx):
            for j in range(ny):
                idx=i*nx+j
                plt.subplot(nx, ny, idx+1)
                ax = plt.gca()
                circ = Circle([seg_aug_np[idx, 0, DOF], seg_aug_np[idx, 0, DOF+1]], radius=goal_r, color="green", alpha=0.5)
                ax.add_patch(circ)
                circ = Circle([obs_x, obs_y], radius=obs_r, color="gray", alpha=0.5)
                ax.add_patch(circ)

                for ti in range(args.nt):
                    plt.plot(edpt[idx,ti:ti+2,0], edpt[idx,ti:ti+2,1], color="blue", alpha=0.2+0.8*ti/args.nt)
                plt.scatter(edpt[idx,0,0], edpt[idx,0,1], color="blue", marker="v")
                plt.scatter(edpt[idx,-1,0], edpt[idx,-1,1], color="blue")

                rect3 = Rectangle([-2, -2], 4, 4, color="green" if acc_np[idx]>0.5 else "red", zorder=100, alpha=0.05)
                ax.add_patch(rect3)

                plt.axis("scaled")
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)

        figname="%s/iter_%05d.png"%(args.viz_dir, epi)
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def step(self, action):
        args = self.args
        action = np.clip(action * args.u_max, -args.u_max, args.u_max)
        action = torch.from_numpy(action).unsqueeze(0)
        state = torch.from_numpy(self.state).unsqueeze(0)
        next_state = self.next_state(state, action)
        self.state = to_np(next_state)[0]
        self.history.append(np.array(self.state))

        seg_aug = self.transform(next_state[0].cuda())
        seg_aug_np = to_np(seg_aug)

        reward = self.generate_reward(seg_aug_np)
        self.t += 1
        terminated = self.t >= args.nt
        return self.state, reward, terminated, {}
    
    def generate_reward_batch_fn(self):
        args = self.args
        def reward_fn(act, state):
            reward = np.zeros(state.shape[:-1])
            collision_penalty = -100
            large_angle_penalty = -100
            reach_reward = 100
            if state.shape[-1]==DOF+3:
                ee = self.endpoint_cpu(state) #to_np(self.endpoint(torch.from_numpy(state[..., :DOF]).cuda()))
            else:
                ee = state[..., DOF+3:DOF+6] #to_np(self.endpoint(torch.from_numpy(state[..., :DOF]).cuda()))
            is_collide = self.check_collision(ee, obs_r)

            angle_list=[]
            for i in range(DOF):
                is_large_angle = np.logical_or(state[..., i]<thmin[i], state[..., i]>thmax[i])
                angle_list.append(is_large_angle)
            angle_list = np.stack(angle_list, axis=-1)
            angle_list = np.any(is_large_angle, axis=-1)
            is_reach = np.linalg.norm(ee - state[..., DOF:DOF+3], axis=-1) < goal_r
            reward[is_collide] += collision_penalty
            reward[is_large_angle] += large_angle_penalty
            reward[is_reach] += reach_reward
            return reward
        return reward_fn
    
    def render(self, mode):
        args = self.args
        state = self.state[None, :]
        ti = self.t
        nt = self.args.nt
        plt.figure(figsize=(8, 8))
        ax = plt.gca()

        ee = self.endpoint(torch.from_numpy(state[:, :DOF]).cuda())
        ee_history = [self.endpoint(torch.from_numpy(xx[None, :DOF]).cuda()).detach().cpu().numpy() for xx in self.history]
        ax.plot([xx[0, 0] for xx in ee_history], [xx[0, 1] for xx in ee_history], color="green", linewidth=4, alpha=0.5, zorder=10, label="Trajectory")

        circ = Circle([state[0, DOF], state[0, DOF+1]], radius=goal_r, color="green", alpha=0.5)
        ax.add_patch(circ)
        circ = Circle([obs_x, obs_y], radius=obs_r, color="gray", alpha=0.5)
        ax.add_patch(circ)

        plt.scatter(to_np(ee[0, 0]), to_np(ee[0, 1]), color="blue", marker="v")
        
        ax.axis("scaled")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend(loc="upper right")

        plt.title("Simulation (%04d/%04d)"%(ti, nt))
        plt.savefig("%s/rl_sim_%05d_%02d.png"%(args.viz_dir, self.sample_idx, self.t), bbox_inches='tight', pad_inches=0.1)
        plt.close()