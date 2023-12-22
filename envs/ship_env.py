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


def ship_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.any(torch.abs(next_obs[:, :]) > 100, dim=-1)
    done = done[:, None]
    return done

class ShipEnv(base_env.BaseEnv):
    def __init__(self, args):
        super(ShipEnv, self).__init__(args)
        self.action_space = spaces.Box(np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        if args.mode=="ship1":   # avoid
            self.observation_space = spaces.Box(
                np.array([0, -args.river_width/2, -args.s_phimax, args.s_umin, -args.s_vmax, -args.s_rmax, 0, -args.river_width/2, 1, 0, -args.river_width/2, 1], dtype=np.float32), 
                np.array([args.range_x, args.river_width/2, args.s_phimax, args.s_umax, args.s_vmax, args.s_rmax, args.range_x, args.river_width/2, args.obs_rmax, args.range_x, args.river_width/2, args.obs_rmax], dtype=np.float32), 
            dtype=np.float32)
        else:   # avoid-n-track
            self.observation_space = spaces.Box(
                np.array([0, -args.canvas_h/2, -args.s_phimax, args.s_umin, -args.s_vmax, -args.s_rmax, 
                            -args.range_x, -args.river_width/2, args.obs_rmin, 0.0], dtype=np.float32), 
                np.array([args.canvas_w, args.canvas_h/2, args.s_phimax, args.s_umax, args.s_vmax, args.s_rmax, 
                            args.range_x, args.river_width/2, args.obs_rmax, (args.tmax+1) * args.dt], dtype=np.float32), 
            dtype=np.float32)
        self.generate_stl()
        
    
    def next_state(self, x, u):
        args = self.args
        num = args.num_sim_steps
        uu = u
        for tti in range(num):
            dt = (args.dt/num)
            new_x = torch.zeros_like(x)
            # (x, y, phi, u, v, r)
            new_dx = x[:, 3] * torch.cos(x[:, 2]) - x[:, 4] * torch.sin(x[:, 2])
            new_dy = x[:, 3] * torch.sin(x[:, 2]) + x[:, 4] * torch.cos(x[:, 2])
            new_dphi = x[:, 5]
            new_du = uu[:, 0]
            new_dv = uu[:, 1] * 0.01
            new_dr = uu[:, 1] * 0.5
            zeros = 0 * new_dx

            if args.mode=="ship1":
                dsdt = torch.stack([new_dx, new_dy, new_dphi, new_du, new_dv, new_dr] + [zeros] * (3 *args.n_obs), dim=-1)
            else:
                new_dT = -soft_step(x[:, 1]**2-args.track_thres**2)
                dsdt = torch.stack([new_dx, new_dy, new_dphi, new_du, new_dv, new_dr] + [zeros, zeros, zeros, new_dT], dim=-1)
            new_x = x + dsdt * dt
            new_xx = new_x.clone()
            new_xx[:, 2] = torch.clamp(new_x[:, 2], -args.s_phimax, args.s_phimax)
            new_xx[:, 3] = torch.clamp(new_x[:, 3], args.s_umin, args.s_umax)
            new_xx[:, 4] = torch.clamp(new_x[:, 4], -args.s_vmax, args.s_vmax)
            new_xx[:, 5] = torch.clamp(new_x[:, 5], -args.s_rmax, args.s_rmax)
    
            x = new_xx
        return new_xx
    
    def init_x_cycle(self, N):
        def mux(scene_type, x0, x1, x2, x3):
            return (scene_type==0).float() * x0 + (scene_type==1).float() * x1 + (scene_type==2).float() * x2 + (scene_type==3).float() * x3
        
        args = self.args
        n = N
        
        if args.mode=="ship1":
            
            x = uniform_tensor(0, 0, (n, 1))
            y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
            phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
            u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
            v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
            r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))

            gap = args.range_x / args.n_obs

            obs = []
            for i in range(args.n_obs):
                if i==0:
                    obs_x = uniform_tensor(0, gap, (n, 1))
                else:
                    obs_x = obs_x + gap
                obs_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
                obs_r = uniform_tensor(1, args.obs_rmax, (n, 1))
                obs.append(obs_x)
                obs.append(obs_y)
                obs.append(obs_r)
            obs = torch.cat(obs, dim=-1).reshape(n, args.n_obs, 3)
            obs_xs, _ = torch.sort(obs[:, :, 0:1], dim=1)
            obs = torch.cat([obs_xs, obs[:, :, 1:]], dim=-1).reshape(n, args.n_obs * 3)
            return torch.cat([x, y, phi, u, v, r] + [obs], dim=1)
        else:
            scene_type = rand_choice_tensor([0, 1, 2, 3], (n, 1))
            # without obs case
            s0_x = uniform_tensor(0, 0, (n, 1))
            if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
                s0_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
                s0_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
            else:
                s0_y = uniform_tensor(-0.5, 0.5, (n, 1))
                s0_phi = uniform_tensor(-args.s_phimax/2, args.s_phimax/2, (n, 1))
            s0_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
            s0_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
            s0_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
            s0_obs_x = uniform_tensor(-5, -5, (n, 1))
            s0_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
            s0_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
            if args.origin_sampling:
                s0_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
            else:
                s0_obs_T = rand_choice_tensor([i * args.dt for i in range(1, 10)], (n, 1))
            
            # far from obs case
            s1_x = uniform_tensor(0, 0, (n, 1))
            if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
                s1_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
                s1_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
            else:
                s1_y = uniform_tensor(-0.5, 0.5, (n, 1))
                s1_phi = uniform_tensor(-args.s_phimax/2, args.s_phimax/2, (n, 1))
            s1_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
            s1_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
            s1_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
            s1_obs_x = uniform_tensor(5, args.obs_xmax, (n, 1))
            s1_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
            s1_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
            if args.origin_sampling:
                s1_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
            elif args.origin_sampling3:
                s1_obs_T = rand_choice_tensor([i * args.dt for i in range(10, args.tmax+1)], (n, 1))
            else:
                s1_obs_T = rand_choice_tensor([i * args.dt for i in range(12, args.tmax+1)], (n, 1))

            ymin = 0.8
            ymax = args.river_width/2
            flip = rand_choice_tensor([-1, 1], (n, 1))
            # closer from obs case (before meet)
            s2_x = uniform_tensor(0, 0, (n, 1))
            if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
                s2_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
            else:
                s2_y = uniform_tensor(ymin, ymax, (n, 1)) * flip
            s2_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
            s2_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
            s2_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
            s2_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
            s2_obs_x = uniform_tensor(0, 5, (n, 1))
            s2_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
            s2_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
            if args.origin_sampling:
                s2_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
            elif args.origin_sampling3:
                s2_obs_T = rand_choice_tensor([i * args.dt for i in range(8, 15)], (n, 1))
            else:
                s2_obs_T = rand_choice_tensor([i * args.dt for i in range(10, 15)], (n, 1))

            # closer from obs case (after meet)
            s3_x = uniform_tensor(0, 0, (n, 1))
            if args.origin_sampling or args.origin_sampling2 or args.origin_sampling3:
                s3_y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
            else:
                s3_y = uniform_tensor(ymin, ymax, (n, 1)) * flip
            s3_phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
            s3_u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
            s3_v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
            s3_r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
            s3_obs_x = uniform_tensor(-1, 0, (n, 1))
            s3_obs_y = uniform_tensor(args.obs_ymin, args.obs_ymax, (n, 1))
            s3_obs_r = uniform_tensor(args.obs_rmin, args.obs_rmax, (n, 1))
            if args.origin_sampling:
                s3_obs_T = rand_choice_tensor([i * args.dt for i in range(1, args.tmax+1)], (n, 1))
            elif args.origin_sampling3:
                s3_obs_T = rand_choice_tensor([i * args.dt for i in range(5, 12)], (n, 1))
            else:
                s3_obs_T = rand_choice_tensor([i * args.dt for i in range(8, 12)], (n, 1))

            x = mux(scene_type, s0_x, s1_x, s2_x, s3_x)
            y = mux(scene_type, s0_y, s1_y, s2_y, s3_y)
            phi = mux(scene_type, s0_phi, s1_phi, s2_phi, s3_phi)
            u = mux(scene_type, s0_u, s1_u, s2_u, s3_u)
            v = mux(scene_type, s0_v, s1_v, s2_v, s3_v)
            r = mux(scene_type, s0_r, s1_r, s2_r, s3_r)
            obs_x = mux(scene_type, s0_obs_x, s1_obs_x, s2_obs_x, s3_obs_x)
            obs_y = mux(scene_type, s0_obs_y, s1_obs_y, s2_obs_y, s3_obs_y)
            obs_r = mux(scene_type, s0_obs_r, s1_obs_r, s2_obs_r, s3_obs_r)
            obs_T = mux(scene_type, s0_obs_T, s1_obs_T, s2_obs_T, s3_obs_T)

            rand_zero = rand_choice_tensor([0, 1], (n, 1))
            res = torch.cat([x, y, phi, u, v, r, obs_x, obs_y, obs_r, obs_T], dim=1)
            return res

    def init_x(self, n):
        args = self.args
        x_list = []
        total_n = 0
        while(total_n<n):
            x_init = self.init_x_cycle(n)
            safe_bloat = 1.5 if args.mode=="ship1" else args.bloat_d
            dd = 5
            n_res = 100
            crit_list = []
            if args.mode=="ship1":
                crit1 = torch.norm(x_init[:, :2] - x_init[:, 6:6+2], dim=-1) > x_init[:, 8] + safe_bloat
                crit_list.append(crit1)
                for i in range(n_res):
                    mid_point = torch.stack([
                        x_init[:, 0] + torch.cos(x_init[:, 2]) * dd / n_res * i,
                        x_init[:, 1] + torch.sin(x_init[:, 2]) * dd / n_res * i,
                    ], dim=-1)
                    crit_list.append(torch.norm(mid_point[:, :2] - x_init[:, 6:6+2], dim=-1) > x_init[:, 8])

                crit_list.append(torch.abs(mid_point[:, 1]) < args.river_width/2)
                crit_list = torch.stack(crit_list, dim=-1)
                valids_indices = torch.where(torch.all(crit_list, dim=-1))
            else:
                crit1 = torch.norm(x_init[:, :2] - x_init[:, 6:6+2], dim=-1) > x_init[:, 8] + safe_bloat
                crit2 = torch.logical_not(torch.logical_and(x_init[:,1]>1.5, x_init[:,2]>0))  # too close from the river boundary
                crit3 = torch.logical_not(torch.logical_and(x_init[:,1]<-1.5, x_init[:,2]<0))  # too close from the river boundary
                if args.origin_sampling3:
                    # cannot be too close to the obstacle
                    crit4 = torch.logical_not(torch.logical_and(torch.logical_and(x_init[:,6]-x_init[:,0]<x_init[:,8]+0.5, x_init[:,6]-x_init[:,0]>0), torch.abs(x_init[:,1]-x_init[:,7])<x_init[:,8]))
                    # cannot be too close to the obstacle
                    crit7 = torch.logical_not(torch.logical_and(torch.logical_and(x_init[:,6]-x_init[:,0]<x_init[:,8]+1.5, x_init[:,6]-x_init[:,0]>0), torch.abs(x_init[:,1]-x_init[:,7])<0.3))
                    # should have enough time to escape
                    crit5 = torch.logical_not(torch.logical_and(x_init[:, 9] < 5 * args.dt, torch.abs(x_init[:,1]) > 1.5))
                    # too large angle
                    crit6 = torch.logical_not(torch.logical_or(
                        torch.logical_and(x_init[:, 1] > 1.5, x_init[:, 2] > args.s_phimax/2), 
                        torch.logical_and(x_init[:, 1] < -1.5, x_init[:, 2] < -args.s_phimax/2), 
                    ))
                    valids_indices = torch.where(torch.all(torch.stack([crit1, crit2, crit3, crit4, crit5, crit6, crit7], dim=-1),dim=-1)>0)
                else:
                    valids_indices = torch.where(torch.all(torch.stack([crit1, crit2, crit3], dim=-1),dim=-1)>0)
            x_val = x_init[valids_indices]
            total_n += x_val.shape[0]
            x_list.append(x_val)
        x_list = torch.cat(x_list, dim=0)[:n]
        return x_list
    
    def generate_stl(self):
        args = self.args
        if args.mode=="ship1":
            avoid_func = lambda obs_i: Always(0, args.nt, AP(
                lambda x: torch.norm(x[..., :2] - x[..., 6+3*obs_i:6+3*obs_i+2], dim=-1)**2 - x[..., 6+3*obs_i+2]**2, comment="AVOID OBS%d"%(obs_i)))
            avoid_list = [avoid_func(obs_i) for obs_i in range(args.n_obs)] + [Always(0, args.nt, AP(lambda x: args.river_width/2 - torch.norm(x[..., 1:2], dim=-1), comment="IN MAP"))]
            stl = ListAnd(avoid_list)
        else:
            avoid = Always(0, args.nt, AP(lambda x: torch.norm(x[..., :2] - x[..., 6:6+2], dim=-1)**2 - x[..., 6+2]**2 - args.bloat_d**2, comment="Avoid Obs"))
            in_river = Always(0, args.nt, AP(lambda x: (args.river_width/2)**2-x[..., 1]**2, comment="In River"))
            diverge = AP(lambda x: - args.track_thres**2 + x[..., 1]**2, comment="Diverge")
            in_T = AP(lambda x: x[..., 9], comment="In Time")
            no_obs = AP(lambda x: -x[..., 6]-3, comment="No Obs")
            track = Always(0, args.nt, Not(diverge))
            track_if_no_obs = Imply(no_obs, Eventually(0, args.nt, track))
            back_to_track_if_obs = Imply(Not(no_obs), Until(0, args.nt, in_T, track))
            finally_track = Until(0, args.nt, in_T, track)
            stl = ListAnd([avoid, in_river, finally_track])
        self.stl = stl
        return stl

    def generate_heur_loss(self, acc, seg):
        if self.args.mode=="ship1":
            return torch.mean(seg) * 0.0
        else:
            dist_loss = torch.mean(seg[..., 1]**2)
            return dist_loss * self.args.dist_w
    
    def visualize(self, x_init, seg, acc, epi):
        args = self.args
        seg_np = to_np(seg)
        acc_np = to_np(acc)
        plt.figure(figsize=(12, 9))
        col = 5
        row = 5
        bloat = 0.0
        for i in range(row):
            for j in range(col):
                idx = i * col + j
                ax = plt.subplot(row, col, idx+1)
                idx = min(i * col + j, seg_np.shape[0]-1)
                ax.add_patch(Rectangle([0, -args.canvas_h/2], args.canvas_w, args.canvas_h, color="green" if acc_np[idx]>0.5 else "red", alpha=0.1))
                if self.args.mode=="ship1":
                    for obs_i in range(args.n_obs):
                        offset = 6 + 3 * obs_i
                        ax.add_patch(Ellipse([seg_np[idx, 0, offset], seg_np[idx, 0, offset + 1]], seg_np[idx, 0, offset + 2] * 2, seg_np[idx, 0, offset + 2] * 2, 
                                                label="obstacle" if obs_i==0 else None, color="gray", alpha=0.8))
                else:
                    offset = 6
                    ax.add_patch(Ellipse([seg_np[idx, 0, offset], seg_np[idx, 0, offset + 1]], seg_np[idx, 0, offset + 2] * 2, seg_np[idx, 0, offset + 2] * 2, 
                                    label="obstacle", color="gray", alpha=0.8))
                    for ti in range(0, args.nt, 2):
                        ax.text(seg_np[idx, ti, 0]+0.25, seg_np[idx, ti, 1]+0.25, "%.1f"%(seg_np[idx, ti, -1]), fontsize=6)
                ax.add_patch(Ellipse([seg_np[idx, 0, 0], seg_np[idx, 0, 1]], 0.5, 0.5, label="ego", color="blue", alpha=0.8))
                plt.plot(seg_np[idx, :, 0], seg_np[idx, :, 1], label="trajectory", color="blue", linewidth=2, alpha=0.5)
                if idx==0:
                    plt.legend(fontsize=6, loc="lower right")
                ax.axis("scaled")
                plt.xlim(0-bloat, args.canvas_w+bloat)
                plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)

        figname="%s/iter_%05d.png"%(args.viz_dir, epi)
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def step(self, action):
        args = self.args
        action[..., 0] = np.clip(action[..., 0] * args.thrust_max, -args.thrust_max, args.thrust_max)
        action[..., 1] = np.clip(action[..., 1] * args.delta_max, -args.delta_max, args.delta_max)
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
        args = self.args
        def reward_fn(act, state):
            if args.mode=="ship1":
                collision_penalty = -100
                base_reward = 10
                reward = np.zeros((state.shape[0],)) + base_reward
                is_collide0 = np.linalg.norm(state[:, :2]-state[:, 6:8], axis=-1) < state[:, 8]
                is_collide1 = np.linalg.norm(state[:, :2]-state[:, 6+3:8+3], axis=-1) < state[:, 8+3]
                is_collide2 = np.abs(state[:, 1]) > args.river_width/2
                reward[is_collide0] += collision_penalty
                reward[is_collide1] += collision_penalty
                reward[is_collide2] += collision_penalty
            else:
                collision_penalty = -100
                timeout_penalty = -100
                base_reward = 10
                reward = np.zeros((state.shape[0],)) + base_reward
                # TODO

                is_collide0 = np.linalg.norm(state[:, :2]-state[:, 6:8], axis=-1) < state[:, 8]
                reward[is_collide0] += collision_penalty

                is_offtrack = np.logical_and(state[:, -1]<0, np.abs(state[:, 1]) > args.track_thres)
                reward[is_offtrack] += timeout_penalty

                is_collide2 = np.abs(state[:, 1]) > args.river_width/2
                reward[is_collide2] += collision_penalty
            return reward
        return reward_fn
    
    def render(self, mode):
        args = self.args
        nt = args.nt
        ti = self.t
        bloat = 0.0
        state = self.state
        ax = plt.gca()
        ax.add_patch(Rectangle([0, -args.canvas_h/2], args.canvas_w, args.canvas_h, color="white", alpha=0.01))
        if self.args.mode=="ship1":
            for obs_i in range(args.n_obs):
                offset = 6 + 3 * obs_i
                ax.add_patch(Ellipse([state[offset], state[offset + 1]], state[offset + 2] * 2, state[offset + 2] * 2, 
                                        label="obstacle" if obs_i==0 else None, color="gray", alpha=0.8))
        else:
            # TODO
            offset = 6
            ax.add_patch(Ellipse([state[offset], state[offset + 1]], state[offset + 2] * 2, state[offset + 2] * 2, 
                                        label="obstacle", color="gray", alpha=0.8))
        ax.add_patch(Ellipse([state[0], state[1]], 0.5, 0.5, label="ego", color="blue", alpha=0.8))
        plt.plot([xx[0] for xx in self.history], [xx[1] for xx in self.history], label="trajectory", color="blue", linewidth=2, alpha=0.5)
        plt.legend(fontsize=6, loc="lower right")
        ax.axis("scaled")
        plt.xlim(0-bloat, args.canvas_w+bloat)
        plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)
        plt.title("Simulation (%04d/%04d)"%(ti, nt))
        plt.savefig("%s/rl_sim_%05d_%02d.png"%(args.viz_dir, self.sample_idx, self.t), bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    # TODO test function
    def test(self):
        pass