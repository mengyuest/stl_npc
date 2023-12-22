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

def rover_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.any(torch.abs(next_obs[:, :]) > 100, dim=-1)
    done = done[:, None]
    return done

class RoverEnv(base_env.BaseEnv):
    def __init__(self, args):
        super(RoverEnv, self).__init__(args)
        self.action_space = spaces.Box(np.array([-1.0, -1.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
                np.array([0, 0, 0, 0, 0, 0, -1, -1], dtype=np.float32), 
                np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float32), 
            dtype=np.float32)
        self.generate_objs()
        self.generate_stl()

    def next_state(self, x, u):
        args = self.args
        new_x = torch.zeros_like(x)
        close_enough_dist = args.close_thres
        if args.hard_soft_step:
            if args.norm_ap:
                near_charger = soft_step_hard(args.tanh_ratio*(close_enough_dist - torch.norm(x[:, 0:2] - x[:, 4:6], dim=-1)))
            else:
                near_charger = soft_step_hard(args.tanh_ratio*(close_enough_dist**2 - (x[:, 0] - x[:, 4])**2 - (x[:, 1] - x[:, 5])**2))
        else:
            if args.norm_ap:
                near_charger = soft_step(args.tanh_ratio*(close_enough_dist - torch.norm(x[:, 0:2] - x[:, 4:6], dim=-1)))
            else:
                near_charger = soft_step(args.tanh_ratio*(close_enough_dist**2 - (x[:, 0] - x[:, 4])**2 - (x[:, 1] - x[:, 5])**2))
        v_rover = u[:, 0] * (args.rover_vmax - args.rover_vmin) + args.rover_vmin
        th_rover = u[:, 1]
        vx0 = v_rover * torch.cos(th_rover)
        vy0 = v_rover * torch.sin(th_rover)

        new_x[:, 0] = x[:, 0] + vx0 * args.dt
        new_x[:, 1] = x[:, 1] + vy0 * args.dt
        new_x[:, 2:6] = x[:, 2:6]
        new_x[:, 6] = (x[:, 6] - args.dt) * (1-near_charger) + args.battery_charge * near_charger
        new_x[:, 7] = x[:, 7] - args.dt * near_charger
        
        return new_x
    
    def init_x_cycle(self, N):
        args = self.args
        n = N
        charger_x = uniform_tensor(0, 10, (n, 1))
        charger_y = uniform_tensor(0, 10, (n, 1))    

        MAX_BATTERY_N = 25
        battery_t = rand_choice_tensor([args.dt * nn for nn in range(MAX_BATTERY_N+1)], (n, 1))
        rover_theta = uniform_tensor(-np.pi, np.pi, (n, 1))
        rover_rho = uniform_tensor(0, 1, (n, 1)) * (battery_t * args.rover_vmax)
        rover_rho = torch.clamp(rover_rho, args.close_thres, 14.14)

        rover_x = charger_x + rover_rho * torch.cos(rover_theta)
        rover_y = charger_y + rover_rho * torch.sin(rover_theta)

        dest_x = uniform_tensor(0, 10, (n, 1))
        dest_y = uniform_tensor(0, 10, (n, 1))

        # place hold case
        ratio = 0.25
        rand_mask = uniform_tensor(0, 1, (n, 1))
        rand = rand_mask>1-ratio
        ego_rho = uniform_tensor(0, args.close_thres, (n, 1))
        rover_x[rand] = (charger_x + ego_rho * torch.cos(rover_theta))[rand]
        rover_y[rand] = (charger_y + ego_rho * torch.sin(rover_theta))[rand]
        battery_t[rand] = args.dt * MAX_BATTERY_N

        hold_t = 0 * dest_x + args.dt * args.hold_t
        hold_t[rand] = rand_choice_tensor([args.dt * nn for nn in range(args.hold_t+1)], (n, 1))[rand]

        return torch.cat([rover_x, rover_y, dest_x, dest_y, charger_x, charger_y, battery_t, hold_t], dim=1)

    
    def init_x(self, N):
        args = self.args
        n = N
        x_list = []
        total_n = 0
        while(total_n<n):
            x_init = self.init_x_cycle(n)
            valids = []
            for obj_i, obj in enumerate(self.objs):
                obs_cpu = obj.detach().cpu()
                xmin, xmax, ymin, ymax = \
                    torch.min(obs_cpu[:,0]), torch.max(obs_cpu[:,0]), torch.min(obs_cpu[:,1]), torch.max(obs_cpu[:,1]), 

                for x,y in [(x_init[:,0], x_init[:,1]), (x_init[:,2], x_init[:,3]),(x_init[:,4], x_init[:,5])]:
                    if obj_i ==0:  # in map
                        val = torch.logical_and(
                            (x - xmin) * (xmax - x)>=0, 
                            (y - ymin) * (ymax - y)>=0, 
                            )
                    else:  # avoid obstacles
                        val = torch.logical_not(torch.logical_and(
                            (x - xmin) * (xmax - x)>=0, 
                            (y - ymin) * (ymax - y)>=0, 
                            ))
                    valids.append(val)
            
            valids = torch.stack(valids, dim=-1)
            valids_indices = torch.where(torch.all(valids, dim=-1)==True)[0]
            x_val = x_init[valids_indices]
            total_n += x_val.shape[0]
            x_list.append(x_val)
        
        x_list = torch.cat(x_list, dim=0)[:n]
        return x_list

    def in_poly(self, xy0, xy1, poly):
        n_pts = 1000
        ts = torch.linspace(0, 1, n_pts)
        xys = xy0.unsqueeze(0) + (xy1-xy0).unsqueeze(0) * ts.unsqueeze(1)
        xmin, xmax, ymin, ymax = torch.min(poly[:,0]), torch.max(poly[:,0]),torch.min(poly[:,1]), torch.max(poly[:,1])
        
        inside = torch.logical_and(
            (xys[:,0]-xmin) * (xmax -xys[:,0])>=0,
            (xys[:,1]-ymin) * (ymax -xys[:,1])>=0,
        )
        
        res = torch.any(inside)
        return res

    def generate_objs(self):
        args = self.args
        objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
        objs_np.append(np.array([[0.0, 0.0], [args.obs_w, 0], [args.obs_w, args.obs_w], [0, args.obs_w]]))  # first obstacle
        objs_np.append(objs_np[1] + np.array([[5-args.obs_w/2, 10-args.obs_w]]))  # second obstacle (top-center)
        objs_np.append(objs_np[1] + np.array([[10-args.obs_w, 0]]))  # third obstacle (bottom-right)
        objs_np.append(objs_np[1] / 2 + np.array([[5-args.obs_w/4, 5-args.obs_w/4]]))  # forth obstacle (center-center, shrinking)

        objs = [to_torch(ele) for ele in objs_np]
        objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
        objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]
        self.objs_np, self.objs, self.objs_t1, self.objs_t2 = objs_np, objs, objs_t1, objs_t2


    def generate_stl(self):
        args = self.args
        objs_np, objs, objs_t1, objs_t2 = self.objs_np, self.objs, self.objs_t1, self.objs_t2

        in_map = Always(0, args.nt, 
            AP(lambda x: pts_in_poly(x[..., :2], objs[0], args, obses_1=objs_t1[0], obses_2=objs_t2[0]))
        )

        avoid_func = lambda y, y1, y2: Always(0, args.nt, And(
            AP(lambda x: -pts_in_poly(x[..., :2], y, args, obses_1=y1, obses_2=y2)), 
            AP(lambda x: args.seg_gain * -seg_int_poly(x[..., :2], y, args, obses_1=y1, obses_2=y2))
        ))

        avoids = []
        for obs, obs1, obs2 in zip(objs[1:], objs_t1[1:], objs_t2[1:]):
            avoids.append(avoid_func(obs, obs1, obs2))
        if args.norm_ap:
            at_dest = AP(lambda x: args.close_thres - torch.norm(x[...,0:2]-x[...,2:4], dim=-1))
            at_charger = AP(lambda x: args.close_thres - torch.norm(x[...,0:2]-x[...,4:6], dim=-1))
        else:
            at_dest = AP(lambda x: -(x[...,0]-x[...,2])**2-(x[...,1]-x[...,3])**2+args.close_thres**2)
            at_charger = AP(lambda x: -(x[...,0]-x[...,4])**2-(x[...,1]-x[...,5])**2+args.close_thres**2)
        
        battery_limit = args.dt*args.nt

        reach0 = Imply(AP(lambda x: x[..., 6] - battery_limit), Eventually(0, args.nt, at_dest))
        battery = Always(0, args.nt, AP(lambda x:x[..., 6]))
        
        reaches = [reach0]
        emergency = Imply(AP(lambda x: battery_limit - x[..., 6]), Eventually(0, args.nt, at_charger))
        if args.hold_t > 0:
            if args.norm_ap:
                stand_by = AP(lambda x: 0.1 - torch.norm(x[..., 0:2] - x[..., 0:1, 0:2], dim=-1), comment="Stand by")
            else:
                stand_by = AP(lambda x: 0.1 **2 - (x[..., 0] - x[..., 0:1, 0])**2 -  (x[..., 1] - x[..., 0:1, 1])**2, comment="Stand by")
            enough_stay = AP(lambda x: -x[..., 7], comment="Stay>%d"%(args.hold_t))
            hold_cond = Imply(at_charger, Always(0, args.hold_t, Or(stand_by, enough_stay)))
            hold_cond = [hold_cond]
        else:
            hold_cond = []
        stl = ListAnd([in_map] + avoids + reaches + hold_cond + [battery, emergency])
        self.stl = stl
        return stl
    
    def generate_heur_loss(self, acc, seg):
        args = self.args
        battery_limit = args.dt*args.nt
        x0 = seg[:, 0]
        shall_charge = (x0[..., 6:7] <= battery_limit).float()
        acc_mask = acc
        if args.no_acc_mask:
            acc_mask = 1
        charge_dist = torch.norm(seg[:, :, :2] - x0.unsqueeze(dim=1)[:, :, 4:6], dim=-1)
        dest_dist = torch.norm(seg[:, :, :2] - x0.unsqueeze(dim=1)[:, :, 2:4], dim=-1)
        dist_loss = torch.mean((charge_dist * shall_charge + dest_dist * (1 - shall_charge)) * acc_mask)
        dist_loss = dist_loss * args.dist_w
        return dist_loss
    
    def visualize(self, x_init, seg, acc, epi):
        args = self.args

        init_np = to_np(x_init)
        seg_np = to_np(seg)
        acc_np = to_np(acc)

        at_goal_sig = np.linalg.norm(seg_np[:, :, 0:2] - seg_np[:, :, 2:4], axis=-1) < args.close_thres
        at_charger_sig = np.linalg.norm(seg_np[:, :, 0:2] - seg_np[:, :, 4:6], axis=-1) < args.close_thres

        col = 5
        row = 5 * 2
        bloat = 0.5
        
        f, ax_list = plt.subplots(row, col, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1] * (row//2), 'width_ratios':[1] * col})
        for i in range(row//2):
            for j in range(col):
                idx = (i * 2) * col + j
                ax = ax_list[i*2, j]
                self.plot_env(ax)

                ax.add_patch(Rectangle([0,0], 10, 10, color="green" if acc_np[idx]>0.5 else "red", alpha=0.1))
                ax.scatter(seg_np[idx,0,0], seg_np[idx,0,1], color="blue", label="rover")
                ax.scatter(seg_np[idx,0,2], seg_np[idx,0,3], color="green", label="dest")
                ax.scatter(seg_np[idx,0,4], seg_np[idx,0,5], color="yellow", label="charger")
                ax.plot(seg_np[idx,:,0], seg_np[idx,:,1], color="blue", linewidth=2, alpha=0.5, zorder=10)
                for ti in range(args.nt):
                    ax.text(seg_np[idx, ti, 0]+0.25, seg_np[idx, ti, 1]+0.25, "%.1f"%(seg_np[idx, ti, 6]), fontsize=5)
                plt.xlim(0-bloat, 10+bloat)
                plt.ylim(0-bloat, 10+bloat)
                if idx==0:
                    ax.legend(fontsize=6, loc="lower right")
                ax.axis("scaled")

                ax = ax_list[i*2+1, j]
                t_ranges = list(range(args.nt+1))
                _alpha = 0.5
                _lw = 2
                ax.plot(t_ranges, seg_np[idx, :, 6] / (25*args.dt), color="orange", alpha=_alpha, linewidth=_lw, label="battery")  # goal_t
                ax.plot(t_ranges, seg_np[idx, :, 7] / (25*args.dt), color="blue", alpha=_alpha, linewidth=_lw, label="c_stay_t")  # stay_t
                ax.plot(t_ranges, (at_charger_sig[idx, :])*1.0, color="purple", alpha=_alpha, linewidth=_lw, label="at charger")
                ax.plot(t_ranges, (at_goal_sig[idx, :])*1.0, color="gray", alpha=_alpha, linewidth=_lw, label="at goal")
                if idx==0:
                    ax.legend(fontsize=6, loc="upper right")
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                plt.ylim(-0.2, 1.2)

        figname="%s/iter_%05d.png"%(args.viz_dir, epi)
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    def step(self, action):
        args = self.args
        action[..., 0] = np.clip(action[..., 0]/2 + 0.5, 0, 1)
        action[..., 1] = np.clip(action[..., 1] * np.pi, -np.pi, np.pi)
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
        def check_in_box(pts, poly):
            xmin, xmax = np.min(poly[:, 0]), np.max(poly[:, 0])
            ymin, ymax = np.min(poly[:, 1]), np.max(poly[:, 1])
            return np.logical_and(
                (pts[...,0]-xmin)*(pts[...,0]-xmax)<0,
                (pts[...,1]-ymin)*(pts[...,1]-ymax)<0,
            )
        args = self.args
        if hasattr(self, "objs_np"):
            objs_np = self.objs_np
        else:
            self.generate_objs()
            objs_np = self.objs_np
        def reward_fn(act, state):
            collision_penalty = -100
            out_of_map_penalty = -100
            out_of_battery_penalty = -100
            reach_station_reward = 10
            reach_dest_reward = 10
            stay_station_reward = 10
            reward = np.zeros((state.shape[0],))
            close_enough_dist = args.close_thres

            is_collide0 = check_in_box(state[:, 0:2], objs_np[1])
            is_collide1 = check_in_box(state[:, 0:2], objs_np[2])
            is_collide2 = check_in_box(state[:, 0:2], objs_np[3])
            is_collide3 = check_in_box(state[:, 0:2], objs_np[4])

            is_collide = np.logical_or(
                np.logical_or(is_collide0, is_collide1),
                np.logical_or(is_collide2, is_collide3)
            )
            out_of_map = np.logical_not(check_in_box(state[:, 0:2], objs_np[0]))
            out_of_battery = state[:, 6] < 0
            reached_station = np.linalg.norm(
                state[:, 0:2] - state[:, 4:6], axis=-1
            ) < close_enough_dist
            reached_dest = np.linalg.norm(
                state[:, 0:2] - state[:, 2:4], axis=-1
            ) < close_enough_dist
            stayed_at_station = state[:, 7] < 0

            reward[is_collide] += collision_penalty
            reward[out_of_map] += out_of_map_penalty
            reward[out_of_battery] += out_of_battery_penalty
            reward[reached_station] += reach_station_reward
            reward[reached_dest] += reach_dest_reward
            reward[stayed_at_station] += stay_station_reward
            
            return reward
        return reward_fn
    
    def render(self, mode):
        args = self.args
        nt = args.nt
        ti = self.t
        bloat = 0.5

        f, ax_list = plt.subplots(2, 1, figsize=(5, 8)) 
        ax = ax_list[0]

        self.plot_env(ax)
        s = self.state
        ax.scatter(s[0], s[1], color="blue", label="rover")
        ax.scatter(s[2], s[3], color="green", label="dest")
        ax.scatter(s[4], s[5], color="yellow", label="charger")
        ax.plot([xx[0] for xx in self.history], 
                [xx[1] for xx in self.history], 
                color="blue", linewidth=2, alpha=0.5, zorder=10)
        ss0 = [xx[0] for xx in self.history]
        ss1 = [xx[1] for xx in self.history]
        ss6 = [xx[6] for xx in self.history]
        for tti in range(len(ss0)):
            ax.text(ss0[tti]+0.25, ss1[tti]+0.25, "%.1f"%(ss6[tti]), fontsize=5)
        plt.xlim(0-bloat, 10+bloat)
        plt.ylim(0-bloat, 10+bloat)
        
        ax.legend(fontsize=6, loc="lower right")
        ax.axis("scaled")

        ax = ax_list[1]
        t_ranges = list(range(len(self.history)))
        _alpha = 0.5
        _lw = 2
        s6 = np.array([xx[6] for xx in self.history])
        s7 = np.array([xx[7] for xx in self.history])
        seg_np = np.stack(self.history, axis=0)
        at_goal_sig = np.linalg.norm(seg_np[:, 0:2] - seg_np[:, 2:4], axis=-1) < args.close_thres
        at_charger_sig = np.linalg.norm(seg_np[:, 0:2] - seg_np[:, 4:6], axis=-1) < args.close_thres

        ax.plot(t_ranges, s6 / (25*args.dt), color="orange", alpha=_alpha, linewidth=_lw, label="battery")  # goal_t
        ax.plot(t_ranges, s7 / (25*args.dt), color="blue", alpha=_alpha, linewidth=_lw, label="c_stay_t")  # stay_t
        ax.plot(t_ranges, at_charger_sig*1.0, color="purple", alpha=_alpha, linewidth=_lw, label="at charger")
        ax.plot(t_ranges, at_goal_sig*1.0, color="gray", alpha=_alpha, linewidth=_lw, label="at goal")
        ax.legend(fontsize=6, loc="upper right")
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.ylim(-0.2, 1.2)
        plt.title("Simulation (%04d/%04d)"%(ti, nt))
        plt.savefig("%s/rl_sim_%05d_%02d.png"%(args.viz_dir, self.sample_idx, self.t), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def plot_env(self, ax):
        for ii, obj in enumerate(self.objs_np[1:]):
            rect = Polygon(obj, color="gray", alpha=0.25, label="obstacle" if ii==0 else None)
            ax.add_patch(rect)
 
    def test(self):
        pass