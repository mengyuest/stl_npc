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


def car_term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    done = torch.any(torch.abs(next_obs[:, :]) > 100, dim=-1)
    done = done[:, None]
    return done

class Lane():
    def __init__(self, id, from_xy, to_xy, lane_width, lane_length, from_id, to_id, viz_xy, viz_w, viz_h, lane_type):
        self.lane_id = id
        self.from_xy = from_xy
        self.to_xy = to_xy
        self.lane_width = lane_width
        self.lane_length = lane_length
        self.from_id = from_id
        self.to_id = to_id
        self.veh_ids = []
        self.viz_xy = viz_xy
        self.viz_w = viz_w
        self.viz_h = viz_h
        self.lane_type = lane_type
    
    def dist(self):
        # > 0, remaining
        # < 0, passed
        return self.lane_length
    
    def get_dx_dy(self, ds):
        if self.lane_type == 0:
            return 0.0, ds
        elif self.lane_type == 1:
            return ds, 0.0
        elif self.lane_type == 2:
            return 0.0, -ds
        else:
            return -ds, 0.0


class Intersection():
    def __init__(self, id, xy, width, is_light, timer, viz_xy, viz_w, viz_h, dt, phase_t):
        self.inter_id = id
        self.xy = xy
        self.width = width
        self.length = width
        self.is_light = is_light
        self.timer = timer
        self.veh_ids = []
        self.viz_xy = viz_xy
        self.viz_w = viz_w
        self.viz_h = viz_h
        self.dt = dt
        self.phase_t = phase_t

    def update(self):
        if self.is_light:
            self.timer = (self.timer + self.dt) % self.phase_t

    def dist(self, direction):
        # dir = -1, left; 0, straight; +1, right
        if direction==0:
            return self.width
        elif direction == -1:
            return 0.75 * self.width * np.pi / 2
        elif direction == 1:
            return 0.25 * self.width * np.pi / 2
    
    def get_dx_dy(self, remain_s, ds, lane_type, direction):
        if direction == 0:
            dx = 0
            dy = ds
        elif direction == -1:
            r = 0.75 * self.width
            arc = r * np.pi / 2
            th0 = (1 - remain_s / arc) * np.pi / 2
            th1 = (1 - (remain_s - ds) / arc) * np.pi / 2
            dx = r * (np.cos(th1) - np.cos(th0))
            dy = r * (np.sin(th1) - np.sin(th0))
        elif direction == 1:
            r = 0.25 * self.width
            arc = r * np.pi / 2
            th0 = (1 - remain_s / arc) * np.pi / 2
            th1 = (1 - (remain_s - ds) / arc) * np.pi / 2
            dx = - r * (np.cos(th1) - np.cos(th0))
            dy = r * (np.sin(th1) - np.sin(th0))

        if lane_type == 1:
            dx, dy = dy, -dx
        
        if lane_type == 2:
            dx, dy = -dx, -dy

        if lane_type == 3:
            dx, dy = -dy, dx

        return dx, dy

class Vehicle():
    def __init__(self, id, v, in_lane, in_id, dist, xy, timer, base_i, is_light, hist, tri, out_a, is_hybrid):
        self.id = id
        self.v = v
        self.in_lane = in_lane
        self.in_id = in_id
        self.dist = dist
        self.xy = xy
        self.timer = timer
        self.base_i = base_i
        self.is_light = is_light
        self.hist = hist
        self.tri = tri
        self.out_a = out_a
        self.is_hybrid = is_hybrid

class CarEnv(base_env.BaseEnv):
    def __init__(self, args):
        super(CarEnv, self).__init__(args)
        self.action_space = spaces.Box(np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-10, 0, 0, 0, 0, 0, 0], dtype=np.float32), 
            np.array([10, args.vmax, 1, 10, 10, args.vmax, 1], dtype=np.float32), dtype=np.float32)
        self.generate_stl()
        
    def next_state(self, x, u):  # (n, 7) x (n, 1) -> (n, 7)
        args = self.args
        new_x = torch.zeros_like(x)
        new_x[:, 0] = x[:, 0] + x[:, 1] * args.dt
        mask = (torch.logical_and(x[:, 0]<args.stop_x, torch.logical_and(x[:, 2]==0, x[:, 4]<0))).float() # stop sign, before the stop region
        if args.test:
            new_x[:, 1] = torch.clip(x[:, 1] + u * args.dt, -0.01, 10) * (1-mask) + torch.clip(x[:, 1] + u * args.dt, 0.1, 10) * mask
        else:
            new_x[:, 1] = torch.clip(x[:, 1] + u* args.dt, -0.01, 10)
        new_x[:, 2] = x[:, 2]
        # stop_timer = (x[:, 3] +args.dt * soft_step(x[:,0]-args.stop_x)) * soft_step(-x[:,0])
        stop_timer = x[:, 3] + args.dt * soft_step(x[:,0]-args.stop_x) * soft_step(-x[:,0])
        light_timer = (x[:, 3] + args.dt) % args.phase_t
        new_x[:, 3] = (1-x[:, 2]) * stop_timer + x[:, 2] * light_timer
        new_x[:, 4] = x[:, 4] + (x[:, 5] - x[:, 1]) * args.dt * (x[:, 4]>=0).float()
        new_x[:, 5] = x[:, 5]
        new_x[:, 6] = x[:, 6]
        return new_x

    def init_x_cycle(self, N):
        args = self.args
        #####################################################################
        n = N // 4
        set0_xe = uniform_tensor(-10, args.stop_x, (n, 1))
        bound = torch.clip(torch.sqrt(2*args.amax*(-set0_xe+args.stop_x)), 0, args.vmax)
        set0_ve = uniform_tensor(0, 1, (n, 1)) * bound  # v<\sqrt{2a|x|}
        set0_id = uniform_tensor(0, 0, (n, 1))
        set0_t = uniform_tensor(0, 0, (n, 1))
        if args.heading:
            if args.mock:
                set0_xo = uniform_tensor(-1, -1, (n, 1))
                set0_vo = uniform_tensor(0, 0, (n, 1))
            else:
                set0_xo, set0_vo = self.heading_base_sampling(set0_ve)
        else:
            set0_xo = uniform_tensor(0, 0, (n, 1))
            set0_vo = uniform_tensor(0, 0, (n, 1))
        if args.triggered:
            if args.mock and not args.no_tri_mock:
                set0_tri = rand_choice_tensor([0, 0], (n, 1))
            else:
                set0_tri = rand_choice_tensor([0, 1], (n, 1))
        else:
            set0_tri = uniform_tensor(0, 0, (n, 1))

        set1_xe = uniform_tensor(args.stop_x, 0, (n, 1))
        set1_ve = uniform_tensor(0, 1.0, (n, 1))
        set1_id = uniform_tensor(0, 0, (n, 1))
        set1_t = uniform_tensor(0, args.stop_t+0.1, (n, 1))
        if args.heading:
            if args.mock:
                set1_xo = uniform_tensor(-1, -1, (n, 1))
                set1_vo = uniform_tensor(0, 0, (n, 1))
            else:
                set1_xo = uniform_tensor(-1, -1, (n, 1))
                set1_vo = uniform_tensor(0, 0, (n, 1))
        else:
            set1_xo = rand_choice_tensor([0, 0], (n, 1))
            set1_vo = uniform_tensor(0, 0, (n, 1))
        set1_tri = uniform_tensor(0, 0, (n, 1))

        n2 = 2*n
        set2_xe = uniform_tensor(-10, args.traffic_x, (n2, 1))
        bound = torch.clip(torch.sqrt(2*args.amax*(-set2_xe + args.traffic_x)), 0, args.vmax)
        set2_ve = uniform_tensor(0, 1, (n2, 1)) * bound  # v<\sqrt{2a|x|}
        set2_id = uniform_tensor(1.0, 1.0, (n2, 1))
        set2_t = uniform_tensor(0, args.phase_t, (n2, 1))
        if args.heading:
            if args.mock:
                set2_xo = uniform_tensor(-1, -1, (n2, 1))
                set2_vo = uniform_tensor(0, 0, (n2, 1))
            else:
                set2_xo, set2_vo = self.heading_base_sampling(set2_ve)
        else:
            set2_xo = uniform_tensor(0, 0, (n2, 1))
            set2_vo = uniform_tensor(0, 0, (n2, 1))
        set2_tri = uniform_tensor(0, 0, (n2, 1))

        set0 = torch.cat([set0_xe, set0_ve, set0_id, set0_t, set0_xo, set0_vo, set0_tri], dim=-1)
        set1 = torch.cat([set1_xe, set1_ve, set1_id, set1_t, set1_xo, set1_vo, set1_tri], dim=-1)
        set2 = torch.cat([set2_xe, set2_ve, set2_id, set2_t, set2_xo, set2_vo, set2_tri], dim=-1)
        
        x_init = torch.cat([set0, set1, set2], dim=0).float().cuda()

        return x_init

    def init_x(self, N):
        return self.init_x_cycle(N)
    
    def generate_stl(self):
        args = self.args
        cond1 = Eventually(0, args.nt, AP(lambda x: x[..., 3] - args.stop_t, comment="t_stop>=%.1fs"%(args.stop_t)))   
        cond_sig = Imply(AP(lambda x: x[..., 6]-0.5, "Triggered"), Always(0,args.nt, AP(lambda x:args.stop_x*0.5 - x[..., 0], "stop")))
        cond1 = And(cond1, cond_sig)
        cond2 = Always(0, args.nt, 
                    Not(And(AP(lambda x: args.phase_red - x[...,3], comment="t=red"),
                            AP(lambda x: -x[..., 0] * (x[..., 0]-args.traffic_x), comment="inside intersection")
                    )))
        
        cond3 = Always(0, args.nt, AP(lambda x:x[..., 4]-args.safe_thres,comment="heading>0"))
        stl = ListAnd([
            Imply(AP(lambda x: 0.5-x[..., 2], comment="I=stop"), cond1),  # stop signal condition
            Imply(AP(lambda x: x[...,2]-0.5, comment="I=light"), cond2),  # light signal condition
            Imply(AP(lambda x: x[..., 4]+0.5, comment="heading"), cond3)  # heading condition
            ])
        self.stl = stl
        return stl

    def generate_heur_loss(self, acc, seg):
        args = self.args
        green_mask = (torch.logical_or(seg[:, :, 2]==0, seg[:, :, 3] > args.phase_red)).float()
        v_loss = torch.mean(acc * torch.relu(5-seg[:,:,1]) * green_mask) * args.v_loss
        s_loss = torch.mean(acc * torch.relu(args.traffic_x-seg[:,:,0])) * args.s_loss
        return v_loss + s_loss

    def visualize(self, x_init, seg, acc, epi):
        args = self.args
        seg_np = to_np(seg)
        acc_np = to_np(acc)
        t_len = seg_np.shape[1]
        N = args.num_samples
        n = N // 4
        n2 = 2*n

        linestyle = lambda x: ("-" if x[0, 6] == 0 else "-.") if args.triggered else "-"
        cond_color = lambda x: "green" if x[0]>0 else "red"
        plt.figure(figsize=(12, 8))
        nv = 25
        plt.subplot(3, 2, 1)
        for i in range(nv):
            plt.plot(range(t_len), seg_np[i, :, 0], color=cond_color(acc_np[i]))
        for i in range(N//4, N//4+nv):
            plt.plot(range(t_len), seg_np[i, :, 0], color=cond_color(acc_np[i]))
        plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
        plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

        plt.subplot(3, 2, 3)
        for i in range(nv):
            plt.plot(range(t_len), seg_np[i, :, 3], color=cond_color(acc_np[i]))
        for i in range(N//4, N//4+nv):
            plt.plot(range(t_len), seg_np[i, :, 3], color=cond_color(acc_np[i]))

        plt.subplot(3, 2, 5)
        for i in range(nv):
            plt.plot(range(t_len), seg_np[i, :, 4], color=cond_color(acc_np[i]))
        for i in range(N//4, N//4+nv):
            plt.plot(range(t_len), seg_np[i, :, 4], color=cond_color(acc_np[i]))
        
        ls = "-."
        plt.subplot(3, 2, 2)
        for i in range(N//8, N//8+nv):
            plt.plot(range(t_len), seg_np[i, :, 0], linestyle=ls, color=cond_color(acc_np[i]))
        plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
        plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

        plt.subplot(3, 2, 4)
        for i in range(N//8, N//8+nv):
            plt.plot(range(t_len), seg_np[i, :, 3], linestyle=ls, color=cond_color(acc_np[i]))

        # plot the 
        plt.subplot(3, 2, 6)
        for i in range(N//8, N//8+nv):
            plt.plot(range(t_len), seg_np[i, :, 4], linestyle=ls, color=cond_color(acc_np[i]))       

        plt.savefig("%s/stopsign_iter_%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
        plt.close()

        plt.figure(figsize=(8, 8))
        seg_np = seg_np[n2:]
        acc_np = acc_np[n2:]
        plt.subplot(2, 1, 1)
        for i in range(10):
            for j in range(args.nt-1):
                # plt.plot(range(seg_np.shape[1]), seg_np[i, :, 0], color="green" if acc_np[i,0]>0 else "red")
                plt.plot([j, j+1], [seg_np[i, j, 0], seg_np[i, j+1, 0]], 
                            color="red" if seg_np[i, j, 3] <= args.phase_red else "green")
        plt.axhline(y=args.traffic_x, xmin=0, xmax=args.nt, color="gray")
        plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
        plt.axis("scaled")
        plt.subplot(2, 1, 2)
        for i in range(args.num_samples//2-50, args.num_samples//2):
            plt.plot(range(seg_np.shape[1]), seg_np[i, :, 4], linestyle=linestyle(seg_np[i]), color="green" if acc_np[i,0]>0 else "red")

        plt.savefig("%s/light_iter_%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def heading_base_sampling(self, set0_ve):
        args = self.args
        relu = nn.ReLU()
        n = set0_ve.shape[0]
        set00_xo = uniform_tensor(-1, -1, (n//2, 1))
        set00_vo = uniform_tensor(0, 0, (n//2, 1))
        set01_xo = uniform_tensor(args.safe_thres, args.xo_max, (n//2, 1))
        lower = torch.sqrt(relu((args.safe_thres - set01_xo)*args.amax*2 + set0_ve[n//2:]**2))
        set01_vo = uniform_tensor(0, 1, (n//2, 1)) * (args.vmax-lower) + lower

        return torch.cat([set00_xo, set01_xo], dim=0), torch.cat([set00_vo, set01_vo], dim=0)

    def step(self, action):
        args = self.args
        action = np.clip(action * self.args.amax, -self.args.amax, self.args.amax)
        action = torch.from_numpy(action).unsqueeze(0)
        state = torch.from_numpy(self.state).unsqueeze(0)
        next_state = self.next_state(state, action)
        self.state = to_np(next_state)[0]
        self.history.append(np.array(self.state))
        reward = self.generate_reward(self.state)
        self.t+=1
        terminated = self.t >= args.nt
        return self.state, reward, terminated, {}

    def generate_reward_batch_fn(self): # (n, 7)
        args = self.args
        def reward_fn(act, state):
            reward = np.zeros((state.shape[0],))
            speed_bonus = 0.5

            # stop sign case
            is_stop = state[:, 2] < 0.5
            stop_triggered_and_violated = np.logical_and(is_stop, 
                np.logical_and(state[:, 6] > 0.5, state[:, 0]>args.stop_x * 0.5))
            reward[stop_triggered_and_violated] -= 100.0

            stop_nowait_violated = np.logical_and(is_stop, 
                np.logical_and(state[:, 3] < args.stop_t, state[:, 0]>0))
            reward[stop_nowait_violated] -= 100.0

            stop_exceeded = np.logical_and(is_stop, state[:, 3]>=args.stop_t)
            reward[stop_exceeded] += state[stop_exceeded, 1] * speed_bonus  # encourage moving more when pass stop time

            # traffic light case
            green_light = np.logical_and(np.logical_not(is_stop), state[:, 3] >= args.phase_red)
            red_light = np.logical_and(np.logical_not(is_stop), state[:, 3] < args.phase_red)
            red_light_and_in = np.logical_and(red_light, np.logical_and(state[:, 0] > args.traffic_x, state[:, 0]<0))
            reward[green_light] += state[green_light, 1] * speed_bonus  # encourage speed when green light
            reward[red_light_and_in] += -100  # penalize for pass through intersection under red light

            # lead vehicle case
            close_thres = 1.0
            is_lead = np.logical_or(state[:, 4] != -1, state[:, 5] != 0)
            lead_and_close = np.logical_and(is_lead, state[:, 4] <= close_thres)
            lead_and_free = np.logical_and(is_lead, state[:, 4] > close_thres)
            reward[lead_and_close] += 100*(state[lead_and_close, 4]-close_thres) 
            reward[lead_and_free] += speed_bonus * state[lead_and_free, 1]  # encourage speed when far away
            
            return reward
        return reward_fn

    def render(self, mode):
        args = self.args
        state = self.state
        dx = 1
        dy = 2
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        road = Rectangle([-10, -dy], 20, 2*dy, color="gray")
        ax.add_patch(road)
        if state[2] == 0:  # stop sign
            sign = Rectangle([args.stop_x, -dy], -args.stop_x, 2*dy, color="red", alpha=0.5)
        else:
            sign = Rectangle([args.stop_x, -dy], -args.stop_x, 2*dy, color="red" if state[3]%args.phase_t<args.phase_red else "green", alpha=0.5)
        ax.add_patch(sign)
        ax.text(args.stop_x*0.5, 0, "%.2f"%state[3])

        car = Rectangle([state[0]-dx, -dy], 2*dx, 2*dy, color="blue", alpha=0.75)
        ax.add_patch(car)

        if state[4] != -1 or state[5] != 0:  # lead car
            lead_car = Rectangle([state[0]+state[4]-dx, -dy], 2*dx, 2*dy, color="purple", alpha=0.75)
            ax.add_patch(lead_car)
        
        if state[6] == 1:  # trigger
            warn = Rectangle([-10, -dy], 20, 2*dy, color="red", alpha=0.3)
            ax.add_patch(warn)
        ax.axis("scaled")
        plt.savefig("%s/rl_sim_%05d_%02d.png"%(args.viz_dir, self.sample_idx, self.t), bbox_inches='tight', pad_inches=0.1)
        plt.close()
    

    def get_dir(self, prev_i, prev_j, curr_i, curr_j, next_i, next_j):
        if prev_i is None:
            return False
        di0 = curr_i - prev_i
        dj0 = curr_j - prev_j
        di1 = next_i - curr_i
        dj1 = next_j - curr_j
        
        lefts = [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0)]
        rights = [(1, 0, 0, -1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, -1, 0)]
        if (di0, dj0, di1, dj1) in lefts:
            return -1
        if (di0, dj0, di1, dj1) in rights:
            return 1
        return 0

    
    def compute_lane_dist(self, ego_xy, other_xy, lane_type):
        if lane_type == 0:
            return other_xy[1] - ego_xy[1]
        if lane_type == 1:
            return other_xy[0] - ego_xy[0]
        if lane_type == 2:
            return -other_xy[1] + ego_xy[1]
        if lane_type == 3:
            return -other_xy[0] + ego_xy[0]


    def other_controller(self, veh, x):
        args = self.args
        def cruise(v_ref):
            if veh.v < v_ref:
                u_output = min((v_ref - veh.v) / args.dt, args.amax)
            else:
                u_output = max((v_ref- veh.v) / args.dt, -args.amax)
            return u_output
        def stop():
            u_output = max((0 - veh.v) / args.dt, -args.amax)
            return u_output
        v_ref = 6.0
        if not veh.in_lane:  # in the intersection
            u_output = cruise(v_ref)
        else:  # on the lane
            # exists leading vehicle too close
            the_dist = -args.safe_thres
            if x[4]>=0 and (veh.v*veh.v- x[5]*x[5])/2/args.amax-x[4]+the_dist>=0:
                u_output = -args.amax
            else:
                if x[0] + x[1]*x[1]/2/args.amax < args.stop_x:  # too far from the stop sign/traffic light
                    u_output = cruise(v_ref)
                else:
                    if x[2]<=0.5: # stop sign case
                        if x[3]<args.stop_t or x[6]>0.5:  # not enough stop time or not triggered 
                            u_output = stop()
                        else:
                            u_output = cruise(v_ref)                   
                    else: # traffic light case
                        if x[3]<args.phase_red or x[3]>args.phase_t-1: # red time
                            u_output = stop()
                        else: # green light
                            u_output = cruise(v_ref)
        return u_output



    def test(self, net, rl_policy, stl, cache):
        args = self.args
        self.cache = cache
        self.net = net
        self.rl_policy = rl_policy
        self.stl = stl

        n_trials = 10
        nt = 300
        nx = 4
        ny = 5
        M = int(nx * ny * 0.6)
        IW = 6
        n_roads = 16
        dx_choices = [10., 12, 15, 18] # [6, 8, 10, 15]
        dy_choices = [12., 13, 15, 20] # [7, 10, 13, 15]
        if args.hybrid:
            n_vehs = 10
            n_hybrids = 10
        else:
            n_vehs = 15 # 10
            n_hybrids = 0
        
        for trial_i in range(n_trials):
            # initialize the env
            dx_list = np.random.choice(dx_choices, nx)
            dy_list = np.random.choice(dy_choices, ny)
            dx_list += IW / 2
            dy_list += IW / 2

            map_xy = np.zeros((ny, nx, 2))
            timers = np.random.rand(ny, nx) * args.phase_t
            is_light = np.random.choice(2, (ny, nx)) 
            is_light[0, 0] = 0
            is_light[0, nx-1] = 0
            is_light[ny-1, 0] = 0
            is_light[ny-1, nx-1] = 0
            
            # generate intersections
            inters = dict()
            inters_2d = [[None for i in range(nx)] for j in range(ny)]
            inter_id = 0
            for i in range(ny):
                for j in range(nx):
                    map_xy[i, j, 0] = np.sum(dx_list[:j]) if j>0 else 0
                    map_xy[i, j, 1] = np.sum(dy_list) - (np.sum(dy_list[:i]) if i>0 else 0)
                    viz_xy = map_xy[i, j] + np.array([-IW/2, -IW/2])
                    viz_w = IW
                    viz_h = IW
                    inters[inter_id] = Intersection(inter_id, map_xy[i, j], IW, is_light[i, j], timers[i, j], viz_xy, viz_w, viz_h)
                    inters_2d[i][j] = inters[inter_id]
                    inter_id += 1



