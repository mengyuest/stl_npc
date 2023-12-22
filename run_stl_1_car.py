from lib_stl_core import *
from matplotlib.patches import Polygon, Rectangle, Ellipse
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.size': 20})

import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, build_relu_nn, soft_step_hard, get_exp_dir, xxyy_2_Ab, eval_proc
from lib_cem import solve_cem_func
from utils_mbrl import get_mbrl_models, get_mbrl_u

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.net = build_relu_nn(7, args.nt, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):
        u = self.net(x)
        if args.no_tanh:
            u = torch.clip(u, -10.0, 10.0)
        else:
            u = torch.tanh(u) * 10.0
        return u

def soft_step(x):
    if args.hard_soft_step:
        return soft_step_hard(args.tanh_ratio * x)
    else:
        return (torch.tanh(500 * x) + 1)/2


def soft_step_ft(x):
    hard = (x>=0).float()
    soft = (torch.tanh(args.tanh_ratio * x) + 1)/2
    return soft

def dynamics(x0, u, include_first=False, finetune=False):
    # input:  x, (n, 6)  # xe, ve, i, t, x_dist, vo, trigger
    # input:  u, (n, T, 1)
    # return: s, (n, T, 6)
    t = u.shape[1]
    x = x0.clone()
    segs = []
    if include_first:
        segs.append(x)
    for ti in range(t):
        new_x = dynamics_per_step(x, u[:, ti:ti+1], finetune=finetune)
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)


def dynamics_per_step(x, u, finetune=False):
    new_x = torch.zeros_like(x)
    new_x[:, 0] = x[:, 0] + x[:, 1] * args.dt
    mask = (torch.logical_and(x[:, 0]<args.stop_x, torch.logical_and(x[:, 2]==0, x[:, 4]<0))).float() # stop sign, before the stop region
    if args.test:
        if args.finetune:
            new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10) * (1-mask) + torch.clip(x[:, 1] + (u[:, 0]) * args.dt, 0.1, 10) * mask
        else:
            new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10) * (1-mask) + torch.clip(x[:, 1] + (u[:, 0]) * args.dt, 0.1, 10) * mask
    else:
        new_x[:, 1] = torch.clip(x[:, 1] + (u[:, 0]) * args.dt, -0.01, 10)
    new_x[:, 2] = x[:, 2]
    if args.finetune and finetune:
        stop_timer = (x[:, 3] + args.dt * soft_step_ft((x[:,0]-args.stop_x))) * soft_step_ft(-x[:,0])
    else:
        stop_timer = (x[:, 3] + args.dt * soft_step(x[:,0]-args.stop_x)) * soft_step(-x[:,0])
    light_timer = (x[:, 3] + args.dt) % args.phase_t
    new_x[:, 3] = (1-x[:, 2]) * stop_timer + x[:, 2] * light_timer
    new_x[:, 4] = x[:, 4] + (x[:, 5] - x[:, 1]) * args.dt * (x[:, 4]>=0).float()
    new_x[:, 5] = x[:, 5]
    new_x[:, 6] = x[:, 6]
    return new_x

def get_rl_xs_us(x, policy, nt, include_first=False):
    xs = []
    us = []
    dt_minus=0
    if include_first:
        xs.append(x)
    for ti in range(nt):
        tt1=time.time()
        if args.rl:
            u, _ = policy.predict(x.cpu(), deterministic=True)
            u = torch.from_numpy(u * args.amax)
        else:
            if args.mbpo:
                u = get_mbrl_u(x, None, policy, mbpo=True)
            elif args.pets:
                u_list=[]
                for iii in range(x.shape[0]):
                    u = get_mbrl_u(x[iii], None, policy, mbpo=False)
                    u_list.append(u)
                u = torch.stack(u_list, dim=0)
            u = u * args.amax
        u = torch.clip(u, -args.amax, args.amax).cuda()
        new_x = dynamics_per_step(x, u)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.cat(us, dim=1)  # because u [N,1] => [N,T]
    return xs, us, dt_minus

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
    def __init__(self, id, xy, width, is_light, timer, viz_xy, viz_w, viz_h):
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

    def update(self):
        if self.is_light:
            self.timer = (self.timer + args.dt) % args.phase_t

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


def get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j):
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


def compute_lane_dist(ego_xy, other_xy, lane_type):
    if lane_type == 0:
        return other_xy[1] - ego_xy[1]
    if lane_type == 1:
        return other_xy[0] - ego_xy[0]
    if lane_type == 2:
        return -other_xy[1] + ego_xy[1]
    if lane_type == 3:
        return -other_xy[0] + ego_xy[0]
    

def other_controller(veh, x):
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


def solve_planner(tti, x_init, vid):
    from lib_pwlplan import plan, Node
    from gurobipy import GRB
    tt1=time.time()
    def func1(m, PWLs, di):
        v = m.addVars(args.nt+1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the_v")
        u = m.addVars(args.nt, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the_u")
        m.addConstr(v[0] == di["v"])
        for i in range(args.nt):
            m.addConstr(PWLs[0][i+1][0][1] == PWLs[0][i][0][1])
            m.addConstr(PWLs[0][i+1][0][0] == PWLs[0][i][0][0] + v[i+1] * args.dt)
            if di["head_x"] > -0.5:
                m.addConstr(-PWLs[0][i+1][0][0]+PWLs[0][0][0][0] + di["head_x"] + di["head_v"] * (i+1) * args.dt >= args.safe_thres)
            m.addConstr(v[i+1] == v[i] + u[i] * args.dt)
            m.addConstr(v[i] >= 0)
            m.addConstr(v[i] <= args.vmax)
            m.addConstr(u[i] <= args.amax)
            m.addConstr(u[i] >= -args.amax)
        return -100 * sum(PWLs[0][i][0][0] for i in range(args.nt))

    
    nt = args.nt
    dt = args.dt
    tmax = (nt+1)*dt
    vmax = args.vmax
    anti_bloat = 0.1
    A_int, b_int = xxyy_2_Ab([args.stop_x*0.5+anti_bloat, 5-anti_bloat, -2, 2])
    A_stop, b_stop = xxyy_2_Ab([args.stop_x+anti_bloat, -anti_bloat, -2, 2])
    A_tra, b_tra = xxyy_2_Ab([args.traffic_x+anti_bloat, 0.5-anti_bloat, -2, 2])
    A_free, b_free = xxyy_2_Ab([5, 10, 5, 10])
    # state: x, v, is_traffic=0/1, timer, head_x, head_v, triggered
    _x, _v, _I, _T, x_head, v_head, triggered = to_np(x_init)[0]
    if _I <= 0.5:
        # wait_until_inside the stop region
        if triggered:
            always_not_in_intesection = Node('A', deps=[Node("negmu", info={"A":A_int, "b":b_int})], info={'int':[0, tmax]})
            specs = [always_not_in_intesection]
        else:
            wait_until_stop = Node("F", deps=[Node("A", deps=[Node("mu", info={"A":A_stop, "b":b_stop})], info={"int":[0, max(0,args.stop_t-_T)]})], info={"int":[0, tmax]})
            specs = [wait_until_stop]
    else:  # green/red light
        A_list = [A_tra if (_T + dt * (i + 1)) % args.phase_t <= args.phase_red else A_free for i in range(nt)]
        b_list = [b_tra if (_T + dt * (i + 1)) % args.phase_t <= args.phase_red else b_free for i in range(nt)]
        not_in_int_when_red = Node("A", deps=[Node("negmus", info={"A":A_list, "b":b_list})], info={"int":[0, tmax]})
        specs = [not_in_int_when_red]
    x0s = [np.array([x_init[0,0].item(), 0.0])]
    cache_dict = {"v": x_init[0, 1].item(), "head_x":x_init[0, 4].item(), "head_v":x_init[0,5].item()}
    PWL, u_out = plan(x0s, specs, bloat=0.01, MIPGap=0.05, num_segs=args.nt, tmax=tmax, vmax=vmax, extra_fn_list=[func1], return_u=True, quiet=True, cache_dict=cache_dict)
    if vid==1:
        print("x:%.2f v:%.2f I:%.1f t:%.2f xh:%.2f vh:%.2f tri:%.1f"%(
            x_init[0, 0].item(), x_init[0, 1].item(), x_init[0, 2].item(), 
            x_init[0, 3].item(), x_init[0, 4].item(), x_init[0, 5].item(), x_init[0, 6].item(), 
        ))
    if PWL[0] is None:
        print(vid, "FAILED~~~~")
        u_out = -args.amax * np.ones(args.nt)
    else:
        if vid==1:
            for i in range(nt+1):
                print(i, PWL[0][i][0][0], PWL[0][i][0][1])
    # avoid front car is encoded in the constraints
    return torch.from_numpy(u_out[None]).cuda()


def gradient_solve(tti, x_init, stl, multi_test=False, init_guess=None):
    relu = torch.nn.ReLU()
    if init_guess is None:
        u_lat = torch.zeros(x_init.shape[0], args.nt).requires_grad_()
    else:
        u_lat = init_guess.detach().clone().cpu().requires_grad_()
    x_init = x_init.cpu()
    optimizer = torch.optim.Adam([u_lat], lr=args.grad_lr)
    tt1=time.time()
    for i in range(args.grad_steps):
        if init_guess is None:
            u = torch.nn.Tanh()(u_lat) * args.amax
        else:
            u = u_lat
        seg = dynamics(x_init, u, include_first=True, finetune=True)
        if init_guess is not None:
            for tti_debug in range(args.nt+1):
                s_ = seg[0, tti_debug]
                print("tti=%02d x:%.2f v:%.2f I:%.2f t:%.2f dx_head:%.2f v_head:%.2f tri:%.2f"%(tti_debug,
                        s_[0], s_[1], s_[2], s_[3], s_[4], s_[5], s_[6]
                        ))
        
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        stl_loss = torch.mean(relu(0.5-score))

        green_mask = (torch.logical_or(seg[:, :, 2]==0, seg[:, :, 3] > args.phase_red)).float()
        v_loss = torch.mean(acc * torch.relu(5-seg[:,:,1]) * green_mask) * args.v_loss
        s_loss = torch.mean(acc * torch.relu(args.traffic_x-seg[:,:,0])) * args.s_loss
        if init_guess is None:
            loss = stl_loss + v_loss + s_loss
        else:
            loss = 1-seg[0, -1, 3]
            print("loss", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.grad_print_freq == 0:
            print(i, loss.item())
    tt2=time.time()
    print("%05d t:%.4f seconds"%(tti, tt2-tt1))
    return u.detach().cuda()


def gurobi_solve(tti, x_init):
    import gurobipy as gp
    from gurobipy import GRB

    nt = args.nt
    dt = args.dt
    M = 10^6
    VMAX = 10
    VMIN = -0.01
    UMAX = args.amax
    UMIN = -args.amax

    stop_x = args.stop_x
    safe_thres = args.safe_thres
    phase_t = args.phase_t
    phase_red = args.phase_red
    traffic_x = args.traffic_x
    stop_t = args.stop_t

    x_init = to_np(x_init)
    print(x_init)

    is_triggered = x_init[0, 6] >= 0.5
    is_traffic_light = x_init[0, 2] >= 0.5
    has_head_veh = x_init[0,4] >= -0.5

    t_start=time.time()
    m = gp.Model("mip1")
    m.setParam( 'OutputFlag', False)
    x = m.addVars(nt+1, name="x", lb=float('-inf'), ub=float('inf'))
    v = m.addVars(nt+1, name="v", lb=float('-inf'), ub=float('inf'))
    
    if is_triggered:
        gamma_tri = m.addVars(nt, name="gamma_tri", lb=float('-inf'), ub=float('inf'))

    if is_traffic_light:
        pt = [(x_init[0, 3] + dt * i) % (phase_t) for i in range(nt+1)]
        passover = m.addVars(nt, vtype=GRB.BINARY, name="passover")
        gamma = m.addVars(nt, name="gamma", lb=float('-inf'), ub=float('inf'))
    else:
        pt = None
        counter_t = x_init[0, 3]
        stays = m.addVars(nt, vtype=GRB.BINARY, name="stays")
        gamma_stop = m.addVars(nt, name="gamma_stop", lb=float('-inf'), ub=float('inf'))

    if has_head_veh:
        head_v = x_init[0, 5]
        x1 = m.addVars(nt+1, name="x1", lb=float('-inf'), ub=float('inf'))
        gamma_head = m.addVars(nt, name="gamma_head", lb=float('-inf'), ub=float('inf'))
    u = m.addVars(nt, name="u", lb=float('-inf'), ub=float('inf'))
    
    # initial setup
    m.addConstr(x[0] == x_init[0, 0])
    m.addConstr(v[0] == x_init[0, 1])
    if has_head_veh:
        m.addConstr(x1[0] == x_init[0, 4])

    # constraints
    for ti in range(nt):
        # clip
        m.addConstr(v[ti]<= VMAX)
        m.addConstr(v[ti]>= VMIN)
        m.addConstr(u[ti]<= UMAX)
        m.addConstr(u[ti]>= UMIN)

        # dynamics
        m.addConstr(x[ti+1] == x[ti] + v[ti] * dt)
        m.addConstr(v[ti+1] == v[ti] + u[ti] * dt)
        if has_head_veh:
            m.addConstr(x1[ti+1] == x1[ti] + (head_v - v[ti]) * dt)

        # constraints
        if is_triggered:  # stop at stop sign if triggered
            m.addConstr(x[ti+1] <= 0.5 * stop_x + gamma_tri[ti])
        if has_head_veh:
            m.addConstr(x1[ti+1] >= safe_thres - gamma_head[ti])  # safety ~ leading vehicle
        if is_traffic_light and pt[ti] < phase_red:  # stop at traffic light if red
            m.addConstr(x[ti+1] <= traffic_x + passover[ti] * M + gamma[ti])
            m.addConstr(x[ti+1] >= 0 - (1-passover[ti]) * M - gamma[ti])
        elif is_traffic_light == False: # stop sign case, update rules, wait until
            m.addConstr(x[ti+1] <= 0 + (1-stays[ti]) * M + gamma_stop[ti])
            m.addConstr(x[ti+1] >= stop_x - (1-stays[ti]) * M + gamma_stop[ti])
            if ti==nt-1:
                m.addConstr(gp.quicksum(stays) * dt + counter_t >= stop_t)

    d_max = x[nt-1]
    u_sum = gp.quicksum(u[i]*u[i] for i in range(nt))
    if is_traffic_light:
        g_sum = 100*gp.quicksum(gamma[i]*gamma[i] for i in range(nt))
        gs_sum = 0
    else:
        g_sum = 0
        gs_sum = gp.quicksum(gamma_stop[i]*gamma_stop[i] for i in range(nt))
    if has_head_veh:
        gh_sum = gp.quicksum(gamma_head[i]*gamma_head[i] for i in range(nt))
    else:
        gh_sum = 0
    if is_triggered:
        gt_sum = gp.quicksum(gamma_tri[i]*gamma_tri[i] for i in range(nt))
    else:
        gt_sum = 0
    m.setObjective(-10 * d_max + u_sum + M * 100 * (g_sum + gs_sum + gh_sum + gt_sum), GRB.MINIMIZE)
    m.optimize()

    t_end=time.time()
    print("%.5f seconds"%(t_end-t_start))
    u_torch = []
    for ti in range(nt):
        if has_head_veh:
            print("%02d x:%5.2f v:%5.2f x1:%5.2f u:%5.2f"%(ti, x[ti].X, v[ti].X, x1[ti].X, u[ti].X))
        else:
            print("%02d x:%5.2f v:%5.2f u:%5.2f"%(ti, x[ti].X, v[ti].X, u[ti].X))
        u_torch.append(u[ti].X)
    u_torch = torch.tensor(u_torch).unsqueeze(0)
    return u_torch.cuda()


def solve_cem(ti, x_input, vid, stl, args):
    def dynamics_step_func(x, u):
        return dynamics_per_step(x, u)
    
    def reward_func(trajs):
        return stl(trajs, args.smoothing_factor, d={"hard":True})[:, 0]

    u_min = torch.tensor([-args.amax]).cuda()
    u_max = torch.tensor([args.amax]).cuda()
    u, _, info = solve_cem_func(
        (x_input[0]).cuda(), state_dim=x_input.shape[-1], nt=args.nt, action_dim=u_min.shape[0],
        num_iters=500, n_cand=10000, n_elites=100, policy_type="direct",
        dynamics_step_func=dynamics_step_func, reward_func=reward_func,
        transform=None, u_clip=(u_min, u_max), seed=None, args=None, 
        extra_dict=None, quiet=False, device="gpu", visualize=False
    )
    return u

def sim_multi(net, rl_policy, stl):
    metrics_str=["acc", "reward", "score", "t", "safety", "avg_x", "avg_v"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.car_env import CarEnv
    car_env = CarEnv(args)

    nt = 150 # 350
    N = 1
    if args.hybrid:
        n_vehs = 10
        n_hybrids = 10
        nx = 4
        ny = 5
    else:
        n_vehs = 15 # 10
        n_hybrids = 0
        nx = 4
        ny = 5
    M = int(nx * ny * 0.6)
    IW = 6
    n_roads = 16
    dx_choices = [10., 12, 15, 18] # [6, 8, 10, 15]
    dy_choices = [12., 13, 15, 20] # [7, 10, 13, 15]
    dx_list = np.random.choice(dx_choices, nx)
    dy_list = np.random.choice(dy_choices, ny)
    dx_list += IW / 2
    dy_list += IW / 2

    map_xy = np.zeros((ny, nx, 2))

    # randomly assign traffic light (phases) and stop signs
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

    # build the roads
    lanes = dict()
    lanes_2d = dict()
    lane_id = 0
    viz_from_dxy = np.array([[IW/4, IW/2], [IW/2, -IW/4], [-IW/4, -IW/2], [-IW/2, IW/4]])
    viz_to_dxy = np.array([[IW/4, -IW/2], [-IW/2, -IW/4], [-IW/4, IW/2], [IW/2, IW/4]])
    for i in range(ny):
        for j in range(nx):
            for dii, (di, dj) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
                if 0<=i+di<ny and 0<=j+dj<nx:
                    from_id = inters_2d[i][j].inter_id
                    to_id = inters_2d[i+di][j+dj].inter_id
                    from_xy = map_xy[i, j] + viz_from_dxy[dii]
                    to_xy = map_xy[i+di, j+dj] + viz_to_dxy[dii]
                    lane_width = IW / 2
                    lane_length = np.linalg.norm(from_xy-to_xy)
                    if dii == 0:
                        viz_xy = map_xy[i, j] + np.array([0, IW/2]) 
                    elif dii == 1:
                        viz_xy = map_xy[i, j] + np.array([IW/2, -IW/2])
                    elif dii == 2:
                        viz_xy = map_xy[i+di, j+dj] + np.array([-IW/2, IW/2])
                    elif dii == 3:
                        viz_xy = map_xy[i+di, j+dj] + np.array([IW/2, 0])
                    viz_w = lane_width if dii in [0, 2] else lane_length
                    viz_h = lane_length if dii in [0, 2] else lane_width
                    lanes[lane_id] = Lane(lane_id, from_xy, to_xy, lane_width, lane_length, from_id, to_id, viz_xy, viz_w, viz_h, dii)
                    lanes_2d[from_id, to_id] = lanes[lane_id]
                    lane_id += 1
    
    poss_starts = []
    poss_starts_idx = list(range(ny*nx))
    for i in range(ny):
        for j in range(nx):
            poss_starts.append([i, j])
    # random routes:
    moves = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    routes = {}
    is_used = []
    vehicles = {}
    for veh_id in range(n_vehs+n_hybrids):
        _road_i = np.random.choice(poss_starts_idx)
        _roads = [poss_starts[_road_i]]
        poss_starts_idx.remove(_road_i)
        is_used.append(_roads[0])
        prev_i, prev_j = None, None
        for k in range(n_roads):
            curr_i, curr_j = _roads[-1]
            done = False
            while not done:
                done = True 
                rand_dir = np.random.choice(4)
                next_i, next_j = curr_i + moves[rand_dir][0], curr_j + moves[rand_dir][1]
                if next_i<0 or next_i>=ny or next_j<0 or next_j>=nx:
                    done = False
                    continue
                if k>0 and next_i == _roads[-2][0] and next_j == _roads[-2][1]:
                    done = False 
                    continue
                if is_light[curr_i, curr_j] and get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j)==-1:
                    done = False
                    continue
            prev_i, prev_j = curr_i, curr_j
            _roads.append((next_i, next_j))

        routes[veh_id] = _roads

        # TODO vectorized the cars info
        from_id = inters_2d[_roads[0][0]][_roads[0][1]].inter_id
        to_id = inters_2d[_roads[1][0]][_roads[1][1]].inter_id
        to_inter = inters_2d[_roads[1][0]][_roads[1][1]]
        lane_id = lanes_2d[from_id, to_id].lane_id
        lane = lanes[lane_id]
        lane.veh_ids.append(veh_id)
        
        if to_inter.is_light:
            if lane.lane_type in [0, 2]:
                car_timer = to_inter.timer
            else:
                car_timer = (to_inter.timer + args.phase_red) % args.phase_t
                # red-3, green-5
                # 1 2 3 4 5 6 7 0
                # R R R G G G G R
                # G G G R R R R G
                # 5 6 7 0 1 2 3 4
        else:
            car_timer = 0
        vehicles[veh_id] = Vehicle(
            id=veh_id, v=5.0, in_lane=True, in_id=lane_id, dist=lane.dist(), xy=np.array(lane.from_xy),
            timer=car_timer, base_i=0, is_light=to_inter.is_light, hist=[], tri=0, out_a=0, is_hybrid=veh_id>=n_vehs,
        )
    
    CAR_L = 2
    CAR_W = 1.5
    fs_list = []
    for ti in range(nt):
        # compute cars_tri LOGIC
        cars_out_a = {}
        for int_i in range(ny):
            for int_j in range(nx):
                veh_queue = []
                inter = inters_2d[int_i][int_j]
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if 0<=di+int_i<ny and 0<=dj+int_j<nx:
                        from_inter = inters_2d[di+int_i][dj+int_j]
                        # TODO check vehicles
                        link = lanes_2d[from_inter.inter_id, inter.inter_id]
                        for veh_id in link.veh_ids:
                            veh = vehicles[veh_id]
                            dist = veh.dist
                            v_ego = veh.v
                            lead = 1 if veh_id == link.veh_ids[0] else 0
                            weight = lead * 1000 - dist + v_ego**2/2/args.amax
                            veh_queue.append((veh_id, weight))
                
                if len(veh_queue)>0:
                    veh_queue = sorted(veh_queue, key=lambda x:x[1], reverse=True)
                    veh = vehicles[veh_queue[0][0]]
                    veh.tri = 0.0
                    for veh_pair in veh_queue[1:]:
                        veh_id, weight = veh_pair
                        vehicles[veh_id].tri = 1.0
        # get the info for cars on lanes
        x_input = []
        vid=[]
        for lane_id in lanes:
            lane = lanes[lane_id]
            for veh_id in lane.veh_ids:
                veh = vehicles[veh_id]
                x_ego = -veh.dist
                v_ego = veh.v
                i_light = veh.is_light
                timer = veh.timer
                if veh_id != lane.veh_ids[0]:
                    other_id = lane.veh_ids[0]
                    other_veh = vehicles[other_id]
                    x_o = compute_lane_dist(veh.xy, other_veh.xy, lane.lane_type)
                    v_o = other_veh.v
                    x_o = np.clip(x_o-args.bloat_dist, 0, 10)
                else:
                    x_o = -1
                    v_o = 0
                i_tri = veh.tri

                # clip
                x_ego = np.clip(x_ego, -10, 0)

                print("LANE  t=%d|id=%d x:%.2f v:%.2f L:%.1f T:%.1f xo:%.2f vo:%.2f Trigger:%d"%(ti, veh_id, x_ego, v_ego, i_light, timer, x_o, v_o, i_tri))
                x_input.append([x_ego, v_ego, i_light, timer, x_o, v_o, i_tri])
                vid.append(veh_id)

        x_input = torch.tensor(x_input).float().cuda()
        dt_minus1 = 0
        if x_input.shape[0]>0:
            debug_t1=time.time()       
            if args.mpc:
                u_output = []
                for iii in range(x_input.shape[0]):
                    u_output_item = gurobi_solve(ti, x_input[iii:iii+1])
                    u_output.append(u_output_item)
                u_output = torch.cat(u_output, dim=0)
            elif args.rl or args.mbpo or args.pets:
                tmp_xs, u_output, dt_minus1 = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=True)
            elif args.plan:
                u_output = []
                for iii in range(x_input.shape[0]):
                    u_output_item = solve_planner(ti, x_input[iii:iii+1], vid[iii])
                    u_output.append(u_output_item)
                u_output = torch.cat(u_output, dim=0)
            elif args.cem:
                u_output = []
                for iii in range(x_input.shape[0]):
                    u_output_item = solve_cem(ti, x_input[iii:iii+1], vid[iii], stl, args)
                    u_output.append(u_output_item[None,:])
                u_output = torch.cat(u_output, dim=0)[..., 0]
                
            elif args.grad:
                u_output = gradient_solve(ti, x_input, stl)
            else:
                u_output = net(x_input)
                if args.finetune:
                    seg = dynamics(x_input, u_output)
                    score_tmp = stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]
                    acc_tmp = (score_tmp>=0).float()
                    err_idx = torch.where(acc_tmp<1)[0]
                    if err_idx.shape[0]>0:
                        ft_n = err_idx.shape[0]
                        print(ti, "[Before] Acc=%.2f, %d vehicles do not satisfy STL %s"%(torch.mean(acc_tmp), ft_n, err_idx))
                        u_output_fix = u_output.clone()
                        for iii in range(ft_n):
                            sel_ = err_idx[iii]
                            u_fix = back_solve(x_input[sel_:sel_+1], u_output[sel_:sel_+1], net, stl)
                            u_output_fix[sel_:sel_+1] = u_fix
                        u_output = u_output_fix
                        seg_fix = dynamics(x_input, u_output_fix)
                        acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                        err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                        ft_n_fix = err_idx_fix.shape[0]
                        print(ti, "[After]  Acc=%.2f, %d vehicles do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
            debug_t2=time.time()
            
            nn_idx=0
            for lane_id in lanes:
                lane = lanes[lane_id]
                for veh_id in lane.veh_ids:
                    veh = vehicles[veh_id]
                    if veh_id>=n_vehs:
                        u_output[nn_idx, 0] = other_controller(veh, x_input[nn_idx])
                    nn_idx += 1
            seg = dynamics(x_input, u_output)
            
            nn_idx = 0
            for lane_id in lanes:
                lane = lanes[lane_id]
                for veh_id in lane.veh_ids:
                    veh = vehicles[veh_id]
                    veh.out_a = u_output[nn_idx, 0].item()
                    veh.timer = seg[nn_idx, 0, 3].item()
                    veh.hist = seg[nn_idx].detach().cpu().numpy()
                    veh.v = seg[nn_idx, 0, 1].item()
                    nn_idx += 1
        else:
            debug_t2 = debug_t1 = 0

        # get the info for cars in the intersections
        x_input2 = []
        vid=[]
        for inter_id in inters:
            inter = inters[inter_id]
            for veh_id in inter.veh_ids:
                # TODO write to veh class
                veh = vehicles[veh_id]
                curr_i, curr_j = routes[veh_id][veh.base_i]
                next_i, next_j = routes[veh_id][veh.base_i+1]
                next_inter = inters_2d[next_i][next_j]
                to_id = next_inter.inter_id
                next_lane = lanes_2d[inter_id, to_id]
                x_ego = -(veh.dist + next_lane.lane_length)
                v_ego = veh.v
                i_light = next_inter.is_light
                timer = veh.timer
                x_o = -1
                v_o = 0
                i_tri = 0
                x_ego = np.clip(x_ego, -10, 0)
                print("INTER t=%d|id=%d x:%.2f v:%.2f L:%.1f T:%.1f xo:%.2f vo:%.2f Trigger:%d"%(ti, veh_id, x_ego, v_ego, i_light, timer, x_o, v_o, i_tri))
                x_input2.append([x_ego, v_ego, i_light, timer, x_o, v_o, i_tri])
                vid.append(veh_id)
        
        x_input2 = torch.tensor(x_input2).float().cuda()
        dt_minus2 = 0
        if x_input2.shape[0]>0:
            debug_t3=time.time()
            if args.mpc:
                u_output2 = []
                for iii in range(x_input2.shape[0]):
                    u_output_item = gurobi_solve(ti, x_input2[iii:iii+1])
                    u_output2.append(u_output_item)
                u_output2 = torch.cat(u_output2, dim=0)
            elif args.rl or args.mbpo or args.pets:
                tmp_xs, u_output2, dt_minus2 = get_rl_xs_us(x_input2, rl_policy, args.nt, include_first=True)
            elif args.plan:
                u_output2 = []
                for iii in range(x_input2.shape[0]):
                    u_output_item = solve_planner(ti, x_input2[iii:iii+1], vid[iii])
                    u_output2.append(u_output_item)
                u_output2 = torch.cat(u_output2, dim=0)
            elif args.grad:
                u_output2 = gradient_solve(ti, x_input2, stl)
            else:
                u_output2 = net(x_input2)

                if args.finetune:
                    seg2 = dynamics(x_input2, u_output2)
                    acc_tmp = (stl(seg2, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                    err_idx = torch.where(acc_tmp<1)[0]
                    if err_idx.shape[0]>0:
                        ft_n = err_idx.shape[0]
                        print(ti, "[Before] Acc=%.2f, %d vehicles do not satisfy STL %s"%(torch.mean(acc_tmp), ft_n, err_idx))

                        u_output_fix = u_output2.clone()
                        for iii in range(ft_n):
                            sel_ = err_idx[iii]
                            u_fix = back_solve(x_input2[sel_:sel_+1], u_output2[sel_:sel_+1], net, stl)
                            u_output_fix[sel_:sel_+1] = u_fix
                        u_output2 = u_output_fix
                        seg_fix = dynamics(x_input2, u_output_fix)
                        acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                        err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                        ft_n_fix = err_idx_fix.shape[0]
                        print(ti, "[After]  Acc=%.2f, %d vehicles do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
            
            debug_t4=time.time()

            nn_idx = 0
            for inter_id in inters:
                inter = inters[inter_id]
                for veh_id in inter.veh_ids:
                    veh = vehicles[veh_id]
                    if veh_id>=n_vehs:
                        u_output2[nn_idx, 0] = other_controller(veh, x_input2[nn_idx])
                    nn_idx += 1
            seg2 = dynamics(x_input2, u_output2)
            nn_idx = 0
            for inter_id in inters:
                inter = inters[inter_id]
                for veh_id in inter.veh_ids:
                    veh = vehicles[veh_id]
                    veh.out_a = u_output2[nn_idx, 0].item()
                    veh.timer = seg2[nn_idx, 0, 3].item()
                    veh.hist = seg2[nn_idx].detach().cpu().numpy()
                    veh.v = seg2[nn_idx, 0, 1].item()
                    nn_idx += 1
        
        else:
            debug_t4 = debug_t3 = 0
            seg2 = None

        ### UDPATE INFO
        is_handled = [False for _ in routes]
        for lane_id in lanes:
            lane = lanes[lane_id]
            for veh_id in lane.veh_ids:
                veh = vehicles[veh_id]
                is_handled[veh_id] = True
                ds = veh.v * args.dt
                dx, dy = lane.get_dx_dy(ds)

                veh.xy[0] += dx
                veh.xy[1] += dy 
                veh.dist -= ds

                # NEW REGISTRATION
                if veh.dist < 0:
                    veh.in_id = lane.to_id
                    inter = inters[lane.to_id]
                    new_ds = -veh.dist
                    lane.veh_ids.remove(veh_id)
                    inter.veh_ids.append(veh_id)

                    veh.base_i += 1
                    prev_i, prev_j = routes[veh_id][veh.base_i - 1]
                    curr_i, curr_j = routes[veh_id][veh.base_i]
                    next_i, next_j = routes[veh_id][veh.base_i + 1]
                    lane_type = lane.lane_type
                    direction = get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j)
                    
                    # clean the distance
                    veh.xy = np.array(lane.to_xy)
                    veh.dist = inter.dist(direction)
                    dx, dy = inter.get_dx_dy(veh.dist, new_ds, lane_type, direction)
                    veh.xy[0] += dx
                    veh.xy[1] += dy 
                    veh.dist -= new_ds

                    next_inter = inters_2d[next_i][next_j]
                    veh.is_light = next_inter.is_light
                    if next_inter.is_light:
                        next_lane = lanes_2d[inter.inter_id, next_inter.inter_id]
                        if next_lane.lane_type in [0, 2]:
                            veh.timer = next_inter.timer
                        else:
                            veh.timer = (next_inter.timer + args.phase_red) % args.phase_t
                    else:
                        veh.timer = 0

        for inter_id in inters:
            inter = inters[inter_id]
            for veh_id in inter.veh_ids:
                veh = vehicles[veh_id]
                if is_handled[veh_id]:
                    continue
                is_handled[veh_id] = True
                ds = veh.v * args.dt
                prev_i, prev_j = routes[veh_id][veh.base_i - 1] if veh.base_i!=0 else (None, None)
                curr_i, curr_j = routes[veh_id][veh.base_i]
                next_i, next_j = routes[veh_id][veh.base_i + 1]
                from_id = inters_2d[prev_i][prev_j].inter_id
                to_id = inters_2d[curr_i][curr_j].inter_id
                lane_id = lanes_2d[from_id, to_id].lane_id
                lane_type = lanes[lane_id].lane_type
                direction = get_dir(prev_i, prev_j, curr_i, curr_j, next_i, next_j)
                dx, dy = inter.get_dx_dy(veh.dist, ds, lane_type, direction)
                veh.xy[0] += dx
                veh.xy[1] += dy 
                veh.dist -= ds

                # NEW REGISTRATION
                if veh.dist < 0:
                    next_i, next_j = routes[veh_id][veh.base_i + 1]
                    from_id = veh.in_id
                    to_id = inters_2d[next_i][next_j].inter_id
                    lane = lanes_2d[from_id, to_id]
                    veh.in_id = lane.lane_id
                    inter.veh_ids.remove(veh_id)
                    lane.veh_ids.append(veh_id)

                    # clean the distance
                    ds = -veh.dist
                    veh.xy = np.array(lane.from_xy)
                    veh.dist = lane.dist()
                    dx, dy = lane.get_dx_dy(ds)
                    veh.xy[0] += dx
                    veh.xy[1] += dy 
                    veh.dist -= ds
        
        # EVALUATION
        debug_dt = debug_t4-debug_t3 - dt_minus2 + debug_t2-debug_t1 - dt_minus1
        if seg is None:
            seg_total = seg2
        elif seg2 is None:
            seg_total = seg
        else:
            seg_total = torch.cat([seg, seg2], dim=0)
        
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(car_env.generate_reward_batch(to_np(seg_total[:,0])))
        safety = np.mean(np.logical_or(to_np(seg_total[:, 0, 4]) > 0, to_np(seg_total[:, 0, 4]) ==-1))
        avg_x = np.mean(to_np(seg_total[:,-1, 0]) - to_np(seg_total[:,0, 0]))
        avg_v = np.mean(to_np(seg_total[:,0, 1]))

        metrics["t"].append(debug_dt)
        metrics["safety"].append(safety)
        metrics["avg_x"].append(avg_x)
        metrics["avg_v"].append(avg_v)
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)

        # car-map visualization
        if args.no_viz==False:
            plt.figure(figsize=(12, 9))
            plt.subplot(1, 2, 1)
            ax = plt.gca()
            light_r = 0.8
            stop_r = 0.8
            for lane_id in lanes:
                alpha=1
                lane = lanes[lane_id]
                rect = Rectangle(lane.viz_xy, lane.viz_w, lane.viz_h, color="gray", alpha=0.5, zorder=10)
                ax.add_patch(rect)
                to_inter = inters[lane.to_id]
                _w = stop_r*5 if lane.lane_type in [0,2] else 0.5*stop_r
                _h = stop_r*5 if lane.lane_type in [1,3] else 0.5*stop_r
                if to_inter.is_light:
                    if lane.lane_type in [0, 2]:
                        if to_inter.timer % args.phase_t < args.phase_red:
                            light_color = "red"
                        else:
                            light_color = "green"
                    else:
                        if to_inter.timer % args.phase_t < args.phase_red:
                            light_color = "green"
                        else:
                            light_color = "red"
                    obj = Ellipse(alpha*lane.to_xy + (1-alpha) * to_inter.xy, _w, _h, color=light_color, zorder=25)
                else:
                    obj = Ellipse(lane.to_xy, _w, _h, color="black", zorder=25)
                ax.add_patch(obj)

            for inter_id in inters:
                inter = inters[inter_id]
                rect = Rectangle(inter.viz_xy, inter.viz_w, inter.viz_h, color="lightgray", alpha=0.5, zorder=10)
                ax.add_patch(rect)

            for veh_id in vehicles:
                veh = vehicles[veh_id]
                color = "royalblue" if veh_id>=n_vehs else "brown"
                label = None
                if veh_id == 0:
                    label = "our cars"
                if veh_id == n_vehs:
                    label = "other cars"
                rect = Rectangle(veh.xy-np.array([CAR_W/2,CAR_W/2]), CAR_W, CAR_W, color=color, alpha=0.9, zorder=50, label=label)
                ax.add_patch(rect)
                plt.text(veh.xy[0]-CAR_W/2, veh.xy[1]-CAR_W/2, veh_id, fontsize=10, zorder=100)
            
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.axis("scaled")
            plt.legend(loc="upper center", fontsize=14, ncol=2, bbox_to_anchor=(0.5, 1.1))
            plt.subplot(1, 2, 2)
            for veh_id in vehicles:
                veh = vehicles[veh_id]
                plt.plot(range(args.nt), veh.hist[..., 0], color="blue")
            plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
            plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")
            plt.ylim(-10, 20)
            plt.suptitle("Simulation (%04d/%04d)" % (ti, nt))
            filename = "%s/t_%03d.png"%(args.viz_dir, ti)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
            fs_list.append(filename)
            plt.close()
        print(ti)

        # update the intersection timers
        for inter_i in inters:
            inters[inter_i].update()
        
        # save the trajectories
    print("Acc:%.3f"%(np.mean(np.array(metrics["acc"]))))
    eval_proc(metrics, "e1_car", args)

    if args.no_viz==False:
        os.makedirs('%s/animation'%(args.viz_dir), exist_ok=True)
        generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)
    return


def heading_base_sampling(set0_ve):
    relu = nn.ReLU()
    n = set0_ve.shape[0]
    set00_xo = uniform_tensor(-1, -1, (n//2, 1))
    set00_vo = uniform_tensor(0, 0, (n//2, 1))
    set01_xo = uniform_tensor(args.safe_thres, args.xo_max, (n//2, 1))
    lower = torch.sqrt(relu((args.safe_thres - set01_xo)*args.amax*2 + set0_ve[n//2:]**2))
    set01_vo = uniform_tensor(0, 1, (n//2, 1)) * (args.vmax-lower) + lower

    return torch.cat([set00_xo, set01_xo], dim=0), torch.cat([set00_vo, set01_vo], dim=0)


def initialize_x(N):
    # generate initial points
    # set-0
    # x ~ [-10, 0]
    # v ~ [] make sure v^2/2a < |x|
    # t ~ 0
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
            set0_xo, set0_vo = heading_base_sampling(set0_ve)
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
            set2_xo, set2_vo = heading_base_sampling(set2_ve)
    else:
        set2_xo = uniform_tensor(0, 0, (n2, 1))
        set2_vo = uniform_tensor(0, 0, (n2, 1))
    set2_tri = uniform_tensor(0, 0, (n2, 1))

    set0 = torch.cat([set0_xe, set0_ve, set0_id, set0_t, set0_xo, set0_vo, set0_tri], dim=-1)
    set1 = torch.cat([set1_xe, set1_ve, set1_id, set1_t, set1_xo, set1_vo, set1_tri], dim=-1)
    set2 = torch.cat([set2_xe, set2_ve, set2_id, set2_t, set2_xo, set2_vo, set2_tri], dim=-1)
    
    x_init = torch.cat([set0, set1, set2], dim=0).float().cuda()

    return x_init

def back_solve(x_input, output_0, policy, stl):
    nt0 = 4
    k = 20
    nt_k_list=[
        (4, 20),
        ]

    best_score=-100
    best_u_merged=None

    for nt0, k in nt_k_list:
        actions = np.linspace(-args.amax, args.amax, k)
        li = [actions] * nt0
        a_seq = np.array(np.meshgrid(*li)).T.reshape(-1, nt0)
        a_seq = torch.from_numpy(a_seq).float().cuda()

        a_seq = torch.cat([a_seq, output_0[:, :nt0]], dim=0)

        x_input_mul = torch.tile(x_input, [a_seq.shape[0], 1])
        seg0 = dynamics(x_input_mul, a_seq, include_first=False)
        x_new = seg0[:, -1]
        
        u_output = policy(x_new).detach()
        seg1 = dynamics(x_new, u_output[:, :args.nt-nt0], include_first=False)
        
        seg = torch.cat([seg0, seg1], dim=1)
        score = stl(seg, args.smoothing_factor)[:, :1]
        idx = torch.argmax(score, dim=0).item()
        
        max_score = score[idx]

        u_merged = torch.cat([a_seq[idx, :],u_output[idx, :args.nt-nt0]], dim=0).unsqueeze(0)
        if best_score<max_score.item():
            best_u_merged = u_merged
            best_score = max_score.item()
    return best_u_merged

def test_backup(policy, rl_policy, stl):
    "x:-10.00 v:0.93 L:0.0 T:0.0 xo:-1.00 vo:0.00 Trigger:0"
    x_input = torch.tensor([[-10.00, 0.93, 0.0, 0.0, -1.00, 0.00, 0]]).float().cuda()
    
    if rl_policy is not None:
        seg, u_output = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=False)
    else:
        u_output = policy(x_input)
        u_output = back_solve(x_input, u_output, policy, stl)
        seg = dynamics(x_input, u_output, include_first=True)  #, include_first=args.include_first)
    
    score = stl(seg, args.smoothing_factor)[:, :1]
    score_avg = torch.mean(score).item()
    acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
    acc_avg = torch.mean(acc).item()
    for ti in range(args.nt+1):
        x_ego, v_ego, i_light, timer, x_o, v_o, i_tri = seg[0, ti].detach()
        if ti == args.nt:
            uu = -100
        else:
            uu = u_output[0, ti].detach().item()
        print("T=%d|x:%.2f v:%.2f L:%.1f T:%.1f xo:%.2f vo:%.2f Trigger:%d || U:%.2f"%(
            ti, x_ego, v_ego, i_light, timer, x_o, v_o, i_tri, uu))
    print("="*40)
    print("score:%.3f  acc:%.3f" % (score_avg, acc_avg))

def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        state_dict = torch.load(utils.find_path(args.net_pretrained_path))
        net.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    cond1 = Eventually(0, args.nt, AP(lambda x: x[..., 3] - args.stop_t, comment="t_stop>=%.1fs"%(args.stop_t)))   
    if args.triggered:
        cond_sig = Imply(AP(lambda x: x[..., 6]-0.5, "Triggered"), Always(0,args.nt, AP(lambda x:args.stop_x*0.5 - x[..., 0], "stop")))
        cond1 = And(cond1, cond_sig)
    cond2 = Always(0, args.nt, 
                Not(And(AP(lambda x: args.phase_red - x[...,3], comment="t=red"),
                        AP(lambda x: -x[..., 0] * (x[..., 0]-args.traffic_x), comment="inside intersection")
                )))
    
    cond3 = Always(0, args.nt, AP(lambda x:x[..., 4]-args.safe_thres,comment="heading>0"))

    if args.heading:
        stl = ListAnd([
            Imply(AP(lambda x: 0.5-x[..., 2], comment="I=stop"), cond1),  # stop signal condition
            Imply(AP(lambda x: x[...,2]-0.5, comment="I=light"), cond2),  # light signal condition
            Imply(AP(lambda x: x[..., 4]+0.5, comment="heading"), cond3)  # heading condition
            ])
    else:
        stl = ListAnd([
            Imply(AP(lambda x: 0.5-x[..., 2], comment="I=stop"), cond1), 
            Imply(AP(lambda x: x[...,2]-0.5, comment="I=light"), cond2)
            ])

    print(stl)
    stl.update_format("word")
    print(stl)

    if args.backup:
        if args.rl:
            from stable_baselines3 import SAC, PPO, A2C
            rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
            net = None
        else:
            rl_policy=None
        test_backup(net, rl_policy, stl)
        return

    if args.test:
        rl_policy = None
        if args.rl:
            from stable_baselines3 import SAC, PPO, A2C
            rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
        elif args.mbpo or args.pets:
            rl_policy = get_mbrl_models(get_exp_dir()+"/"+args.rl_path, args, args.mbpo)
        sim_multi(net, rl_policy, stl)
        return

    relu = nn.ReLU()
    x_init = initialize_x(args.num_samples)

    if args.add_val:
        x_init_val = initialize_x(args.num_samples//10)

    n = args.num_samples // 4
    n2 = 2 * n

    for epi in range(args.epochs):
        eta.update()
        if args.update_init_freq > 0 and epi % args.update_init_freq == 0 and epi != 0:
             x_init = initialize_x(args.num_samples)

        x0 = x_init.detach()
        u = net(x0)
        seg = dynamics(x0, u, include_first=args.include_first)
        
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc)
        acc_stop_avg = torch.mean(acc[:n2])
        acc_light_avg = torch.mean(acc[n2:])

        pass_avg = torch.mean((seg[:args.num_samples//2, -1, 0]+seg[:args.num_samples//2, -1, -1] > 0).float())

        stl_loss = torch.mean(relu(0.5-score))
        green_mask = (torch.logical_or(seg[:, :, 2]==0, seg[:, :, 3] > args.phase_red)).float()
        v_loss = torch.mean(acc * torch.relu(5-seg[:,:,1]) * green_mask) * args.v_loss
        s_loss = torch.mean(acc * torch.relu(args.traffic_x-seg[:,:,0])) * args.s_loss
        loss = stl_loss + v_loss + s_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epi % args.print_freq == 0:
            acc_avg_val = torch.tensor(0.0)
            if args.add_val:
                u_val = net(x_init_val.detach())
                seg_val = dynamics(x_init_val, u_val, include_first=args.include_first)
                acc_val = (stl(seg_val, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_avg_val = torch.mean(acc_val)

            print("%s| %03d  loss:%.3f  stl:%.3f  vloss:%.3f sloss:%.3f acc:%.3f (%.3f | %.3f) acc_val:%.3f pass:%.3f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), stl_loss.item(), v_loss.item(), s_loss.item(), acc_avg.item(), acc_stop_avg.item(), acc_light_avg.item(), 
                acc_avg_val.item(),
                pass_avg.item(),
                eta.interval_str(), eta.elapsed_str(), eta.eta_str()
                ))
        
        # Save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))
        
        if epi % args.viz_freq == 0:
            seg_np = to_np(seg)
            acc_np = to_np(acc)
            t_len = seg_np.shape[1]
            N = args.num_samples
            
            linestyle = lambda x: ("-" if x[0, 6] == 0 else "-.") if args.triggered else "-"
            cond_color = lambda x: "green" if x[0]>0 else "red"
            if args.heading:
                plt.figure(figsize=(12, 8))
                nv = 25
                # plot the stop sign curve
                # plot the non-heading case
                # plot the ego-x curve
                # plot the timer curve
                # plot the lead_x curve
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

                # plot the heading case
                # plot the ego-x curve
                # plot the timer curve
                # plot the lead_x curve
                ls = "-."
                plt.subplot(3, 2, 2)
                for i in range(N//8, N//8+nv):
                    plt.plot(range(t_len), seg_np[i, :, 0], linestyle=ls, color=cond_color(acc_np[i]))
                plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
                plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

                plt.subplot(3, 2, 4)
                for i in range(N//8, N//8+nv):
                    plt.plot(range(t_len), seg_np[i, :, 3], linestyle=ls, color=cond_color(acc_np[i]))

                plt.subplot(3, 2, 6)
                for i in range(N//8, N//8+nv):
                    plt.plot(range(t_len), seg_np[i, :, 4], linestyle=ls, color=cond_color(acc_np[i]))

            else:
                plt.figure(figsize=(8, 8))
                # plot the stop sign curve
                # plot the ego-x curve
                plt.subplot(2, 1, 1)
                for i in range(10):
                    plt.plot(range(t_len), seg_np[i, :, 0], color=cond_color(acc_np[i]))
                for j in range(N//2-10, N//2):
                    plt.plot(range(t_len), seg_np[j, :, 0], color=cond_color(acc_np[j]))
                plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
                plt.axhline(y=args.stop_x, xmin=0, xmax=args.nt, color="gray")

                # plot the timer curve
                plt.subplot(2, 1, 2)
                for i in range(50):
                    plt.plot(range(t_len), seg_np[i, :, 3], color=cond_color(acc_np[i]))
                for j in range(N//2-50, N//2):
                    plt.plot(range(t_len), seg_np[j, :, 3], color=cond_color(acc_np[j]))                

            plt.savefig("%s/stopsign_iter_%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
            plt.close()

            plt.figure(figsize=(8, 8))
            seg_np = seg_np[n2:]
            acc_np = acc_np[n2:]
            plt.subplot(2, 1, 1)
            for i in range(10):
                for j in range(args.nt-1):
                    plt.plot([j, j+1], [seg_np[i, j, 0], seg_np[i, j+1, 0]], 
                             color="red" if seg_np[i, j, 3] <= args.phase_red else "green")
            plt.axhline(y=args.traffic_x, xmin=0, xmax=args.nt, color="gray")
            plt.axhline(y=0, xmin=0, xmax=args.nt, color="gray")
            plt.axis("scaled")
            if args.heading:
                plt.subplot(2, 1, 2)
                for i in range(args.num_samples//2-50, args.num_samples//2):
                    plt.plot(range(seg_np.shape[1]), seg_np[i, :, 4], linestyle=linestyle(seg_np[i]), color="green" if acc_np[i,0]>0 else "red")

            plt.savefig("%s/light_iter_%05d.png"%(args.viz_dir, epi), bbox_inches='tight', pad_inches=0.1)
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default="traffic")
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=3e-5)
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
    add("--hiddens", type=int, nargs="+", default=[64, 64, 64])
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=1.0)
    add("--update_init_freq", type=int, default=-1)
    add("--add_val", action="store_true", default=False)
    add("--include_first", action="store_true", default=False)

    add("--mpc", action="store_true", default=False)
    add("--plan", action="store_true", default=False)
    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)
    add("--rl", action="store_true", default=False)
    add("--rl_stl", action="store_true", default=False)
    add("--rl_acc", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)

    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)

    add("--eval_path", type=str, default="eval_result")

    add("--finetune", action="store_true", default=False)
    add("--backup", action='store_true', default=False)
    add("--cem", action='store_true', default=False)
    args = parser.parse_args()
    args.triggered=True
    args.heading=True

    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))

