from lib_stl_core import *
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.size': 12})

import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, \
            check_pts_collision, check_seg_collision, soft_step, to_torch, \
            pts_in_poly, seg_int_poly, build_relu_nn, soft_step_hard, get_exp_dir, \
            eval_proc, xxyy_2_Ab
from lib_cem import solve_cem_func
from utils_mbrl import get_mbrl_models, get_mbrl_u


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        # input  (rover xy; dest xy; charger xy; battery t; hold_t)
        # output (rover v theta; astro v theta)
        self.net = build_relu_nn( 2 + 2 + 2 + 2, 2 * args.nt, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        u = self.net(x).reshape(num_samples, args.nt, -1)
        if self.args.no_tanh:
            u0 = torch.clip(u[..., 0], 0, 1)
            u1 = torch.clip(u[..., 1], -np.pi, np.pi)
        else:
            u0 = torch.tanh(u[..., 0]) * 0.5 + 0.5
            u1 = torch.tanh(u[..., 1]) * np.pi
        uu = torch.stack([u0, u1], dim=-1)
        return uu

def dynamics(x0, u, include_first=False):
    # input:  x0, y0, x1, y1, x2, y2, T, hold_t
    # input:  u, (n, T)
    # return: s, (n, T, 9)
    
    t = u.shape[1]
    x = x0.clone()
    if include_first:
        segs=[x0]
    else:
        segs = []
    for ti in range(t):
        new_x = dynamics_per_step(x, u[:, ti])
        segs.append(new_x)
        x = new_x

    return torch.stack(segs, dim=1)

def dynamics_per_step(x, u):
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


def get_rl_xs_us(x, policy, nt, include_first=False):
    xs = []
    us = []
    if include_first:
        xs.append(x)
    dt_minus = 0
    for ti in range(nt):
        tt1=time.time()        
        if args.rl:
            u, _ = policy.predict(x.cpu(), deterministic=True)
            u = torch.from_numpy(u)
        else:
            if args.mbpo:
                u = get_mbrl_u(x, None, policy, mbpo=True)
            elif args.pets:
                u_list=[]
                for iii in range(x.shape[0]):
                    u = get_mbrl_u(x[iii], None, policy, mbpo=False)
                    u_list.append(u)
                u = torch.stack(u_list, dim=0)

        u[..., 0] = torch.clip((u[..., 0] + 1)/2, 0, 1)
        u[..., 1] = torch.clip(u[..., 1] * np.pi, -np.pi, np.pi)
        u = u.cuda()
        
        new_x = dynamics_per_step(x, u)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)  # (N, 2) -> (N, T, 2)
    return xs, us, dt_minus


def initialize_x_cycle(n):
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

def initialize_x(n, objs, test=False):
    x_list = []
    total_n = 0
    while(total_n<n):
        x_init = initialize_x_cycle(n)
        valids = []
        for obj_i, obj in enumerate(objs):
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


def in_poly(xy0, xy1, poly):
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


def max_dist_init(xys, n):
    n_trials = 1000
    N = xys.shape[0]
    max_dist=-1
    for i in range(n_trials):
        choice = np.random.choice(N, n)
        _xy = xys[choice]
        dist = torch.mean(torch.norm(_xy[:, None] - _xy[None, :], dim=-1))
        if dist > max_dist:
            max_dist = dist
            max_choice = choice
    return xys[max_choice]

def solve_planner(x_init, objs_np, tti, multi_test=True, need_to_find=True):
    from lib_pwlplan import plan, Node
    from gurobipy import GRB
    tt1=time.time()
    def func1(m, PWLs, di):
        u = m.addVars(args.nt, 2, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the_u")
        for i in range(args.nt):
            m.addConstr(PWLs[0][i+1][1] - PWLs[0][i][1] == args.dt)
            m.addConstr(u[i,0]*u[i,0] + u[i,1]*u[i,1] >= (1*args.rover_vmin)**2)
            m.addConstr(u[i,0]*u[i,0] + u[i,1]*u[i,1] <= (1*args.rover_vmax)**2)
            m.addConstr(PWLs[0][i+1][0][0] == PWLs[0][i][0][0] + u[i,0] * args.dt)
            m.addConstr(PWLs[0][i+1][0][1] == PWLs[0][i][0][1] + u[i,1] * args.dt)
    
    nt = args.nt
    dt = args.dt
    vmax = args.rover_vmax

    Abs = []
    for obj in objs_np:
        obj_xxyy = [np.min(obj[:, 0]), np.max(obj[:, 0]), np.min(obj[:, 1]), np.max(obj[:, 1])]
        A0, b0 = xxyy_2_Ab(obj_xxyy)
        Abs.append([A0, b0])

    R = args.close_thres
    dw = R / 4
    dest_x = x_init[0, 2].item()
    dest_y = x_init[0, 3].item()
    dest_left = dest_x - dw
    dest_right = dest_x + dw
    dest_up = dest_y + dw
    dest_down = dest_y - dw
    dest_Ab = xxyy_2_Ab([dest_left, dest_right, dest_down, dest_up]) 
    
    station_Abs = []
    for i in range(args.num_stations):
        R = args.close_thres
        dw = R / 4
        station_x = x_init[i, 4].item()
        station_y = x_init[i, 5].item()
        station_left = station_x - dw
        station_right = station_x + dw
        station_down = station_y - dw
        station_up = station_y + dw
        station_Ab = xxyy_2_Ab([station_left, station_right, station_down, station_up])
        station_Abs.append(station_Ab)
    
    tmax = (nt + 1) * dt
    
    # avoid obstacles
    avoids = []
    for Ai, bi in Abs[1:]:
        avoids.append(Node('negmu', info={'A':Ai, 'b':bi}))
    always_avoid_obs = Node('A', deps=[Node('and', deps=avoids)], info={'int':[0, tmax]})
    
    # keep battery high
    # reach station (which one)
    # hold conditions
    # reach goal (if possible?)
    # if battery < thres, go to charge and stay there for a few sec
    # if battery > thres and hold <=0 , go to dest
    battery = x_init[0, 6].item()
    battery_limit = args.dt * args.nt
    t_val = battery
    stay_time = args.hold_t * args.nt
    stay_timer = x_init[0, 7].item()
    
    in_map = Node('mu', info={'A':Abs[0][0], 'b':Abs[0][1]})
    always_in_map = Node('A', deps=[in_map], info={'int': [0, tmax]})
    at_dest = Node('mu', info={'A':dest_Ab[0], 'b':dest_Ab[1]})
    at_stations = [Node('mu', info={'A':station_Abs[i][0], 'b':station_Abs[i][1]}) for i in range(args.num_stations)]

    station_x = x_init[0, 4].item()
    station_y = x_init[0, 5].item()
    if need_to_find:
        station_i = find_station(x_init, None)
    else:
        station_i = 0

    if battery > battery_limit:
        eventually_go_to_dest = Node('F', deps=[at_dest,], info={'int':[0, tmax]})
        if stay_timer > 0 and station_i is not None:
            first_stay = Node('A', deps=[at_stations[station_i]],info={'int':[0, stay_timer]})
            conditions = [Node('and', deps=[first_stay, eventually_go_to_dest])]
            conditions = [first_stay, eventually_go_to_dest]
        else:
            conditions = [eventually_go_to_dest]
    else:
        go_to_stations_and_stay = [
            Node('F', deps=[
                Node('A', deps=[at_stations[i]], info={'int':[0, min(stay_time, battery)]})
                # at_stations[i]
            ], info={'int':[0, battery]}) for i in range(args.num_stations)
        ]
        go_one_station_stay = Node('or', deps=go_to_stations_and_stay)
        conditions = [go_one_station_stay]

    specs = [in_map, always_avoid_obs] + conditions
    specs = [Node("and", deps=specs)]
    x0s = [np.array([x_init[0,0].item(), x_init[0, 1].item()])]
    
    PWL, u_out = plan(x0s, specs, bloat=0.01, MIPGap=0.05, num_segs=args.nt, tmax=tmax, vmax=vmax, extra_fn_list=[func1], return_u=True, quiet=False)
    if PWL[0] is not None:
        for i in range(args.nt+1):
            print("%d x:%.3f y:%.3f t:%.2f"%(
                i, PWL[0][i][0][0], PWL[0][i][0][1], PWL[0][i][1],))
        if need_to_find:
            station_i = find_station(x_init, [PWL[0][i][0] for i in range(args.nt)])
        else:
            station_i = 0
    else:
        u_out = np.zeros((args.nt, 2))
    tt2 = time.time()

    u_original = np.array(u_out)
    u_original[:, 0] = np.linalg.norm(u_out, axis=-1)
    u_original[:, 0] = (u_original[:, 0] - args.rover_vmin) / (args.rover_vmax - args.rover_vmin)
    u_original[:, 1] = np.arctan2(u_out[:, 1], u_out[:,0])

    if station_i is None:
        station_i = 0

    return torch.from_numpy(u_original[None]), station_i


def find_station(x_init, segs):
    max_i = None
    for i in range(args.num_stations):
        if torch.norm(x_init[i,0:2]-x_init[i, 4:6])<=args.close_thres:
            return i
    if segs is not None:
        for ti in range(args.nt):
            for i in range(args.num_stations):
                if (segs[ti][0]-x_init[i, 4])**2 + (segs[ti][1]-x_init[i,5])**2 <= args.close_thres**2:
                    return i
    return max_i



def solve_gurobi(x_init, objs_np, tti, multi_test=False):
    import gurobipy as gp
    from gurobipy import GRB
    x_init = to_np(x_init)
    # TODO
    nt = args.nt
    dt = args.dt
    M = 10^6
    XMAX = 10
    XMIN = 0
    YMAX = 10
    YMIN = 0
    VMAX = args.rover_vmax
    VMIN = args.rover_vmin
    R = args.close_thres
    num_stations = args.num_stations
    if multi_test:
        cx_list = [x_init[i, 4] for i in range(num_stations)]
        cy_list = [x_init[i, 5] for i in range(num_stations)]
    else:
        cx = x_init[0, 4]
        cy = x_init[0, 5]
    dx = x_init[0, 2]
    dy = x_init[0, 3]

    timeLimit = 5

    obs0 = [np.min(objs_np[1][:,0]), np.max(objs_np[1][:,0]),
            np.min(objs_np[1][:,1]), np.max(objs_np[1][:,1]),
            ]  # xmin, xmax, ymin, ymax
    obs1 = [np.min(objs_np[2][:,0]), np.max(objs_np[2][:,0]),
            np.min(objs_np[2][:,1]), np.max(objs_np[2][:,1]),
            ]  # xmin, xmax, ymin, ymax
    obs2 = [np.min(objs_np[3][:,0]), np.max(objs_np[3][:,0]),
            np.min(objs_np[3][:,1]), np.max(objs_np[3][:,1]),
            ]  # xmin, xmax, ymin, ymax
    obs3 = [np.min(objs_np[4][:,0]), np.max(objs_np[4][:,0]),
            np.min(objs_np[4][:,1]), np.max(objs_np[4][:,1]),
            ]  # xmin, xmax, ymin, ymax
    obses = [obs0, obs1, obs2, obs3]

    t_start=time.time()
    m = gp.Model("mip1")
    m.setParam( 'OutputFlag', False)

    x = m.addVars(nt+1, name="x", lb=float('-inf'), ub=float('inf'))
    y = m.addVars(nt+1, name="y", lb=float('-inf'), ub=float('inf'))
    battery = m.addVars(nt+1, name="battery", lb=float('-inf'), ub=float('inf'))
    hold = m.addVars(nt+1, name="hold", lb=float('-inf'), ub=float('inf'))
    vx = m.addVars(nt, name="vx", lb=float('-inf'), ub=float('inf'))
    vy = m.addVars(nt, name="vy", lb=float('-inf'), ub=float('inf'))
    reached = m.addVars(nt, name="reached", vtype=GRB.BINARY)  # reached the charge station
    chase = m.addVars(nt, name="chase", vtype=GRB.BINARY)
    if multi_test:
        select = m.addVars(num_stations, name="select", vtype=GRB.BINARY)

    gamma_g0 = m.addVars(nt+1, 5, name="gamma_g0", lb=float('-inf'), ub=float('inf'))

    # obstacles
    I0 = m.addVars(nt, 4, vtype=GRB.BINARY, name="I0")
    I1 = m.addVars(nt, 4, vtype=GRB.BINARY, name="I1")
    I2 = m.addVars(nt, 4, vtype=GRB.BINARY, name="I2")
    I3 = m.addVars(nt, 4, vtype=GRB.BINARY, name="I3")
    IS = [I0, I1, I2, I3]

    # initial setup
    m.addConstr(x[0] == x_init[0, 0])
    m.addConstr(y[0] == x_init[0, 1])
    m.addConstr(battery[0] == x_init[0, 6])
    if (x_init[0, 0]-x_init[0, 4])**2+(x_init[0,1]-x_init[0,5])**2>R**2:
        x_init[0, 7] = -1
    m.addConstr(hold[0] == x_init[0, 7])
    print(x_init[0])
    
    # constraints
    for ti in range(nt):
        # region
        m.addConstr(x[ti+1]<= XMAX)
        m.addConstr(x[ti+1]>= XMIN)
        m.addConstr(y[ti+1]<= YMAX)
        m.addConstr(y[ti+1]>= YMIN)

        # control output
        m.addConstr(vx[ti]<= VMAX)
        m.addConstr(vx[ti]>= -VMAX)
        m.addConstr(vy[ti]<= VMAX)
        m.addConstr(vy[ti]>= -VMAX)
        m.addConstr(vx[ti]**2+vy[ti]**2<= VMAX**2)

        # dynamics
        m.addConstr(x[ti+1] == x[ti] + vx[ti] * dt)
        m.addConstr(y[ti+1] == y[ti] + vy[ti] * dt)
        m.addConstr(battery[ti+1] == 10 * reached[ti] - (1-reached[ti]) * args.dt)
        if ti==0:
            m.addConstr(hold[ti+1] == hold[ti] - reached[ti] * args.dt)
        else:
            m.addConstr(hold[ti+1] == hold[ti] - reached[ti] * args.dt + (reached[ti]-reached[ti-1]) * reached[ti] * args.hold_t * args.dt)

        # battery condition
        m.addConstr(battery[ti+1] >= 0 - gamma_g0[ti, 4])

        # charger station (select one of them)
        _r = R/4
        if multi_test:
            _cx = gp.quicksum(cx_list[ii]*select[ii] for ii in range(num_stations))
            _cy = gp.quicksum(cy_list[ii]*select[ii] for ii in range(num_stations))
            m.addConstr(x[ti+1] <= _cx + _r + (1-reached[ti]) * M)
            m.addConstr(x[ti+1] >= _cx - _r - (1-reached[ti]) * M)
            m.addConstr(y[ti+1] <= _cy + _r + (1-reached[ti]) * M)
            m.addConstr(y[ti+1] >= _cy - _r - (1-reached[ti]) * M)
        else:
            m.addConstr(x[ti+1] <= cx + _r + (1-reached[ti]) * M)
            m.addConstr(x[ti+1] >= cx - _r - (1-reached[ti]) * M)
            m.addConstr(y[ti+1] <= cy + _r + (1-reached[ti]) * M)
            m.addConstr(y[ti+1] >= cy - _r - (1-reached[ti]) * M)

        # stay constraint       # can't be reached == 0 and hold>0
        m.addConstr(reached[ti] * M >= hold[ti])

        # obstacles
        n_res = 10
        bloat = 0.2
        for j in range(4):
            for alpha_i in range(n_res):
                alpha = (alpha_i+1) / n_res
                m.addConstr(x[ti+1] * alpha + x[ti] * (1-alpha) <= obses[j][0] - bloat + IS[j][ti, 0] * M + gamma_g0[ti,0])
                m.addConstr(x[ti+1] * alpha + x[ti] * (1-alpha) >= obses[j][1] + bloat - IS[j][ti, 1] * M + gamma_g0[ti,1])
                m.addConstr(y[ti+1] * alpha + y[ti] * (1-alpha) <= obses[j][2] - bloat + IS[j][ti, 2] * M + gamma_g0[ti,2])
                m.addConstr(y[ti+1] * alpha + y[ti] * (1-alpha) >= obses[j][3] + bloat - IS[j][ti, 3] * M + gamma_g0[ti,3])  
            m.addConstr(gp.quicksum(IS[j][ti, i] for i in range(4)) <= 3)

        # destination constraint
        m.addConstr(x[ti+1] <= dx + _r + (1-chase[ti]) * M)
        m.addConstr(x[ti+1] >= dx - _r - (1-chase[ti]) * M)
        m.addConstr(y[ti+1] <= dy + _r + (1-chase[ti]) * M)
        m.addConstr(y[ti+1] >= dy - _r - (1-chase[ti]) * M)

    battery_limit = args.dt*args.nt
    if x_init[0, 6] >= battery_limit:  # must go to the dest
        m.addConstr(gp.quicksum(chase) >= 1)

    sum_u = gp.quicksum(vx[i]*vx[i] for i in range(nt)) + gp.quicksum(vy[i]*vy[i] for i in range(nt))
    sum_g1 = gp.quicksum(gamma_g0[i,j]*gamma_g0[i,j] for i in range(nt) for j in range(5))
    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)

    # choose one of the stations
    if multi_test:
        m.addConstr(gp.quicksum(select) == 1)

    # objective function
    m.setObjective(M * 1000 * (sum_g1) + 0.1*sum_u, GRB.MINIMIZE)
    m.optimize()

    t_end=time.time()
    print("ti-%04d t= %.4f sec"%(tti, t_end-t_start))
    u_torch=[]
    for ti in range(nt):
        v_i = ((vx[ti].X**2+vy[ti].X**2)**0.5-args.rover_vmin) / (args.rover_vmax-args.rover_vmin)
        th_i = np.arctan2(vy[ti].X, vx[ti].X)
        u_torch.append([v_i, th_i])
    u_torch=torch.tensor(u_torch).float().unsqueeze(0)
    if multi_test:
        select_i_np = np.array([select[ii].X for ii in range(num_stations)])
        cands = np.where(select_i_np==1)[0]
        if len(cands) >0:
            select_i = cands[0]
        else:
            select_i = 0
        return u_torch, select_i
    else:
        return u_torch


def gradient_solve(tti, x_init, stl, multi_test=False):
    relu = torch.nn.ReLU()
    u_lat = torch.zeros(x_init.shape[0], args.nt, 2).cuda().requires_grad_()
    optimizer = torch.optim.Adam([u_lat], lr=args.grad_lr)
    tt1=time.time()
    prev_loss = None
    for i in range(args.grad_steps):
        u0 = (torch.tanh(u_lat[..., 0]) + 1) / 2
        u1 = torch.tanh(u_lat[..., 1]) * np.pi
        u = torch.stack([u0, u1], dim=-1)
        seg = dynamics(x_init, u, include_first=True)
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        
        battery_limit = args.dt*args.nt
        shall_charge = (x_init[..., 6:7] <= battery_limit).float()
        acc_mask = acc
        if args.no_acc_mask:
            acc_mask = 1
        charge_dist = torch.norm(seg[:, :, :2] - x_init.unsqueeze(dim=1)[:, :, 4:6], dim=-1)
        dest_dist = torch.norm(seg[:, :, :2] - x_init.unsqueeze(dim=1)[:, :, 2:4], dim=-1)
        dist_loss = torch.mean((charge_dist * shall_charge + dest_dist * (1 - shall_charge)) * acc_mask)
        dist_loss = dist_loss * args.dist_w

        loss = torch.mean(relu(0.5-score)) + dist_loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if i>0:
            if abs(loss.item())<1e-5 or abs(prev_loss.item()-loss.item())<1e-5:
                break
        if i % (args.grad_steps//5) == 0 or i==-1:
            print(i, loss.item())
        prev_loss = loss.detach()
    tt2=time.time()
    print("%05d t:%.4f seconds"%(tti, tt2-tt1))
    if multi_test:
        return u.detach(), score.squeeze(-1)
    else:
        return u.detach()

def solve_cem(ti, x_input, stl, args):
    def dynamics_step_func(x, u):
        return dynamics_per_step(x, u)
    
    def reward_func(trajs):
        return stl(trajs, args.smoothing_factor, d={"hard":True})[:, 0]

    u_min = torch.tensor([0, -np.pi]).cuda()
    u_max = torch.tensor([1, np.pi]).cuda()
    us = []
    stl_scores = []
    for i in range(x_input.shape[0]):
        u, _, info = solve_cem_func(
            x_input[i], state_dim=x_input.shape[-1], nt=args.nt, action_dim=2,
            num_iters=500, n_cand=10000, n_elites=100, policy_type="mlp_16",
            dynamics_step_func=dynamics_step_func, reward_func=reward_func,
            transform=None, u_clip=(u_min, u_max), seed=None, args=None, 
            extra_dict=None, quiet=False, device="gpu", visualize=False
        )
        us.append(u)
        stl_scores.append(info["best_reward"])
    return torch.stack(us, dim=0), torch.stack(stl_scores, dim=0)  # (nt, 7)


def test_mars(net, rl_policy, stl, objs_np, objs):
    metrics_str=["acc", "reward", "score", "t", "safety", "battery", "distance", "goals"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.rover_env import RoverEnv
    the_env = RoverEnv(args)

    nt = 200
    n_astros = 100
    update_freq = args.mpc_update_freq
    debug_t1 = time.time()

    x_init = initialize_x(n_astros, objs, test=True)
    # pick the first rover, charge station, battery time, and
    # rolling for all sampled astro_xy and dest_xy
    # (rover_xy, astro_xy, dest_xy, charge_xy, battery_T)
    num_stations = args.num_stations
    x = x_init[0:1]
    if args.multi_test:
        tmp_map = max_dist_init(x_init[:, 4:6], num_stations)
        x[:, 4:6] = tmp_map[0:1]

    fs_list = []
    history = []
    seg_list = []
    astro_i = 0
    close_enough_dist = args.close_thres

    # statistics
    stl_success = 0
    safety = 0
    battery_ok = 0
    move_distance = 0
    arrived_charger = False
    prev_arrived = False
    cnt = 0
    if args.multi_test:
        map_list = []
        marker_list = []
        stl_max_i = None
    for ti in range(nt):
        # if update
        if cnt % update_freq == 0 or arrived_dest or (arrived_charger and not prev_arrived):
            cnt = 0
            x_input = x.cuda()
            x_input2 = x_input.clone()
            x_input2[0, 6] = torch.clamp(x_input[0, 6], 0, 25 * args.dt)
            x_input2[0, 7] = torch.clamp(x_input[0, 7], -0.2, args.hold_t * args.dt)
            print("xe:%.2f ye:%.2f xd:%.2f yd:%.2f xc:%.2f yc:%.2f TB:%.2f hold:%.2f"%(
                x_input2[0, 0], x_input2[0, 1], x_input2[0, 2], x_input2[0, 3], x_input2[0, 4], 
                x_input2[0, 5], x_input2[0, 6], x_input2[0, 7], 
            ))
            dt_minus = 0
            debug_tt1=time.time()
            if args.multi_test and not arrived_charger:
                # only when not in charger stations
                x_input3 = x_input2.repeat([num_stations, 1])
                x_input3[:, 4:6] = tmp_map[:, 0:2]
                

                if args.mpc:
                    u, stl_max_i = solve_gurobi(x_input3, objs_np, ti, multi_test=True)
                    stl_max_i = torch.tensor(stl_max_i)
                    u = u.cuda()
                    x_tmptmp = x_input3[stl_max_i:stl_max_i+1]
                    seg = dynamics(x_tmptmp, u, include_first=True)
                elif args.rl or args.mbpo or args.pets:

                    _, u3, dt_minus = get_rl_xs_us(x_input3, rl_policy, args.nt, include_first=True)
                    seg3 = dynamics(x_input3, u3, include_first=True)
                    stl_score = stl(seg3, args.smoothing_factor, d={"hard":False})[:, :1]
                    stl_max_i = torch.argmax(stl_score, dim=0)
                    u = u3[stl_max_i:stl_max_i+1]
                    seg = seg3[stl_max_i:stl_max_i+1]
                elif args.grad:
                    u3, stl_scores = gradient_solve(ti, x_input3, stl, multi_test=True)
                    stl_max_i = torch.argmax(stl_scores)
                    x_tmptmp = x_input3[stl_max_i:stl_max_i+1]
                    u = u3[stl_max_i:stl_max_i+1]
                    seg = dynamics(x_tmptmp, u, include_first=True)
                elif args.plan:
                    u, stl_max_i = solve_planner(x_input3, objs_np, ti, multi_test=True)
                    stl_max_i = torch.tensor(stl_max_i)
                    u = u.cuda()
                    x_tmptmp = x_input3[stl_max_i:stl_max_i+1]
                    seg = dynamics(x_tmptmp, u, include_first=True)
                elif args.cem:
                    u3, stl_scores = solve_cem(ti, x_input3, stl, args)
                    stl_max_i = torch.argmax(stl_scores)
                    x_tmptmp = x_input3[stl_max_i:stl_max_i+1]
                    u = u3[stl_max_i:stl_max_i+1]
                    seg = dynamics(x_tmptmp, u, include_first=True)
                else:
                    u3 = net(x_input3)
                    seg3 = dynamics(x_input3, u3, include_first=True)
                    stl_score = stl(seg3, args.smoothing_factor, d={"hard":False})[:, :1]
                    stl_max_i = torch.argmax(stl_score, dim=0)
                    u = u3[stl_max_i:stl_max_i+1]
                    seg = seg3[stl_max_i:stl_max_i+1]

                    if args.finetune:
                        seg_tmp = seg.cuda()
                        acc_tmp = (stl(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                        err_idx = torch.where(acc_tmp<1)[0]
                        if err_idx.shape[0]>0:
                            ft_n = err_idx.shape[0]
                            print(ti, "[Before] Acc=%.2f, %d do not satisfy STL"%(torch.mean(acc_tmp), ft_n))
                            u_output_fix = u.clone()
                            for iii in range(ft_n):
                                sel_ = err_idx[iii]
                                u_fix = back_solve(seg[sel_:sel_+1, 0], u[sel_:sel_+1], net, stl)
                                u_output_fix[sel_:sel_+1] = u_fix
                            u = u_output_fix
                            seg_fix = dynamics(x_input, u, include_first=True)
                            acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                            err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                            ft_n_fix = err_idx_fix.shape[0]
                            print(ti, "[After]  Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
                            seg = seg_fix #dynamics(x_input3[stl_max_i:stl_max_i+1], u, include_first=True)
                            for ttti in range(args.nt):
                                print(ti, "T:%03d xe:%.2f ye:%.2f xd:%.2f yd:%.2f xc:%.2f yc:%.2f TB:%.2f hold:%.2f"%(ttti, 
                                    seg[0, ttti, 0], seg[0, ttti, 1], seg[0, ttti, 2], seg[0, ttti, 3], 
                                    seg[0, ttti, 4], seg[0, ttti, 5], seg[0, ttti, 6], seg[0, ttti, 7], 
                                    ))
            else:
                if args.mpc:
                    u = solve_gurobi(x_input2, objs_np, ti).cuda()
                elif args.rl or args.mbpo or args.pets:
                    _, u, dt_minus = get_rl_xs_us(x_input2, rl_policy, args.nt, include_first=True)
                elif args.grad:
                    u = gradient_solve(ti, x_input2, stl, multi_test=False)
                elif args.plan:
                    u, _ = solve_planner(x_input2, objs_np, ti, multi_test=True, need_to_find=False)
                    u = u.cuda()
                elif args.cem:
                    u, stl_scores = solve_cem(ti, x_input2, stl, args)
                else:
                    u = net(x_input2)
                    seg = dynamics(x_input2, u, include_first=True)
                    if args.finetune:
                        seg_tmp = seg.cuda()
                        acc_tmp = (stl(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                        err_idx = torch.where(acc_tmp<1)[0]
                        if err_idx.shape[0]>0:
                            ft_n = err_idx.shape[0]
                            print(ti, "[Before] Acc=%.2f, %d do not satisfy STL"%(torch.mean(acc_tmp), ft_n))
                            u_output_fix = u.clone()
                            for iii in range(ft_n):
                                sel_ = err_idx[iii]
                                u_fix = back_solve(seg[sel_:sel_+1, 0], u[sel_:sel_+1], net, stl)
                                u_output_fix[sel_:sel_+1] = u_fix
                            u = u_output_fix
                            seg_fix = dynamics(x_input2, u, include_first=True)
                            acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                            err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                            ft_n_fix = err_idx_fix.shape[0]
                            print(ti, "[After]  Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
                            seg = seg_fix
                            for ttti in range(args.nt):
                                print(ti, "T:%03d xe:%.2f ye:%.2f xd:%.2f yd:%.2f xc:%.2f yc:%.2f TB:%.2f hold:%.2f"%(ttti, 
                                    seg[0, ttti, 0], seg[0, ttti, 1], seg[0, ttti, 2], seg[0, ttti, 3], 
                                    seg[0, ttti, 4], seg[0, ttti, 5], seg[0, ttti, 6], seg[0, ttti, 7], 
                                    ))
                seg = dynamics(x_input, u, include_first=True)
            debug_tt2=time.time()
        if args.multi_test:
            map_list.append(tmp_map)
            marker_list.append(stl_max_i)
        
        seg_list.append(seg.detach().cpu())
        history.append(x.clone())

        move_distance += torch.norm(seg[:, cnt+1, :2].detach().cpu()-x[:, :2], dim=-1)
        stl_success += (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        safety += 1-int(any([in_poly(x[0, :2], seg[0,cnt+1,:2].detach().cpu(), obs.detach().cpu()) for obs in objs[1:]]))
        battery_ok += int(x[0, 6]>=0)
        x = seg[:, cnt+1].detach().cpu()
        cnt+=1

        seg_total = seg.clone()
        # EVALUATION
        debug_dt = debug_tt2 - debug_tt1 - dt_minus
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(the_env.generate_reward_batch(to_np(seg_total[:,0])))
        
        metrics["t"].append(debug_dt)
        metrics["safety"].append(safety/nt)
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)
        metrics["battery"].append(battery_ok/nt)
        metrics["distance"].append(move_distance.item())
        metrics["goals"].append(astro_i)

        # any update and corresponding handling
        arrived_dest = torch.norm(x[0, 0:2] - x[0, 2:4], dim=-1) < close_enough_dist
        if arrived_dest:
            astro_i += 1
            x[0, 2:6] = x_init[astro_i, 2:6]

        arrived_charger = torch.norm(x[0, 0:2] - x[0, 4:6], dim=-1) < close_enough_dist
        if arrived_charger==False:
            x[0, 7] = args.hold_t * args.dt

        prev_arrived = arrived_charger

        NT = ti+1
        print("t:%03d| MPC:%d, stl-acc:%.2f safety:%.2f battery_ok:%.2f distance:%.2f goals:%d" %(
            ti, args.mpc_update_freq, stl_success/NT, safety/NT, battery_ok/NT, move_distance, astro_i
        ))
    stat_str="MPC:%d, stl-acc:%.2f safety:%.2f battery_ok:%.2f distance:%.2f goals:%d" %(
            args.mpc_update_freq, stl_success/nt, safety/nt, battery_ok/nt, move_distance, astro_i
        )
    print(stat_str)
    
    metrics_avg = {xx:np.mean(np.array(metrics[xx])) for xx in metrics}
    metrics_avg["safety"] = metrics["safety"][-1]
    metrics_avg["battery"] = metrics["battery"][-1]
    metrics_avg["distance"] = metrics["distance"][-1]
    metrics_avg["goals"] = metrics["goals"][-1]
    if args.no_eval==False:
        eval_proc(metrics, "e5_rover", args, metrics_avg)

    if args.no_viz==False:
        history = torch.stack(history, dim=1)
        seg_list = torch.stack(seg_list, dim=1)
        # visualization
        for ti in range(nt):
            if ti % args.sim_freq == 0 or ti == nt - 1:
                if ti % 5 == 0:
                    print("Viz", ti)
                ax = plt.gca()
                plot_env(ax, objs_np)
                s = history[0, ti]
                ax.add_patch(Circle([s[0], s[1]], args.close_thres/2, color="blue", label="rover"))
                ax.add_patch(Circle([s[2], s[3]], args.close_thres/2, color="green", label="destination"))
                if args.multi_test:
                    if map_list[ti] is not None:
                        for j in range(num_stations):
                            tmp_map = map_list[ti]
                            ax.add_patch(Circle([tmp_map[j,0], tmp_map[j,1]], args.close_thres/2, color="orange", label="charger" if j==0 else None))
                        tmp_max_i = to_np(marker_list[ti])
                        ax.add_patch(Circle([to_np(tmp_map[tmp_max_i,0]), to_np(tmp_map[tmp_max_i,1])], args.close_thres/2, color="chocolate"))
                    else:
                        ax.add_patch(Circle([s[4], s[5]], args.close_thres/2, color="orange", label="charger"))
                else:
                    ax.add_patch(Circle([s[4], s[5]], args.close_thres/2, color="orange", label="charger"))
                ax.plot(seg_list[0, ti,:,0], seg_list[0, ti,:,1], color="blue", linewidth=2, alpha=0.5, zorder=10)
                ax.text(s[0]+0.25, s[1]+0.25, "%.1f"%(s[6]), fontsize=12)
                plt.xlim(0, 10)
                plt.ylim(0, 10)
                plt.legend(fontsize=14, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.3))
                ax.axis("scaled")
                figname="%s/t_%03d.png"%(args.viz_dir, ti)
                plt.title("Simulation (%04d/%04d)"%(ti, nt))
                plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
                plt.close()
                fs_list.append(figname)
        
        os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
        generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.3, fs_list)
    debug_t2 = time.time()
    print("Finished in %.2f seconds"%(debug_t2 - debug_t1))


def plot_env(ax, objs_np):
    for ii, obj in enumerate(objs_np[1:]):
        rect = Polygon(obj, color="gray", alpha=0.25, label="obstacle" if ii==0 else None)
        ax.add_patch(rect)


def generate_objs():
    objs_np = [np.array([[0.0, 0.0], [10, 0], [10, 10], [0, 10]])]  # map
    objs_np.append(np.array([[0.0, 0.0], [args.obs_w, 0], [args.obs_w, args.obs_w], [0, args.obs_w]]))  # first obstacle
    objs_np.append(objs_np[1] + np.array([[5-args.obs_w/2, 10-args.obs_w]]))  # second obstacle (top-center)
    objs_np.append(objs_np[1] + np.array([[10-args.obs_w, 0]]))  # third obstacle (bottom-right)
    objs_np.append(objs_np[1] / 2 + np.array([[5-args.obs_w/4, 5-args.obs_w/4]]))  # forth obstacle (center-center, shrinking)

    objs = [to_torch(ele) for ele in objs_np]
    objs_t1 = [ele.unsqueeze(0).unsqueeze(0) for ele in objs]
    objs_t2 = [torch.roll(ele, shifts=-1, dims=2) for ele in objs_t1]

    return objs_np, objs, objs_t1, objs_t2


def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    objs_np, objs, objs_t1, objs_t2 = generate_objs()

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

    if args.test:
        rl_policy = None
        if args.rl:
            from stable_baselines3 import SAC, PPO, A2C
            rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
        elif args.mbpo or args.pets:
            rl_policy = get_mbrl_models(get_exp_dir()+"/"+args.rl_path, args, args.mbpo)
        test_mars(net, rl_policy, stl, objs_np, objs)
        exit()

    # TODO init
    x_init = initialize_x(args.num_samples, objs).float().cuda()

    x_init_val = initialize_x(args.num_samples//10, objs).float().cuda()

    print(stl)
    stl.update_format("word")
    print(stl)
    relu = nn.ReLU()
    for epi in range(args.epochs):
        eta.update()
        if args.update_init_freq >0 and epi % args.update_init_freq == 0 and epi!=0:
            x_init = initialize_x(args.num_samples, objs).float().cuda()
        x0 = x_init.detach()
        u = net(x0)        
        seg = dynamics(x0, u, include_first=True)

        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc)
        
        shall_charge = (x0[..., 6:7] <= battery_limit).float()
        acc_mask = acc
        if args.no_acc_mask:
            acc_mask = 1
        charge_dist = torch.norm(seg[:, :, :2] - x0.unsqueeze(dim=1)[:, :, 4:6], dim=-1)
        dest_dist = torch.norm(seg[:, :, :2] - x0.unsqueeze(dim=1)[:, :, 2:4], dim=-1)
        dist_loss = torch.mean((charge_dist * shall_charge + dest_dist * (1 - shall_charge)) * acc_mask)
        dist_loss = dist_loss * args.dist_w

        loss = torch.mean(relu(0.5-score)) + dist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epi % args.print_freq == 0:
            u_val = net(x_init_val.detach())        
            seg_val = dynamics(x_init_val, u_val, include_first=True)
            acc_val = (stl(seg_val, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
            acc_avg_val = torch.mean(acc_val)

            print("%s|%03d  loss:%.3f acc:%.3f dist:%.3f acc_val:%.3f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), acc_avg.item(),
                dist_loss.item(), acc_avg_val.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str()))

        # Save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))
        
        if epi % args.viz_freq == 0 or epi == args.epochs - 1:
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
                    plot_env(ax, objs_np)

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

                    # plot the time curves
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
                    # ax.axis("off")
                    ax.axes.get_xaxis().set_visible(False)
                    ax.axes.get_yaxis().set_visible(False)
                    plt.ylim(-0.2, 1.2)

            figname="%s/iter_%05d.png"%(args.viz_dir, epi)
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
            plt.close()


def back_solve(x_input, output_0, policy, stl):
    nt0 = 4
    A0 = np.array([0] + [1.0] * 8 + [0.5] * 8)
    A1 = np.array([0] + [-np.pi + np.pi/4*ii for ii in range(8)] + [-np.pi + np.pi/4*ii for ii in range(8)])
    li0 = [A0] * nt0
    li1 = [A1] * nt0
    a_seq0 = np.array(np.meshgrid(*li0)).T.reshape(-1, nt0)
    a_seq1 = np.array(np.meshgrid(*li1)).T.reshape(-1, nt0)
    a_seq = np.stack([a_seq0, a_seq1], axis=-1)
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
    
    u_merged = torch.cat([a_seq[idx, :],u_output[idx, :args.nt-nt0]], dim=0).unsqueeze(0)
    return u_merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=250000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=10)
    add("--dt", type=float, default=0.2)
    add("--print_freq", type=int, default=500)
    add("--viz_freq", type=int, default=5000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)   

    add("--sim_freq", type=int, default=1)
    add("--rover_vmax", type=float, default=10.0)
    add("--astro_vmax", type=float, default=0.0)
    add("--rover_vmin", type=float, default=0.0)
    add("--astro_vmin", type=float, default=0.0)
    add("--close_thres", type=float, default=0.8)
    add("--battery_decay", type=float, default=1.0)
    add("--battery_charge", type=float, default=5.0)
    add("--obs_w", type=float, default=3.0)
    add("--ego_turn", action="store_true", default=False)
    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--no_obs", action="store_true", default=False)
    add("--one_obs", action="store_true", default=False)
    add("--limited", action="store_true", default=False)
    add("--if_cond", action='store_true', default=False)
    add("--nominal", action='store_true', default=False)
    add("--dist_w", type=float, default=0.01)
    add("--no_acc_mask", action='store_true', default=False)
    add("--seq_reach", action='store_true', default=False)
    add("--together_ratio", type=float, default=0.2)
    add("--until_emergency", action='store_true', default=False)
    add("--list_and", action='store_true', default=False)
    add("--mpc_update_freq", type=int, default=1)
    add("--seg_gain", type=float, default=1.0)
    add("--hold_t", type=int, default=3)
    add("--no_tanh", action='store_true', default=False)
    add("--hard_soft_step", action='store_true', default=False)
    add("--norm_ap", action='store_true', default=False)
    add("--tanh_ratio", type=float, default=0.05)
    add("--update_init_freq", type=int, default=500)

    add("--multi_test", action='store_true', default=False)
    add("--mpc", action='store_true', default=False)
    add("--num_stations", type=int, default=5)

    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)

    add("--plan", action="store_true", default=False)
    add("--rl", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)
    add("--rl_stl", action="store_true", default=False)
    add("--rl_acc", action="store_true", default=False)
    add("--eval_path", type=str, default="eval_result")
    add("--no_viz", action="store_true", default=False)
    add("--no_eval", action="store_true", default=False)

    add("--finetune", action="store_true", default=False)
    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)
    add("--cem", action='store_true', default=False)
    args = parser.parse_args()
    args.no_acc_mask = True
    args.no_tanh = True
    args.norm_ap = True
    args.hard_soft_step = True
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))