from lib_stl_core import *
from matplotlib.patches import Polygon, Rectangle, Ellipse
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.size': 12})
import casadi

import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, to_torch, xxyy_2_Ab, get_exp_dir, eval_proc
from lib_cem import solve_cem_func
from utils_mbrl import get_mbrl_models, get_mbrl_u

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.net = nn.Sequential(
                nn.Linear(3 + 3 + 3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, args.nt)
            )
    
    def forward(self, x):     
        num_samples = x.shape[0]
        u = self.net(x)
        u1 = torch.tanh(u) * 40.0
        return u1

def dynamics(x0, u, include_first=False):
    # input:  x, vx, dy, (dx, L, dG)  # 3 + 3 * 2 = 9
    # input:  u, (n, T)
    # return: s, (n, T, 9)
    
    t = u.shape[1]
    x = x0.clone()
    if include_first:
        segs=[x0]
    else:
        segs = []
    for ti in range(t):
        new_x = dynamics_per_step(x, u[:, ti:ti+1])
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)


def dynamics_per_step(x, u):
    new_x = torch.zeros_like(x)
    new_x[:, 0] = torch.clip(x[:, 0] + x[:, 1] * args.dt, 0.1, args.canvas_w-0.1)
    new_x[:, 1] = torch.clip(x[:, 1] + u[:, 0] * args.dt, args.v_min, args.v_max)
    new_x[:, 2] = x[:, 2] + args.vy * args.dt
    new_x[:, 3:] = x[:, 3:]
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
            u = torch.from_numpy(u * 40)
        else:
            if args.mbpo:
                u = get_mbrl_u(x, None, policy, mbpo=True)
            elif args.pets:
                u_list=[]
                for iii in range(x.shape[0]):
                    u = get_mbrl_u(x[iii], None, policy, mbpo=False)
                    u_list.append(u)
                u = torch.stack(u_list, dim=0)
            u = u * 40
        u = torch.clip(u, -40, 40)

        new_x = dynamics_per_step(x, u)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.cat(us, dim=1)
    return xs, us, dt_minus


def dynamics_pseudo(x0, u, include_first=False):
    # input:  x, vx, dy, (dx, L, dG)  # 3 + 3 * 2 = 9
    # input:  u, (n, T)
    # return: s, (n, T, 9)
    
    t = u.shape[1]
    x = x0.clone()
    if include_first:
        segs=[x0]
    else:
        segs = []
    for ti in range(t):
        new_x = torch.zeros_like(x)
        new_x[:, 0] = x[:, 0] + x[:, 1] * args.dt
        new_x[:, 1] = x[:, 1] + u[:, ti] * args.dt
        new_x[:, 2] = x[:, 2] + args.vy * args.dt
        new_x[:, 3:] = x[:, 3:]
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)


def generate_seg(n):
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

def initialize_x():
    total_sum = 0
    n_try = 0
    x_list = []
    while total_sum <= args.num_samples:
        n_try+=1
        print("try",n_try)
        x = initialize_x_cycle()
        collide = torch.logical_and(x[..., 2]>=-args.obs_h, torch.logical_and(x[..., 0]>=x[...,3], x[..., 0]<=x[..., 3]+x[...,4]))
        collide_next = torch.logical_and(x[..., 2]+args.vy*args.dt>=-args.obs_h, 
                                         torch.logical_and(x[..., 0]+x[...,1]*args.dt>=x[...,3], x[..., 0]+x[...,1]*args.dt<=x[..., 3]+x[...,4]))
        safe_idx = torch.where(torch.logical_and(collide==False, collide_next==False))[0]
        
        x = x[safe_idx]
        total_sum += x.shape[0]
        x_list.append(x)
    x = torch.cat(x_list, dim=0)[:args.num_samples]
    
    return x

def initialize_x_cycle():
    n = args.num_samples
    x = uniform_tensor(0, args.canvas_w, (n, 1))
    vx = uniform_tensor(args.v_min, args.v_max, (n, 1))
    dy = uniform_tensor(-args.y_level, 0, (n, 1))
    dx0, L0, G0 = generate_seg(n)
    dx1, L1, G1 = generate_seg(n)  

    # make sure feasible
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


def solve_gurobi(tti, x_init):
    import gurobipy as gp
    from gurobipy import GRB
    x_init = to_np(x_init)
    nt = args.nt
    dt = args.dt
    M = 10^6
    XMAX = args.canvas_w
    XMIN = 0
    VMAX = args.v_max
    VMIN = args.v_min
    UMAX = 40
    UMIN = -40
    VY = args.vy
    is_goal_circle = True
    timeLimit = 1

    obs0_left = x_init[0, 3]
    bloat = 0.25
    if obs0_left==0:
        obs0_left=-0.1
    obs0_right = max(x_init[0, 3],0) + x_init[0, 4]
    if obs0_right==args.canvas_w:
        obs0_right = args.canvas_w + 0.1
    obs0_up = 0
    obs0_down = -args.obs_h

    obs1_left = x_init[0, 3+3]
    if obs1_left==0:
        obs1_left=-0.1
    obs1_right = max(x_init[0, 3+3],0) + x_init[0, 4+3]
    if obs1_right==args.canvas_w:
        obs1_right = args.canvas_w + 0.1
    obs1_up = 0 + args.y_level
    obs1_down = -args.obs_h + args.y_level

    has_goal0 = x_init[0, 5] >= 0
    has_goal1 = x_init[0, 5+3] >= 0

    dw = args.goal_w / 2
    if has_goal0:
        goal0_x = x_init[0, 5]
        goal0_y = -args.obs_h/2
        goal0_left = goal0_x - dw
        goal0_right = goal0_x + dw
        goal0_up = goal0_y + dw
        goal0_down = goal0_y - dw
    if has_goal1:
        goal1_x = x_init[0, 5+3]
        goal1_y = -args.obs_h/2 + args.y_level
        goal1_left = goal1_x- dw
        goal1_right = goal1_x + dw
        goal1_up = goal1_y + dw
        goal1_down = goal1_y - dw

    t_start=time.time()
    m = gp.Model("mip1")
    m.setParam( 'OutputFlag', False)
    x = m.addVars(nt+1, name="x", lb=float('-inf'), ub=float('inf'))
    v = m.addVars(nt+1, name="v", lb=float('-inf'), ub=float('inf'))
    y = m.addVars(nt+1, name="y", lb=float('-inf'), ub=float('inf'))
    u = m.addVars(nt, name="u", lb=float('-inf'), ub=float('inf'))
    I0 = m.addVars(nt, 4, vtype=GRB.BINARY, name="I0")
    I1 = m.addVars(nt, 4, vtype=GRB.BINARY, name="I1")
    I_goal0 = m.addVars(nt, vtype=GRB.BINARY, name="I_goal0")
    I_goal1 = m.addVars(nt, vtype=GRB.BINARY, name="I_goal1")
    if has_goal0:
        if is_goal_circle:
            gamma_g0 = m.addVars(nt, name="gamma_g0", lb=float('-inf'), ub=float('inf'))
        else:
            gamma_g0 = m.addVars(nt, 4, name="gamma_g0", lb=float('-inf'), ub=float('inf'))
    if has_goal1:
        if is_goal_circle:
            gamma_g1 = m.addVars(nt, name="gamma_g1", lb=float('-inf'), ub=float('inf'))
        else:
            gamma_g1 = m.addVars(nt, 4, name="gamma_g1", lb=float('-inf'), ub=float('inf'))
    gamma_o0 = m.addVars(nt, 4, name="gamma_o0", lb=float('-inf'), ub=float('inf'))
    gamma_o1 = m.addVars(nt, 4, name="gamma_o1", lb=float('-inf'), ub=float('inf'))

    # global integer constraints
    m.addConstr(gp.quicksum(I_goal0)>=1)  # should achieve goal at least once
    m.addConstr(gp.quicksum(I_goal1)>=1)  # should achieve goal at least once

    # initial setup
    m.addConstr(x[0] == x_init[0, 0])
    m.addConstr(v[0] == x_init[0, 1])
    m.addConstr(y[0] == x_init[0, 2])
    
    # constraints
    for ti in range(nt):
        m.addConstr(x[ti]<= XMAX)
        m.addConstr(x[ti]>= XMIN)
        m.addConstr(v[ti]<= VMAX)
        m.addConstr(v[ti]>= VMIN)
        m.addConstr(u[ti]<= UMAX)
        m.addConstr(u[ti]>= UMIN)
        m.addConstr(x[ti+1] == x[ti] + v[ti] * dt)
        m.addConstr(v[ti+1] == v[ti] + u[ti] * dt)
        m.addConstr(y[ti+1] == y[ti] + VY * dt)
        if has_goal0:
            if is_goal_circle:
                m.addConstr((x[ti+1]-goal0_x)**2 + (y[ti+1]-goal0_y)**2 <= (args.goal_w/2)**2 + (1-I_goal0[ti]) * M + gamma_g0[ti])
            else:
                m.addConstr(x[ti+1] >= goal0_left - (1-I_goal0[ti]) * M + gamma_g0[ti,0])
                m.addConstr(x[ti+1] <= goal0_right + (1-I_goal0[ti]) * M + gamma_g0[ti,1])
                m.addConstr(y[ti+1] >= goal0_down - (1-I_goal0[ti]) * M + gamma_g0[ti,2])
                m.addConstr(y[ti+1] <= goal0_up + (1-I_goal0[ti]) * M + gamma_g0[ti,3])  

        if has_goal1:
            if is_goal_circle:
                m.addConstr((x[ti+1]-goal1_x)**2 + (y[ti+1]-goal1_y)**2 <= (args.goal_w/2)**2 + (1-I_goal1[ti]) * M + gamma_g1[ti])
            else:
                m.addConstr(x[ti+1] >= goal1_left - (1-I_goal1[ti]) * M + gamma_g1[ti,0])
                m.addConstr(x[ti+1] <= goal1_right + (1-I_goal1[ti]) * M + gamma_g1[ti,1])
                m.addConstr(y[ti+1] >= goal1_down - (1-I_goal1[ti]) * M + gamma_g1[ti,2])
                m.addConstr(y[ti+1] <= goal1_up + (1-I_goal1[ti]) * M + gamma_g1[ti,3])

        m.addConstr(gp.quicksum(I0[ti,i] for i in range(4)) <= 3)
        m.addConstr(gp.quicksum(I1[ti,i] for i in range(4)) <= 3)

        m.addConstr(x[ti+1] <= obs0_left-bloat + I0[ti,0] * M + gamma_o0[ti,0])
        m.addConstr(x[ti+1] >= obs0_right+bloat - I0[ti,1] * M + gamma_o0[ti,1])
        m.addConstr(y[ti+1] <= obs0_down + I0[ti,2] * M + gamma_o0[ti,2])
        m.addConstr(y[ti+1] >= obs0_up - I0[ti,3] * M + gamma_o0[ti,3])

        m.addConstr(x[ti+1] <= obs1_left-bloat + I1[ti,0] * M + gamma_o1[ti,0])
        m.addConstr(x[ti+1] >= obs1_right+bloat - I1[ti,1] * M + gamma_o1[ti,1])
        m.addConstr(y[ti+1] <= obs1_down + I1[ti,2] * M + gamma_o1[ti,2])
        m.addConstr(y[ti+1] >= obs1_up - I1[ti,3] * M + gamma_o1[ti,3])

    sum_u = gp.quicksum(u[i]*u[i] for i in range(nt))
    sum_o0 = gp.quicksum(gamma_o0[i,j]*gamma_o0[i,j] for i in range(nt) for j in range(4))
    sum_o1 = gp.quicksum(gamma_o1[i,j]*gamma_o1[i,j] for i in range(nt) for j in range(4))
    if has_goal0:
        if is_goal_circle:
            sum_g0 = gp.quicksum(gamma_g0[i]*gamma_g0[i] for i in range(nt))
        else:
            sum_g0 = gp.quicksum(gamma_g0[i,j]*gamma_g0[i,j] for i in range(nt) for j in range(4))
    else:
        sum_g0 = 0
    if has_goal1:
        if is_goal_circle:
            sum_g1 = gp.quicksum(gamma_g1[i]*gamma_g1[i] for i in range(nt))
        else:
            sum_g1 = gp.quicksum(gamma_g1[i,j]*gamma_g1[i,j] for i in range(nt) for j in range(4))
    else:
        sum_g1 = 0

    m.Params.TimeLimit = timeLimit - m.getAttr(GRB.Attr.Runtime)
    m.setObjective(M * 1000 * (sum_o0+sum_o1+sum_g0+sum_g1) + 0.1*sum_u, GRB.MINIMIZE)
    m.optimize()
    t_end=time.time()
    print("ti-%04d t= %.4f sec"%(tti, t_end-t_start))
    u_torch=[]
    for ti in range(nt):
        u_torch.append(u[ti].X)
    u_torch=torch.tensor(u_torch).float().unsqueeze(0)
    return u_torch

def gradient_solve(tti, x_init, stl, multi_test=False):
    relu = torch.nn.ReLU()
    u_lat = torch.zeros(x_init.shape[0], args.nt).requires_grad_()
    x_init = x_init.cpu()
    optimizer = torch.optim.Adam([u_lat], lr=args.grad_lr)
    tt1=time.time()
    print(x_init)
    prev_loss = None
    for i in range(args.grad_steps):
        u = torch.nn.Tanh()(u_lat) * 40
        u = u_lat
        seg = dynamics_pseudo(x_init, u, include_first=True)
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        loss = torch.mean(relu(0.5-score))

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        if i>0:
            if abs(loss.item())<1e-5 or abs(prev_loss.item()-loss.item())<1e-5:
                break
        prev_loss = loss.detach()
    tt2=time.time()
    print("%05d t:%.4f seconds"%(tti, tt2-tt1))
    return u.detach().cuda()

def solve_mpc(x_init):
    x_init = to_np(x_init)
    u = None
    mpc_t1 = time.time()
    
    opti = casadi.Opti()
    mpc_max_iters = 10000
    quiet = True
    umax = 40
    r = args.goal_w/2

    x = opti.variable(args.nt + 1, 3)  # x, vx, dy
    u = opti.variable(args.nt, 1)  # a
    gamma = opti.variable(args.nt, 1)

    for i in range(3):
        x[0, i] = x_init[0, i]

    obs = x_init  # (x, vx, dy, | dx, dy, G0, dx1, dx2, G1)
    print(obs)

    # dynamics 
    for ti in range(args.nt):
        XMIN = 0.0
        XMAX = args.canvas_w - 0.0
        VMIN = args.v_min
        VMAX = args.v_max
        x[ti+1, 0] = x[ti, 0] + x[ti, 1] * args.dt
        x[ti+1, 1] = x[ti, 1] + u[ti, 0] * args.dt
        x[ti+1, 2] = x[ti, 2] + args.vy * args.dt
    
    level1_list=[]
    level2_list=[]
    goal1_list=[]
    goal2_list=[]
    x_sim = np.array(x_init)
    for ti in range(args.nt):
        x_sim[:, 2] = x_sim[:, 2] + args.vy * args.dt
        if (x_sim[:, 2]+args.obs_h) * x_sim[:, 2] <= 0:
            level1_list.append(ti+1)
            if obs[0, 5] > 0:
                goal1_list.append(ti+1)
        if (x_sim[:, 2]+args.obs_h-args.y_level) * (x_sim[:, 2]-args.y_level) <= 0:
            level2_list.append(ti+1)
            if obs[0, 5+3] > 0:
                goal2_list.append(ti+1)
    for ti in range(args.nt):
        opti.subject_to(u[ti, 0] <= umax)
        opti.subject_to(u[ti, 0] >= -umax)
        opti.subject_to(x[ti+1, 0] <= XMAX)
        opti.subject_to(x[ti+1, 0] >= XMIN)
        opti.subject_to(x[ti+1, 1] <= VMAX)
        opti.subject_to(x[ti+1, 1] >= VMIN)

        if ti in level1_list:
            # opti.subject_to((obs[0, 3]+obs[0,4]-x[ti+1,0])*(obs[0,3]-x[ti+1,0])>=0)
            left=obs[0,3]
            right=obs[0, 3]+obs[0,4]
            opti.subject_to((x[ti+1,0]-obs[0, 3]-obs[0,4]) * (x[ti+1,0]-obs[0,3]) >=gamma[ti,0])
            if ti in goal1_list:
                opti.subject_to(-(x[ti+1,0]-obs[0,5])*(x[ti+1,0]-obs[0,5]) - (x[ti+1,2]+args.obs_h/2)*(x[ti+1,2]+args.obs_h/2) + r*r<=-gamma[ti,0])

        elif ti in level2_list:
            opti.subject_to((obs[0, 3+3]+obs[0,4+3]-x[ti+1,0])*(obs[0,3+3]-x[ti+1,0])>=gamma[ti,0])
            if ti in goal2_list:
                opti.subject_to(-(x[ti+1,0]-obs[0,5+3])*(x[ti+1,0]-obs[0,5+3]) - (x[ti+1,2]+args.obs_h/2)*(x[ti+1,2]+args.obs_h/2) + r*r<=-gamma[ti,0])

    loss = casadi.sumsqr(gamma) * 100000 #+ casadi.sumsqr(eval_term_list) * 1
    opti.minimize(loss)

    p_opts = {"expand": True}
    s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)

    sol1 = opti.solve()
    x_np = opti.debug.value(x)
    u_np = opti.debug.value(u)
    mpc_t2 = time.time()

    return to_torch(u_np[None, :, None]).cpu()


def solve_planner(ti, x_init):
    from lib_pwlplan import plan, Node
    from gurobipy import GRB
    tt1=time.time()
    def func1(m, PWLs, di):
        v = m.addVars(args.nt+1, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        u = m.addVars(args.nt, lb=-40, ub=40, name="the_u")
        m.addConstr(v[0] == x_init[0, 1])
        for i in range(args.nt):
            m.addConstr(PWLs[0][i+1][0][1] == PWLs[0][i][0][1] + VY * args.dt)
            m.addConstr(PWLs[0][i+1][0][0] - PWLs[0][i][0][0] == v[i+1] * args.dt)
            m.addConstr(v[i+1] - v[i] == u[i] * args.dt)
            m.addConstr(PWLs[0][i+1][1] - PWLs[0][i][1] == args.dt)

            m.addConstr(PWLs[0][i+1][0][0]>=XMIN)
            m.addConstr(PWLs[0][i+1][0][0]<=XMAX)
    
    def func2(m, PWLs, di):
        do_nothing=1

    nt = args.nt
    dt = args.dt
    XMAX = args.canvas_w
    XMIN = 0
    YMAX = None
    YMIN = None
    VMAX = args.v_max
    VMIN = args.v_min
    UMAX = 40
    UMIN = -40
    VY = args.vy

    tmax = args.nt * args.dt + 0.015
    vmax = VMAX
    
    # (xmin, xmax, ymin, ymax)
    obs0_left = x_init[0, 3]
    bloat = 0.25
    if obs0_left==0:
        obs0_left=-0.1
    obs0_right = max(x_init[0, 3],0) + x_init[0, 4]
    if obs0_right==args.canvas_w:
        obs0_right = args.canvas_w + 0.1
    obs0_up = 0
    obs0_down = -args.obs_h

    obs1_left = x_init[0, 3+3]
    if obs1_left==0:
        obs1_left=-0.1
    obs1_right = max(x_init[0, 3+3],0) + x_init[0, 4+3]
    if obs1_right==args.canvas_w:
        obs1_right = args.canvas_w + 0.1
    obs1_up = 0 + args.y_level
    obs1_down = -args.obs_h + args.y_level

    has_goal0 = x_init[0, 5] >= 0
    has_goal1 = x_init[0, 5+3] >= 0
    
    dw = args.goal_w / 2
    if has_goal0:
        goal0_x = x_init[0, 5]
        goal0_y = -args.obs_h/2
        goal0_left = goal0_x - dw
        goal0_right = goal0_x + dw
        goal0_up = goal0_y + dw
        goal0_down = goal0_y - dw
    if has_goal1 and (-args.obs_h/2 + args.y_level - x_init[0, 2] - dw) < (args.nt * args.dt * args.vy):
        goal1_x = x_init[0, 5+3]
        goal1_y = -args.obs_h/2 + args.y_level
        goal1_left = goal1_x- dw
        goal1_right = goal1_x + dw
        goal1_up = goal1_y + dw
        goal1_down = goal1_y - dw
    
    obs0 = [obs0_left, obs0_right, obs0_down, obs0_up]
    obs1 = [obs1_left, obs1_right, obs1_down, obs1_up]
    A0, b0 = xxyy_2_Ab(obs0)
    A1, b1 = xxyy_2_Ab(obs1)

    dw = args.goal_w / 2
    reach_goals = []
    if has_goal0:
        goal0 = [goal0_left, goal0_right, goal0_down, goal0_up]
        A2, b2 = xxyy_2_Ab(goal0)
        reach_goal = Node('mu', info={'A':A2, 'b':b2})
        finally_reach_goal = Node('F', deps=[reach_goal,], info={'int':[0, tmax]})
        reach_goals.append(finally_reach_goal)

    if has_goal1:
        goal1 = [goal1_left, goal1_right, goal1_down, goal1_up]
        A3, b3 = xxyy_2_Ab(goal1)
        reach_goal = Node('mu', info={'A':A3, 'b':b3})
        finally_reach_goal = Node('F', deps=[reach_goal,], info={'int':[0, tmax]})
        reach_goals.append(finally_reach_goal)

    avoids = [
        Node('negmu', info={'A':A0, 'b':b0}),
        Node('negmu', info={'A':A1, 'b':b1})
        ]
    avoid_obs = Node('and', deps=avoids)
    always_avoid_obs = Node('A', deps=[avoid_obs,], info={'int':[0, tmax]})

    specs = [always_avoid_obs] + reach_goals
    x0s = [np.array([x_init[0,0], x_init[0, 2]])]
    PWL, u_out = plan(x0s, specs, bloat=0.01, MIPGap=0.05, num_segs=args.nt, tmax=tmax, vmax=vmax, extra_fn_list=[func1], return_u=True, quiet=True)
    
    tt2=time.time()
    print("%03d t:%.4f seconds" %(ti, tt2-tt1)) 
    u = to_torch(u_out[None, :, None]).cpu()
    return u

def solve_cem(ti, x_input, stl, args):
    def dynamics_step_func(x, u):
        return dynamics_per_step(x, u)
    
    def reward_func(trajs):
        return stl(trajs, args.smoothing_factor, d={"hard":True})[:, 0]

    u_min = torch.tensor([-40]).cuda()
    u_max = torch.tensor([40]).cuda()
    u, _, info = solve_cem_func(
        (x_input[0]).cuda(), state_dim=x_input.shape[-1], nt=args.nt, action_dim=u_min.shape[0],
        num_iters=500, n_cand=10000, n_elites=100, policy_type="direct",
        dynamics_step_func=dynamics_step_func, reward_func=reward_func,
        transform=None, u_clip=(u_min, u_max), seed=None, args=None, 
        extra_dict=None, quiet=False, device="gpu", visualize=False
    )
    return u


def test_game(net, rl_policy, stl):
    metrics_str=["acc", "reward", "score", "t"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.maze_env import MazeEnv
    maze_env = MazeEnv(args)

    nt = 500  # 500
    levels = 200
    env = []
    for i in range(levels):
        dx0, L0, G0 = generate_seg(1)
        env.append([dx0, i*args.y_level, L0, G0])
    
    env = torch.tensor(env).float()

    # TODO(debug)
    env[0, 0] = -0.1

    x = torch.tensor([[0, 0, -args.y_level]]).float()
    history = [x.clone()]
    fs_list = []
    seg_list = []
    base_i=0
    for ti in range(nt):
        if ti % 5==0:
            print(ti)
        # find the two levels
        if env[base_i, 1] < x[0, 2]:
            base_i+=1
        
        # collect the input
        debug_t1=time.time()
        dt_minus = 0
        if args.mpc:
            x_input = torch.cat([x, env[base_i:base_i+1, [0, 2, 3]], env[base_i+1:base_i+2, [0, 2, 3]]], dim=-1)
            x_input[0, 2] = x[0, 2] - env[base_i, 1]
            u = solve_gurobi(ti, x_input)
        elif args.rl or args.mbpo or args.pets:
            x_input = torch.cat([x, env[base_i:base_i+1, [0, 2, 3]], env[base_i+1:base_i+2, [0, 2, 3]]], dim=-1)
            x_input[0, 2] = x[0, 2] - env[base_i, 1]
            tmp_xs, u, dt_minus = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=True)

        elif args.plan:
            x_input = torch.cat([x, env[base_i:base_i+1, [0, 2, 3]], env[base_i+1:base_i+2, [0, 2, 3]]], dim=-1)
            x_input[0, 2] = x[0, 2] - env[base_i, 1]
            u = solve_planner(ti, x_input)
        elif args.grad:
            x_input = torch.cat([x, env[base_i:base_i+1, [0, 2, 3]], env[base_i+1:base_i+2, [0, 2, 3]]], dim=-1)
            x_input[0, 2] = x[0, 2] - env[base_i, 1]
            u = gradient_solve(ti, x_input, stl).cpu()
        elif args.cem:
            x_input = torch.cat([x, env[base_i:base_i+1, [0, 2, 3]], env[base_i+1:base_i+2, [0, 2, 3]]], dim=-1)
            x_input[0, 2] = x[0, 2] - env[base_i, 1]
            u = solve_cem(ti, x_input, stl, args).cpu()[None,:]
        else:
            x_input = torch.cat([x, env[base_i:base_i+1, [0, 2, 3]], env[base_i+1:base_i+2, [0, 2, 3]]], dim=-1)
            x_input[0, 2] = x[0, 2] - env[base_i, 1]
            x_input = x_input.cuda()
            u = net(x_input)

            if args.finetune:
                seg = dynamics(x_input, u, include_first=True)
                acc_tmp = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                err_idx = torch.where(acc_tmp<1)[0]
                if err_idx.shape[0]>0:
                    ft_n = err_idx.shape[0]
                    print(ti, "[Before] Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp), ft_n, err_idx))
                    u_output_fix = u.clone()
                    for iii in range(ft_n):
                        sel_ = err_idx[iii]
                        u_fix = back_solve(x_input[sel_:sel_+1], u[sel_:sel_+1], net, stl)
                        u_output_fix[sel_:sel_+1] = u_fix
                    u = u_output_fix
                    seg_fix = dynamics(x_input, u)
                    acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                    err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                    ft_n_fix = err_idx_fix.shape[0]
                    print(ti, "[After]  Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))

        debug_t2=time.time()

        seg = dynamics(x_input, u, include_first=True)
        seg_total = seg.clone()
        seg[:, :, 2] = seg[:, :, 2] + env[base_i, 1]
        
        seg_list.append(seg.detach().cpu())
        x = seg[:, 1, :3].detach().cpu()  # only get the ego part
        history.append(x.clone())

        # EVALUATION
        debug_dt = debug_t2 - debug_t1
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(maze_env.generate_reward_batch(to_np(seg_total[:,0])))

        metrics["t"].append(debug_dt-dt_minus)
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)

    history = torch.stack(history, dim=1)
    seg_list = torch.stack(seg_list, dim=1)
    
    # save evaluation result
    print("Acc:%.3f"%(np.mean(np.array(metrics["acc"]))))
    eval_proc(metrics, "e2_game", args)
    if args.no_viz:
        return
    # visualization
    for ti in range(nt):
        if ti % args.sim_freq == 0 or ti == nt-1:
            print(ti)
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            first_time=True
            for i in range(levels):
                # plot obstacles
                rect0 = Rectangle([env[i, 0], env[i, 1]-args.obs_h], env[i, 2], args.obs_h, color="brown", zorder=5, label="Obstacle" if i==0 else None)
                ax.add_patch(rect0)

                # plot goal
                r = args.goal_w / 2
                if env[i, 3]>=0:
                    ell0 = Ellipse([env[i, 3], env[i, 1]-args.obs_h/2], 2*r, 2*r, color="orange", zorder=8, label="Goal" if first_time else None)
                    ax.add_patch(ell0)
                    first_time=False
            
            # plot trajectory
            ymin = history[0, ti, 2] - args.y_level
            ymax = history[0, ti, 2] + 3 * args.y_level
            ax.axvline(x=0, ymin=ymin, ymax=ymax, linestyle="--", color="gray")
            ax.axvline(x=args.canvas_w, ymin=ymin, ymax=ymax, linestyle="--", color="gray", label="Boundary")
            
            ax.plot(history[0,:ti+1,0], history[0,:ti+1,2], color="green", linewidth=4, alpha=0.5, zorder=10, label="Trajectory")
            ax.plot(seg_list[0, ti, :, 0], seg_list[0, ti, :, 2], color="gray", linewidth=4, alpha=0.5, zorder=10, label="MPC")
            ax.axis("scaled")
            ax.set_xlim(0, args.canvas_w)
            ax.set_ylim(ymin, ymax)
            ax.legend(loc="upper right")
            figname="%s/t_%03d.png"%(args.viz_dir, ti)
            plt.title("Simulation (%04d/%04d)"%(ti, nt))
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            fs_list.append(figname)
    os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
    generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)


def back_solve(x_input, output_0, policy, stl):
    nt0 = 4
    k = 20

    actions = np.linspace(-10, 10, k)
    li = [actions] * nt0
    a_seq = np.array(np.meshgrid(*li)).T.reshape(-1, nt0)
    a_seq = torch.from_numpy(a_seq).float().cuda()

    a_seq = torch.cat([a_seq, output_0[:, :nt0]], dim=0)

    x_input_mul = torch.tile(x_input, [a_seq.shape[0], 1])
    seg0 = dynamics(x_input_mul, a_seq, include_first=False)
    x_new = seg0[:, -1]
    
    x_new_input = x_new.clone()
    base_x = x_new_input[:, 0]
    x_new_input[:, 0] = x_new_input[:, 0] - base_x
    x_new_input[:, 2] = x_new_input[:, 2] - base_x

    u_output = policy(x_new_input).detach()
    seg1 = dynamics(x_new, u_output[:, :args.nt-nt0], include_first=False)
    
    seg = torch.cat([seg0, seg1], dim=1)
    score = stl(seg, args.smoothing_factor)[:, :1]
    idx = torch.argmax(score, dim=0).item()
    
    u_merged = torch.cat([a_seq[idx, :],u_output[idx, :args.nt-nt0]], dim=0).unsqueeze(0)
    return u_merged


def main():
    utils.setup_exp_and_logger(args, test=args.test)

    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        state_dict = torch.load(utils.find_path(args.net_pretrained_path))
        net.load_state_dict(state_dict)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # speed: v_min, v_max
    # canvas width: canvas_w
    # obstacle height: obs_h
    # obstacle width: obs_w_min, obs_w_max
    # y-roll-velocity
    # goal-radius
    # (x, vx, dy, dx0, L0, G0, dx1, L1, G1)
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
    x_init = initialize_x()

    print(stl)
    stl.update_format("word")
    print(stl)
    relu = nn.ReLU()
    if args.test:
        rl_policy = None
        if args.rl:
            from stable_baselines3 import SAC, PPO, A2C
            rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
        elif args.mbpo or args.pets:
            rl_policy = get_mbrl_models(get_exp_dir()+"/"+args.rl_path, args, args.mbpo)
        test_game(net, rl_policy, stl)
        exit()

    for epi in range(args.epochs):
        x0 = x_init.detach()
        u = net(x0)        
        seg = dynamics(x0, u, include_first=True)
        
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc)

        if args.extra:
            x1 = seg[:, 1].detach()
            u1 = net(x1)
            seg1 = dynamics(x1, u1, include_first=True)
            
            score1 = stl(seg1, args.smoothing_factor)[:, :1]
            acc1 = (stl(seg1, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
            acc1_avg = torch.mean(acc1)

            loss_sim = torch.mean(torch.square(u[:,1:]-u1[:,:-1]))
            loss_stl = torch.mean(relu(0.5-score))
            loss_stl1 = torch.mean(relu(0.5-score1))
            loss = loss_stl + loss_stl1 + loss_sim
        else:
            loss = torch.mean(relu(0.5-score))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epi % args.print_freq == 0:
            if args.extra:
                print("%03d  loss:%.3f stl:%.3f stl1:%.3f sim:%.3f acc:%.3f acc1:%.3f" % (
                    epi, loss.item(), loss_stl.item(), loss_stl1.item(), loss_sim.item(), acc_avg.item(), acc1_avg.item()))
            else:
                print("%03d  loss:%.3f acc:%.3f" % (epi, loss.item(), acc_avg.item()))

        # Save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))
        
        if epi % args.viz_freq == 0 or epi == args.epochs - 1:
            init_np = to_np(x_init)
            seg_np = to_np(seg)
            acc_np = to_np(acc)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=25)
    add("--dt", type=float, default=0.2)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)   

    add("--canvas_w", type=float, default=5)
    add("--dy_min", type=float, default=1.5)
    add("--dy_max", type=float, default=2.5)
    add("--goal_w", type=float, default=0.8)
    add("--obs_h", type=float, default=0.5)
    add("--obs_w_min", type=float, default=2.0)
    add("--obs_w_max", type=float, default=4)
    add("--v_min", type=float, default=-5)
    add("--v_max", type=float, default=5)
    add("--only_two", action='store_true', default=False)
    add("--sim_freq", type=int, default=1)
    add("--extra", action='store_true', default=False)
    add("--vy", type=float, default=1.0)
    add("--obs_ratio", type=float, default=1.0)
    add("--y_level", type=float, default=2.5)

    add("--mpc", action='store_true', default=False)
    add("--mpc_freq", type=int, default=1)
    
    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)

    add("--plan", action="store_true", default=False)
    add("--rl", action="store_true", default=False)
    add("--rl_path", "-R", type=str, default=None)
    add("--rl_stl", action="store_true", default=False)
    add("--rl_acc", action="store_true", default=False)
    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)

    add("--eval_path", type=str, default="eval_result")
    add("--no_viz", action="store_true", default=False)

    add("--finetune", action="store_true", default=False)
    add("--cem", action='store_true', default=False)
    args = parser.parse_args()
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))

