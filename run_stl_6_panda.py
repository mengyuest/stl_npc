from lib_stl_core import *
from matplotlib.patches import Polygon, Rectangle, Ellipse, Circle
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.size': 12})

import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, to_torch, xxyy_2_Ab, get_exp_dir, eval_proc, xyzr_2_Ab, xxyyzz_2_Ab, build_relu_nn
import casadi
import matplotlib.pylab as pl
import pytorch_kinematics as pk
from forwardkinematics.urdfFks.pandaFk import PandaFk
from lib_cem import solve_cem_func
from utils_mbrl import get_mbrl_models, get_mbrl_u


DOF = 7

# define the fk Function
q_ca = casadi.SX.sym("q", 7)
fk_panda = PandaFk()
fk_casadi = fk_panda.fk(q_ca, "panda_link8", positionOnly=False)
global_f = casadi.Function('f',[q_ca],\
    [fk_casadi[0:3,3]],\
    ['q'],['xyz'])

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.net = build_relu_nn(DOF+3, args.nt * DOF, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        u = self.net(x)
        u = u.reshape(list(u.shape)[:-1]+[args.nt, DOF])
        amin = -args.u_max
        amax = args.u_max
        if self.args.no_tanh:
            return torch.clip(u, amin, amax)
        else:
            return torch.tanh(u) * (amax - amin) / 2 + (amax  + amin) / 2

def dynamics(x0, u, include_first=False):
    # input:  th, x, y, z  # 3 + 3 * 2 = 9
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
    for i in range(DOF):
        new_x[:, i] = x[:, i] + u[:, i] * args.dt
    new_x[:, DOF:DOF+3] = x[:, DOF:DOF+3]
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
            u = torch.from_numpy(u * args.u_max)
        else:
            if args.mbpo:
                u = get_mbrl_u(x, None, policy, mbpo=True)
            elif args.pets:
                u_list=[]
                for iii in range(x.shape[0]):
                    u = get_mbrl_u(x[iii], None, policy, mbpo=False)
                    u_list.append(u)
                u = torch.stack(u_list, dim=0)
            u = u * args.u_max
        u = torch.clip(u, -args.u_max, args.u_max).cuda()
        new_x = dynamics_per_step(x, u)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti>0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)
    return xs, us, dt_minus

def gradient_solve(chain, tti, x_init, stl):
    relu = torch.nn.ReLU()
    u_lat = torch.zeros(x_init.shape[0], args.nt, DOF).cuda().requires_grad_()
    optimizer = torch.optim.Adam([u_lat], lr=args.grad_lr)
    tt1=time.time()
    prev_loss = None
    for i in range(args.grad_steps):
        u = torch.clamp(u_lat * args.u_max, -args.u_max, args.u_max)
        seg = dynamics(x_init, u, include_first=True)
        seg_ee = endpoint(chain, seg)
        seg_aug = torch.cat([seg, seg_ee], dim=-1)
        score = stl(seg_aug, args.smoothing_factor)[:, :1]
        acc = (stl(seg_aug, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        
        stl_loss = torch.mean(relu(0.5-score))
        dist = dist_to_goal_vec(seg_aug[..., DOF+3:DOF+6], seg_aug[..., DOF:DOF+3])
        goal_loss = torch.mean(dist) * 1
        loss = stl_loss + goal_loss

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
    return u.detach().cuda()


def check_collision(xyz, r):
    collide = r**2 >= (xyz[..., 0]-obs_x)**2+(xyz[..., 1]-obs_y)**2+(xyz[..., 2]-obs_z)**2
    return collide

def endpoint(chain, s, all_points=False, reverse_cat=False):
    shape = s.shape
    s_2d = s.reshape(-1, s.shape[-1])
    M = chain.forward_kinematics(s_2d[..., :DOF], end_only=not all_points)
    if all_points:
        if reverse_cat:
            res = torch.stack([M[mk].get_matrix()[..., :3, 3] for mk in list(M)[::-1]], dim=-2)
            return res.reshape(list(s.shape[:-1]) + [len(M)*3,])
        else:
            res = torch.stack([M[mk].get_matrix()[..., :3, 3] for mk in M], dim=0)
            return res.reshape([len(M),]+list(s.shape[:-1]) + [3,] )
    else:  # end_only=True
        res = M.get_matrix()[..., :3, 3]
        return res.reshape(list(s.shape[:-1]) + [3,] )


def add_object(config, xyz, euler, color, penetrate=False):
    import pybullet as p
    if penetrate:
        obs_id_coll = -1
    if config[0] == p.GEOM_BOX:
        if not penetrate:
            obs_id_coll = p.createCollisionShape(p.GEOM_BOX, halfExtents=config[1:4])
        obs_id_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=config[1:4], rgbaColor=color)
    elif config[0] == p.GEOM_CYLINDER:
        if not penetrate:
            obs_id_coll = p.createCollisionShape(p.GEOM_CYLINDER, height=config[1], radius=config[2])
        obs_id_vis = p.createVisualShape(p.GEOM_CYLINDER, length=config[1], radius=config[2], rgbaColor=color)
    elif config[0] == p.GEOM_SPHERE:
        if not penetrate:
            obs_id_coll = p.createCollisionShape(p.GEOM_SPHERE, radius=config[1])
        obs_id_vis = p.createVisualShape(p.GEOM_SPHERE, radius=config[1], rgbaColor=color)
    else:
        raise NotImplementedError
            
    obs_id = p.createMultiBody(baseMass=0,
                baseCollisionShapeIndex=obs_id_coll,
                baseVisualShapeIndex=obs_id_vis,
                basePosition=xyz,
                baseOrientation=p.getQuaternionFromEuler(euler))

    return obs_id

def initialize_x(chain):
    total_sum = 0
    n_try = 0
    x_list = []
    while total_sum <= args.num_samples:
        n_try+=1
        x = initialize_x_cycle()
        xyz = endpoint(chain, x)
        collide0 = check_collision(xyz, obs_r)
        collide1 = check_collision(x[..., DOF:DOF+3], obs_r)
        safe_idx = torch.where(torch.logical_and(collide0==False, collide1==False))
        x = x[safe_idx]
        total_sum += x.shape[0]
        x_list.append(x)
    x = torch.cat(x_list, dim=0)[:args.num_samples]
    return x

def initialize_x_cycle():
    n = args.num_samples
    q = uniform_tensor(0, 1, (n, 7))
    q_min = torch.from_numpy(thmin).float() 
    q_max = torch.from_numpy(thmax).float() 
    q = q * (q_max-q_min) + q_min
    
    xx = uniform_tensor(-0.5, 0.5, (n, 1))
    yy = uniform_tensor(-0.5, 0.5, (n, 1))
    zz = uniform_tensor(0.2, 0.8, (n, 1))

    x_init = torch.cat([q, xx,yy,zz], dim=-1).cuda()
    return x_init

def dist_to_goal_vec(xyz0, xyz1):
    dist = torch.norm(xyz0-xyz1, dim=-1)
    return dist


def solve_sample_mpc(chain, x_input, ti):
    x_init = x_input
    obs_vec = torch.tensor([obs_x, obs_y, obs_z]).cuda()
    goal_vec = x_init[0, DOF:DOF+3]
    
    min_loss = None
    min_u = None
    ntrials = 50
    for ni in range(ntrials):
        n_rand = 500000
        x_mul = x_init.tile(n_rand, 1).cpu()
        seg = [x_mul]
        us = []
        eps = 1e-2
        thmin_th = torch.from_numpy(thmin) + eps
        thmax_th = torch.from_numpy(thmax) - eps
        for i in range(args.nt):
            umin = (thmin_th - x_mul[:, :DOF])/args.dt
            umax = (thmax_th - x_mul[:, :DOF])/args.dt
            umin = torch.clamp(umin, -args.u_max, args.u_max)
            umax = torch.clamp(umax, -args.u_max, args.u_max)
            u = torch.rand(n_rand, DOF) * (umax-umin) + umin
            x_mul_next = x_mul * 1.0
            x_mul_next[:, :DOF] = x_mul[:, :DOF] + u * args.dt
            us.append(u)
            seg.append(x_mul_next)
            x_mul = x_mul_next
        seg = torch.stack(seg, dim=1).cuda()
        u = torch.stack(us, dim=1).cuda()
        ee = endpoint(chain, seg)
        
        avoid_obstacle_loss = torch.nn.ReLU()(obs_r + eps- torch.min(torch.norm(ee - obs_vec, dim=2), dim=1)[0])
        goal_reach_loss = torch.mean(torch.norm(ee - goal_vec, dim=2), dim=-1)
        loss = avoid_obstacle_loss + goal_reach_loss

        min_idx = torch.argmin(loss, dim=0)
        u_optim = u[min_idx]
        if min_loss is None or loss[min_idx] < min_loss:
            min_loss = loss[min_idx]
            min_u = u_optim 
    u_np = to_np(min_u)
    return u_np

def solve_casadi(chain, x_input, ti):
    import casadi
    x_init = to_np(x_input)
    u = None

    DIRECT_ASSIGNMENT = False

    mpc_t1 = time.time()
    opti = casadi.Opti()
    mpc_max_iters = 10000
    quiet = True
    
    x = opti.variable(args.nt + 1, DOF)  # x, vx, dy
    u = opti.variable(args.nt, DOF)  # a
    mid = opti.variable(args.nt, DOF)
    ee = opti.variable(args.nt+1, 3)  # a
    gamma = opti.variable(args.nt, DOF)

    for i in range(DOF):
        opti.subject_to(x[0, i] == x_init[0, i])

    # control constraints
    opti.subject_to(opti.bounded(-args.u_max, u, args.u_max))

    goal_dist = 0

    for ti in range(args.nt):
        # forward kinematics
        if DIRECT_ASSIGNMENT:
            ee[ti+1, 0:3] = global_f(x[ti+1,:]).T
        else:
            opti.subject_to(ee[ti+1, 0:3].T == global_f(x[ti+1,:]))

        for i in range(DOF):
            # dynamics
            if DIRECT_ASSIGNMENT:
                x[ti+1, i] = x[ti,i] + u[ti, i]
            else:
                opti.subject_to(x[ti+1, i] == x[ti,i] + u[ti, i])

            # angle constraints
            opti.subject_to(opti.bounded(thmin[i], x[ti+1, i], thmax[i]))
            
            # safety constraints
            opti.subject_to((ee[ti,0]-obs_x)**2+(ee[ti,1]-obs_y)**2+(ee[ti,2]-obs_z)**2>obs_r**2 + gamma[ti, i])

            # goal reaching objective
            goal_dist += ((ee[ti,0]-x_init[0, DOF])**2+(ee[ti,1]-x_init[0, DOF+1])**2+(ee[ti,2]-x_init[0, DOF+2])**2)**0.5

    loss = casadi.sumsqr(gamma) * 10000 + goal_dist

    opti.minimize(loss)
    p_opts = {"expand": True}
    s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)
    # sol1 = opti.solve()
    try:
        sol1 = opti.solve()
    except Exception as e: 
        print("An exception occurred")

    x_np = opti.debug.value(x).reshape((args.nt+1, DOF))
    u_np = opti.debug.value(u).reshape((args.nt, DOF))
    return u_np  #, x_np, info

def solve_planner(chain, tti, x_init):
    from lib_pwlplan import plan, Node
    from gurobipy import GRB
    tt1=time.time()

    TRACK_OPTION = "rand"  # (mpc, rand, grad, cem)

    def func1(m, PWLs, di):
        for i in range(args.nt):
            m.addConstr(PWLs[0][i+1][1] - PWLs[0][i][1] == args.dt)

        goal_x, goal_y, goal_z = x_init[0, DOF:DOF+3]
        return 10000 * sum((PWLs[0][i+1][0][0]-goal_x)**2 for i in range(args.nt)) +\
            10000 * sum((PWLs[0][i+1][0][1]-goal_y)**2 for i in range(args.nt)) +\
            10000 * sum((PWLs[0][i+1][0][2]-goal_z)**2 for i in range(args.nt))
    
    nt = args.nt + 1
    dt = args.dt
    tmax = (nt+1)*dt
    vmax = 10
    bloat_r = -0.1

    x_ee = endpoint(chain, x_init)
    x_np = to_np(x_init)

    obs_A0, obs_b0 = xyzr_2_Ab(obs_x, obs_y, obs_z, obs_r, num_edges=8)
    goal_A0, goal_b0 = xyzr_2_Ab(x_np[0, DOF], x_np[0, DOF+1], x_np[0, DOF+2], 0.5*goal_r-0.01, num_edges=8)

    A_map, b_map = xxyyzz_2_Ab([-0.9, 0.9, -0.9, 0.9, -0.4, 1.2])
    in_map = Node("mu", info={"A":A_map, "b":b_map})
    avoid = Node("negmu", info={"A":obs_A0, "b":obs_b0})
    reach = Node("mu", info={"A":goal_A0, "b":goal_b0})
    always_in_map = Node("A", deps=[in_map], info={"int":[0, tmax]})
    always_avoid = Node("A", deps=[avoid], info={"int":[0, tmax]})
    always_reach = Node("A", deps=[reach], info={"int":[0, 3*dt]})
    eventually_reach = Node("F", deps=[always_reach], info={"int":[0, tmax]})

    specs = [Node("and", deps=[always_in_map, always_avoid, eventually_reach])]

    x0s = [np.array([x_ee[0, 0].item(), x_ee[0, 1].item(), x_ee[0, 2].item()])]

    PWL = plan(x0s, specs, bloat=0.01, MIPGap=0.05, num_segs=args.nt, tmax=tmax, vmax=vmax, return_u=False, quiet=True, less_bloat=True, extra_fn_list=[func1])

    if PWL[0] is None:
        print("MILP failed... using dead-reckoning reference trajectory")
        ref_w = torch.from_numpy(np.linspace(0, 1, args.nt+1)[:, None]).to(x_init.device)
        ref_trajs = x_ee * (1-ref_w) + x_init[:, DOF:DOF+3] * ref_w
        ref_trajs = to_np(ref_trajs)
        milp_status = 0
    else:
        ref_trajs = np.array([PWL[0][i][0] for i in range(len(PWL[0]))])
        milp_status = 1

    if TRACK_OPTION == "mpc":
        # convert traj to th commands, using casadi MPC to track
        import casadi
        x_init = to_np(x_init)
        u = None

        mpc_t1 = time.time()
        opti = casadi.Opti()
        mpc_max_iters = 3000
        quiet = True
        
        x = opti.variable(args.nt + 1, DOF)  # x, vx, dy
        u = opti.variable(args.nt, DOF)  # a
        mid = opti.variable(args.nt, DOF)
        ee = opti.variable(args.nt + 1, 3)  # a
        gamma = opti.variable(args.nt, DOF)

        for i in range(DOF):
            opti.subject_to(x[0, i] == x_init[0, i])

        # control constraints
        opti.subject_to(opti.bounded(-args.u_max, u, args.u_max))

        track_err = 0
        for ti in range(args.nt):
            # forward kinematics
            opti.subject_to(ee[ti+1, 0:3].T == global_f(x[ti+1, :]))
            for i in range(DOF):
                # dynamics
                opti.subject_to(x[ti+1, i] == x[ti,i] + u[ti, i])
                # angle constraints
                opti.subject_to(opti.bounded(thmin[i], x[ti+1, i], thmax[i]))
                # tracking error
                track_err += ((ee[ti,0]-ref_trajs[ti+1, 0])**2+(ee[ti,1]-ref_trajs[ti+1, 1])**2+(ee[ti,2]-ref_trajs[ti+1, 2])**2)**0.5

        loss = casadi.sumsqr(gamma) * 10000 + track_err
        opti.minimize(loss)
        p_opts = {"expand": True}
        s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
        if quiet:
            p_opts["print_time"] = 0
            s_opts["print_level"] = 0
            s_opts["sb"] = "yes"
        opti.solver("ipopt", p_opts, s_opts)
        # sol1 = opti.solve()
        try:
            sol1 = opti.solve()
        except Exception as e: 
            print("An exception occurred")

        x_np = opti.debug.value(x).reshape((args.nt+1, DOF))
        u_np = opti.debug.value(u).reshape((args.nt, DOF))
    elif TRACK_OPTION == "grad":
        ref_trajs_cuda = torch.from_numpy(ref_trajs).float().cuda()
        num_iters = 100
        grad_lr = 1e-1
        u_out = torch.zeros(args.nt, DOF).cuda().requires_grad_()
        
        optimizer = torch.optim.Adam([u_out], lr=grad_lr)
        for iter_i in range(num_iters):
            traj_out = [x_init[:, :DOF]]
            for ti in range(args.nt):
                new_th = traj_out[-1] + u_out[ti:ti+1] * args.dt
                traj_out.append(new_th)
            traj_out = torch.stack(traj_out, dim=1)

            ee = endpoint(chain, traj_out)[0]
            loss = torch.mean(torch.norm(ee-ref_trajs_cuda, dim=-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_i % (num_iters//10) == 0 or iter_i==-1:
                print(iter_i, loss.item())

        u_np = to_np(u_out)
    elif TRACK_OPTION == "rand":
        
        min_loss = None
        min_u = None
        ntrials = 10
        for ni in range(ntrials):
            n_rand = 500000
            ref_trajs_cuda = torch.from_numpy(ref_trajs).float().cuda()
            x_mul = x_init.tile(n_rand, 1).cpu()
            seg = [x_mul]
            us = []
            eps = 1e-2
            thmin_th = torch.from_numpy(thmin) + eps
            thmax_th = torch.from_numpy(thmax) - eps
            for i in range(args.nt):
                umin = (thmin_th - x_mul[:, :DOF])/args.dt
                umax = (thmax_th - x_mul[:, :DOF])/args.dt
                umin = torch.clamp(umin, -args.u_max, args.u_max)
                umax = torch.clamp(umax, -args.u_max, args.u_max)
                u = torch.rand(n_rand, DOF) * (umax-umin) + umin
                x_mul_next = x_mul * 1.0
                x_mul_next[:, :DOF] = x_mul[:, :DOF] + u * args.dt
                us.append(u)
                seg.append(x_mul_next)
                x_mul = x_mul_next
            seg = torch.stack(seg, dim=1).cuda()
            u = torch.stack(us, dim=1).cuda()
            ee = endpoint(chain, seg)
            loss = torch.mean(torch.norm(ee-ref_trajs_cuda[None,:], dim=-1), dim=-1)
            min_idx = torch.argmin(loss, dim=0)
            u_optim = u[min_idx]
            if min_loss is None or loss[min_idx] < min_loss:
                min_loss = loss[min_idx]
                min_u = u_optim 
        u_np = to_np(min_u)

    return u_np, ref_trajs, milp_status  #, x_np, info

def solve_cem(chain, ti, x_input, stl, args):
    def dynamics_step_func(x, u):
        new_x = x * 1.0
        for i in range(DOF):
            new_x[:, i] = x[:, i] + u[:, i] * args.dt
        new_x[:, DOF:DOF+3] = x[:, DOF:DOF+3]
        return new_x
    
    def reward_func(trajs_aug):
        return stl(trajs_aug, args.smoothing_factor, d={"hard":True})[:, 0]
    
    def transform_func(trajs, extra_dict):
        trajs_ee = endpoint(chain, trajs)
        trajs_aug = torch.cat([trajs, trajs_ee], dim=-1)
        return trajs_aug

    u_min = torch.from_numpy(np.ones((7,)) * -args.u_max).cuda()
    u_max = torch.from_numpy(np.ones((7,)) * args.u_max).cuda()
    u, _, _ = solve_cem_func(
        x_input[0], state_dim=DOF+3, nt=args.nt, action_dim=DOF,
        num_iters=500, n_cand=1000, n_elites=100, policy_type="direct",
        dynamics_step_func=dynamics_step_func, reward_func=reward_func,
        transform=transform_func, u_clip=(u_min, u_max), seed=None, args=None, 
        extra_dict=None, quiet=False, device="gpu", visualize=False
    )
    return to_np(u)  # (nt, 7)


goal_r = 0.1
obs_x, obs_y, obs_z = 0.3, 0.3, 0.5
obs_r = 0.2
thmin = np.array([-166,-101,-166,-176,-166,-1,-166]) / 180 * np.pi
thmax = np.array([166,101,166,-4,166,215,166]) / 180 * np.pi

def test(chain, net, rl_policy, stl):
    metrics_str=["acc", "reward", "score", "t", "safety", "goals", "valid"]
    prev_metrics = {k:0 for k in metrics_str}
    metrics = {xx:[] for xx in metrics_str}
    from envs.panda_env import PandaEnv
    the_env = PandaEnv(args)

    debug_t1 = time.time()

    if args.no_viz==False and args.pybullet:
        import pybullet as p
        import pybullet_data
        physicsClient = p.connect(p.DIRECT)  #or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(os.path.dirname(__file__) + '/model_description')
        panda_model = "panda_new.urdf"
        robot_id = p.loadURDF(panda_model, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # optionally
        plane_id = p.loadURDF("plane.urdf")
        dof = p.getNumJoints(robot_id) - 1
        print("DOF",dof)
        for i in range(dof):
            p.resetJointState(robot_id, i, targetValue=0)
    
    ntrials = args.test_trials
    nt = args.test_nt
    x_init_list = [initialize_x(chain) for i in range(ntrials)]
    trajs = []
    next_seg_ee_list = []
    ref_trajs_list = []
    for trial_i in range(ntrials):
        valid = 1
        x_init = x_init_list[trial_i]
        x = x_init[0:1]

        cnt = 0
        goal_i = 0
        picked=0
        trajs.append(x)
        for ti in range(nt):
            
            if ti % 5 == 0:
                print(ti)
            if cnt % args.mpc_update_freq==0:
                cnt = 0
                x_input = torch.cat([x[:, :DOF], x_init[goal_i:goal_i+1, DOF:DOF+3]], dim=-1)
                x_input = x_input.cuda()

                dt_minus = 0
                debug_tt1=time.time()
                if valid==1:
                    if args.mpc:
                        if args.sample_mpc:
                            u = solve_sample_mpc(chain, x_input, ti)
                        else:
                            u = solve_casadi(chain, x_input, ti)
                        u = torch.from_numpy(u[None,:]).float().cuda()
                    elif args.rl or args.mbpo or args.pets:
                        _, u, dt_minus = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=True)
                    elif args.grad:
                        u = gradient_solve(chain, ti, x_input, stl)
                    elif args.plan:
                        # TODO (milp planner)
                        # first milp
                        # then fk a lot angles, and pick one that gives the closest L2 dist to the set point
                        u, ref_trajs, milp_status = solve_planner(chain, ti, x_input)
                        u = u[None, :]
                        u = torch.from_numpy(u).float().cuda()
                        ref_trajs_list.append((ref_trajs, milp_status))
                    elif args.cem:
                        u = solve_cem(chain, ti, x_input, stl, args)
                        u = u[None, :]
                        u = torch.from_numpy(u).float().cuda()
                    else:        
                        u = net(x_input)
                        # TODO (finetune)
                        seg = dynamics(x_input, u, include_first=True)
                        if args.finetune:
                            do_nothing = True
                            seg_tmp = seg.cuda()
                            if args.all_joints:
                                seg_tmp_ee = endpoint(chain, seg_tmp, all_points=True, reverse_cat=True)
                            else:
                                seg_tmp_ee = endpoint(chain, seg_tmp, all_points=False)
                            seg_aug_tmp = torch.cat([seg_tmp, seg_tmp_ee], dim=-1)
                            acc_tmp = (stl(seg_aug_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()

                            err_idx = torch.where(acc_tmp<1)[0]
                            if err_idx.shape[0] > 0:
                                ft_n = err_idx.shape[0]
                                print(ti, "[Before] Acc=%.2f, %d do not satisfy STL"%(torch.mean(acc_tmp), ft_n))
                                u_output_fix = u.clone()
                                for iii in range(ft_n):
                                    sel_ = err_idx[iii]
                                    u_fix = back_solve(chain, seg[sel_:sel_+1, 0], u[sel_:sel_+1], net, stl)
                                    u_output_fix[sel_:sel_+1] = u_fix
                                u = u_output_fix
                                seg_fix = dynamics(x_input, u, include_first=True)
                                if args.all_joints:
                                    seg_fix_ee = endpoint(chain, seg_fix, all_points=True, reverse_cat=True)
                                else:
                                    seg_fix_ee = endpoint(chain, seg_fix, all_points=False)
                                seg_fix_aug = torch.cat([seg_fix, seg_fix_ee], dim=-1)
                                acc_tmp_fix = (stl(seg_fix_aug, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                                err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                                ft_n_fix = err_idx_fix.shape[0]
                                print(ti, "[After]  Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
                                seg = seg_fix
                else:
                    u = torch.zeros(1, args.nt, DOF).cuda()
                seg = dynamics(x_input, u, include_first=True)
                debug_tt2=time.time()
                seg_total = seg.clone()
            next_x = seg[:,1+cnt]
            trajs.append(next_x)
            x = next_x
            if args.all_joints:
                next_seg_ee = endpoint(chain, seg_total, all_points=True, reverse_cat=True)
            else:
                next_seg_ee = endpoint(chain, seg_total, all_points=False)
            next_seg_ee_list.append(to_np(next_seg_ee))
            next_ee = endpoint(chain, next_x, all_points=False)
            dist_ee = dist_to_goal_vec(next_ee, next_x[:, DOF:DOF+3])
            
            # EVALUATION
            seg_total_aug = torch.cat([seg_total, next_seg_ee], dim=-1)
            debug_dt = debug_tt2 - debug_tt1 - dt_minus
            score = stl(seg_total_aug, args.smoothing_factor)[:, :1]
            score_avg= torch.mean(score).item()
            acc = (stl(seg_total_aug, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
            acc_avg = torch.mean(acc).item()
            if args.all_joints:
                reward = np.mean(the_env.generate_reward_batch(to_np(seg_total_aug[:])))
            else:
                reward = np.mean(the_env.generate_reward_batch(to_np(seg_total_aug[:])))
            
            safety = (torch.max(check_collision(next_ee, obs_r).float()).item()<=0.5) * 1.0
            
            metrics["t"].append(debug_dt)
            metrics["safety"].append(safety)
            metrics["acc"].append(acc_avg)
            metrics["score"].append(score_avg)
            metrics["reward"].append(reward)
            metrics["goals"].append(picked)
            metrics["valid"].append(valid)

            for key in metrics:
                if key!="valid":
                    metrics[key][-1] = metrics[key][-1] * valid + (1-valid) * prev_metrics[key]
                    prev_metrics[key] = metrics[key][-1]

            if safety == 0:
                valid = 0

            print("t=%03d picked:%3d dist:%.3f [%d] valid:%.2f safe:%.3f (%.3f) acc:%.3f"%(ti, picked, dist_ee, cnt, valid, metrics["safety"][-1],
                    np.mean(np.array(metrics["safety"]) * np.array(metrics["valid"])) / np.mean(np.array(metrics["valid"])),
                    np.mean(np.array(metrics["acc"]) * np.array(metrics["valid"])) / np.mean(np.array(metrics["valid"]))
                    ))
            cnt+=1
            if dist_ee < goal_r:
                picked += 1
                goal_i += 1
                cnt = 0

    accum_acc = np.cumsum(np.array(metrics['acc']) * np.array(metrics["valid"])) / np.cumsum(np.array(metrics["valid"]))
    accum_safety = np.cumsum(np.array(metrics['safety']) * np.array(metrics["valid"])) / np.cumsum(np.array(metrics["valid"]))
    
    metrics_avg = {xx:np.mean(np.array(metrics[xx]) * np.array(metrics["valid"])) / np.mean(np.array(metrics["valid"])) 
                    for xx in metrics}
    metrics_avg["safety"] = np.mean(np.array(metrics["safety"]).reshape((ntrials, nt))[:, -1])
    metrics_avg["goals"] = np.mean(np.array(metrics["goals"]).reshape((ntrials, nt))[:, -1])
    print("safe:%.3f goals:%.3f"%(metrics_avg["safety"] , metrics_avg["goals"] ))
    if args.no_eval==False:
        eval_proc(metrics, "e6_panda", args, metrics_avg)

    trajs = torch.stack(trajs, dim=1)
    seg_np = to_np(trajs)
    edpt = endpoint(chain, trajs)
    edpt_np = to_np(edpt)
    edpt_full = endpoint(chain, trajs, all_points=True)
    edpt_full_np = to_np(edpt_full)

    if args.no_viz==False:
        fs_list = []
        fs_list1 = []
        obs_id = None
        goal_id = None
        NT = len(next_seg_ee_list)
        print(seg_np.shape, edpt.shape, edpt_full_np.shape, NT)
        seg_np = seg_np.reshape((args.test_trials, args.test_nt+1, -1))[:, 1:].reshape((1, args.test_trials* args.test_nt, -1))
        edpt = edpt.reshape((args.test_trials, args.test_nt+1, -1))[:, 1:].reshape((1, args.test_trials* args.test_nt, -1))
        edpt_full_np = edpt_full_np.reshape((len(edpt_full_np), args.test_trials, args.test_nt+1, -1))[:, :, 1:].reshape((len(edpt_full_np), 1, args.test_trials* args.test_nt, -1))
        ttti = 0
        for ti in range(NT):
            if ti % args.sim_freq == 0:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(1, 2, 1)
                circ = Circle([seg_np[0, ti, DOF], seg_np[0, ti, DOF+1]], radius=goal_r, color="green", alpha=0.5)
                ax.add_patch(circ)
                circ = Circle([obs_x, obs_y], radius=obs_r, color="gray", alpha=0.5)
                ax.add_patch(circ)
                plt.scatter(edpt_np[0,ti,0], edpt_np[0,ti,1], color="blue")
                plt.plot(next_seg_ee_list[ti][0,:,0], next_seg_ee_list[ti][0,:,1], color="brown")

                if args.plan:
                    if len(ref_trajs_list)>ti and len(ref_trajs_list[ti])>1:
                        plt.plot(ref_trajs_list[ti][0][:,0], ref_trajs_list[ti][0][:,1], color="green" if ref_trajs_list[ti][1] else "red", linestyle="--")

                plt.axis("scaled")
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)

                ax = fig.add_subplot(1, 2, 2, projection='3d')
                for ii in range(len(edpt_full_np)-1):
                    ax.plot3D(
                        [edpt_full_np[ii][0, ti, 0], edpt_full_np[ii+1][0, ti, 0]], 
                        [edpt_full_np[ii][0, ti, 1], edpt_full_np[ii+1][0, ti, 1]], 
                        [edpt_full_np[ii][0, ti, 2], edpt_full_np[ii+1][0, ti, 2]], color="blue", linewidth=3)
                ax.scatter3D(edpt_np[0, ti, 0], edpt_np[0, ti, 1], edpt_np[0, ti, 2], color="red", alpha=0.5)
                ax.scatter3D(seg_np[0, ti, DOF], seg_np[0, ti, DOF+1], seg_np[0, ti, DOF+2], color="green", alpha=0.5)
                if args.plan:
                    if len(ref_trajs_list)>ti and len(ref_trajs_list[ti])>1:
                        ax.plot3D(ref_trajs_list[ti][0][:,0], ref_trajs_list[ti][0][:,1], ref_trajs_list[ti][0][:,2], color="green" if ref_trajs_list[ti][1] else "red", linestyle="--")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-0.5, 1.5)

                figname="%s/iter_%05d.png"%(args.viz_dir, ti)
                plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
                plt.close()

                fs_list.append(figname)

                if args.pybullet:
                    # remove prev ones
                    if obs_id is not None:
                        p.removeBody(obs_id)
                    if goal_id is not None:
                        p.removeBody(goal_id)
                    # put robot
                    for i in range(dof):
                        p.resetJointState(robot_id, i, targetValue=seg_np[0, ti, i])
                    
                    # put obstacle
                    config = [p.GEOM_SPHERE, obs_r]
                    xyz = [obs_x, obs_y, obs_z]
                    euler = [0, 0, 0]
                    color = [0.9, 0.0, 0.0, 1]
                    obs_id = add_object(config, xyz, euler, color)

                    # put goal point
                    config = [p.GEOM_SPHERE, goal_r]
                    xyz = [seg_np[0, ti, DOF], seg_np[0, ti, DOF+1], seg_np[0, ti, DOF+2]]
                    euler = [0, 0, 0]
                    color = [0.0, 0.9, 0.0, 1]
                    goal_id = add_object(config, xyz, euler, color)

                    plt.figure(figsize=(6, 6))
                    camTargetPos = [0,0,0.5]
                    camDistance = 3
                    roll = 0
                    yaw = 30
                    pitch = -20
                    
                    upAxisIndex = 2
                    viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)

                    height = 960
                    width = 960
                    nearPlane = 0.001
                    farPlane = 100
                    fov = 30
                    aspect = width / height
                    projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)

                    images = p.getCameraImage(width, height, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix)
                    images = np.reshape(images[2], (height, width, 4)) * 1. / 255.
                    plt.imshow(images)
                    plt.axis("off")
                    
                    if metrics["valid"][ttti] == 0:
                        red_patch = Rectangle([0, 0], width, height, color="red", alpha=0.2, zorder=9999)
                        ax = plt.gca()
                        ax.add_patch(red_patch)
                        plt.title("Simulation %02d/%02d STL Acc:%.2f CRASHED"%(ti, NT, accum_acc[ttti]))
                    else:
                        plt.title("Simulation %02d/%02d STL Acc:%.2f Safety:%.2f"%(ti, NT, accum_acc[ttti], accum_safety[ttti]))
                    figname1="%s/bullet_%05d.png"%(args.viz_dir, ti)
                    plt.savefig(figname1, bbox_inches='tight', pad_inches=0.1)
                    plt.close()
                    fs_list1.append(figname1)
                    ttti+=1

        os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
        generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)
        if args.pybullet:
            generate_gif('%s/animation/bullet.gif'%(args.viz_dir), 0.1, fs_list1)
    debug_t2 = time.time()
    print("Finished in %.2f seconds"%(debug_t2 - debug_t1))

def back_solve(chain, x_input, output_0, policy, stl):
    nt0 = 5
    N = 200000
    du_max = 2.0
    du_min = -2.0
    a_seq = torch.rand(N, nt0, DOF) * (du_max - du_min) + du_min
    a_seq = a_seq.float().cuda()
    a_seq = torch.cat([a_seq, output_0[:, :nt0]], dim=0)
    a_seq = torch.clamp(a_seq, -args.u_max, args.u_max)

    x_input_mul = torch.tile(x_input, [a_seq.shape[0], 1])
    seg0 = dynamics(x_input_mul, a_seq, include_first=False)
    x_new = seg0[:, -1]

    u_output = policy(x_new).detach()
    seg1 = dynamics(x_new, u_output[:, :args.nt-nt0], include_first=False)
    
    seg = torch.cat([seg0, seg1], dim=1)
    seg_ee = endpoint(chain, seg, all_points=False)
    seg_aug = torch.cat([seg, seg_ee], dim=-1)
    score = stl(seg_aug, args.smoothing_factor)[:, :1]
    idx = torch.argmax(score, dim=0).item()
    u_merged = torch.cat([a_seq[idx, :],u_output[idx, :args.nt-nt0]], dim=0).unsqueeze(0)
    return u_merged

def time_shift(x, alpha):
    xt_1 = torch.cat([x[..., 0:1, :], x[..., :-1, :]], dim=-2)
    return x * alpha + xt_1 * (1-alpha)  # x_{t-1} is from prev step; alpha=1 means only care about current step; alpha=0 only prev step

def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)  
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        state_dict = torch.load(utils.find_path(args.net_pretrained_path))
        net.load_state_dict(state_dict)
    
    chain = pk.build_serial_chain_from_urdf(open("model_description/panda.urdf").read(), "panda_link8")
    chain = chain.to(device="cuda")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    obs_vec = torch.tensor([obs_x, obs_y, obs_z]).cuda()
    reach_goal = Eventually(0, args.nt, Always(0, args.nt, AP(lambda x_aug: goal_r - dist_to_goal_vec(x_aug[..., DOF+3:DOF+6], x_aug[..., DOF:DOF+3]), comment="REACH1")))
    if args.all_joints:
        avoid = []
        for ii in range(args.n_cares):
            avoid.append(Always(0, args.nt, AP(lambda x_aug: - obs_r + dist_to_goal_vec(x_aug[..., DOF+3*ii+3:DOF+3*ii+6], obs_vec), comment="AVOID_%d"%(ii))))
            if args.extra_steps>0:
                ap_func = lambda alpha: Always(0, args.nt, AP(lambda x_aug: - obs_r + dist_to_goal_vec(time_shift(x_aug[..., DOF+3*ii+3:DOF+3*ii+6], alpha), obs_vec), comment="AVOID_%d_%.2f"%(ii, alpha)))
                alpha_list = [0.5, 0.25, 0.75, 0.1, 0.9, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8]
                for alpha in alpha_list[:args.extra_steps]:
                    avoid.append(ap_func(alpha))
    else:
        avoid = [Always(0, args.nt, AP(lambda x_aug: - obs_r + dist_to_goal_vec(x_aug[..., DOF+3:DOF+6], obs_vec), comment="AVOID"))]
    angle_cons_fn=lambda i: Always(0, args.nt, AP(lambda x: (thmax[i]-x[..., i])*(x[...,i]-thmin[i]), comment="q%d"%(i)))
    angle_cons_list=[]
    for i in range(7):
        angle_cons_list.append(angle_cons_fn(i))
    stl = ListAnd([reach_goal] + avoid + angle_cons_list)

    x_init = initialize_x(chain)
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
        test(chain, net, rl_policy, stl)
        return

    for epi in range(args.epochs):
        eta.update()
        if args.update_init_freq >0 and epi % args.update_init_freq == 0 and epi!=0:
            x_init = initialize_x(chain)
        x0 = x_init.detach()
        u = net(x0)
        seg = dynamics(x0, u, include_first=True)

        if args.all_joints:
            seg_ee = endpoint(chain, seg, all_points=True, reverse_cat=True)
        else:
            seg_ee = endpoint(chain, seg, all_points=False)
        seg_aug = torch.cat([seg, seg_ee], dim=-1)

        score = stl(seg_aug, args.smoothing_factor)[:, :1]
        acc = (stl(seg_aug, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc)
        dist = dist_to_goal_vec(seg_aug[..., DOF+3:DOF+6], seg_aug[..., DOF:DOF+3])
        stl_loss = torch.mean(relu(0.5-score))
        goal_loss = torch.mean(dist) * 1
        loss = stl_loss + goal_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epi % args.print_freq == 0:
            print("%s|%06d  loss:%.3f stl:%.3f goal:%.3f acc:%.3f  dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), stl_loss.item(), goal_loss.item(), acc_avg.item(), eta.interval_str(), eta.elapsed_str(), eta.eta_str()))
        
        # Save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))
        
        if epi % args.viz_freq == 0 or epi == args.epochs - 1:
            init_np = to_np(x_init)
            seg_aug_np = to_np(seg_aug)
            acc_np = to_np(acc)

            plt.figure(figsize=(12, 12))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=1000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=5e-4)
    add("--nt", type=int, default=10)
    add("--dt", type=float, default=0.1)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)   

    add("--u_max", type=float, default=4.0)

    add("--sim_freq", type=int, default=1)
    add("--update_init_freq", type=int, default=-1)
    add("--mpc_update_freq", type=int, default=1)

    add("--pybullet", action='store_true', default=False)

    # test 
    add("--mpc", action='store_true', default=False)

    add("--grad", action="store_true", default=False)
    add("--grad_lr", type=float, default=0.10)
    add("--grad_steps", type=int, default=200)
    add("--grad_print_freq", type=int, default=10)

    add("--plan", action="store_true", default=False)
    add("--hiddens", type=int, nargs="+", default=[256,256,256])

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

    add("--test_trials", type=int, default=5)
    add("--test_nt", type=int, default=50)

    add("--no_tanh", action="store_true", default=False)
    add("--sample_mpc", action='store_true', default=False)

    add("--cem", action='store_true', default=False)

    add("--all_joints", action='store_true', default=False)
    add("--n_cares", type=int, default=7)
    add("--extra_steps", type=int, default=0)

    args = parser.parse_args()
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))