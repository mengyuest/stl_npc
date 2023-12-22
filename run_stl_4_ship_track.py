from lib_stl_core import *
from matplotlib.patches import Polygon, Rectangle, Ellipse
from matplotlib.collections import PatchCollection
plt.rcParams.update({'font.size': 12})

import utils
from utils import to_np, uniform_tensor, rand_choice_tensor, generate_gif, \
            check_pts_collision, check_seg_collision, soft_step, to_torch, pts_in_poly, seg_int_poly, build_relu_nn, get_exp_dir, eval_proc 
from utils import xyr_2_Ab, xxyy_2_Ab
from lib_cem import solve_cem_func
from utils_mbrl import get_mbrl_models, get_mbrl_u


def soft_step(x):
    return (torch.tanh(500 * x) + 1)/2

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        # input (x, y, phi, u, v, r, obs_x, obs_y, obs_r, T)
        # output (acc)
        input_dim = 10
        output_dim = 2 * args.nt
        self.net = build_relu_nn(input_dim, output_dim, args.hiddens, activation_fn=nn.ReLU)
    
    def forward(self, x):     
        num_samples = x.shape[0]
        u = self.net(x).reshape(num_samples, args.nt, -1)
        u0 = torch.tanh(u[..., 0]) * args.thrust_max
        u1 = torch.tanh(u[..., 1]) * args.delta_max
        uu = torch.stack([u0, u1], dim=-1)
        return uu

def dynamics(x0, u, include_first=False):
    t = u.shape[1]
    x = x0.clone()
    if include_first:
        segs=[x0]
    else:
        segs = []
    for ti in range(t):
        new_x = dynamics_s(x, u[:, ti], num=args.stl_sim_steps)
        segs.append(new_x)
        x = new_x
    return torch.stack(segs, dim=1)

def dynamics_s(x, uu, num=1):
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
        new_dT = -soft_step(x[:, 1]**2-args.track_thres**2)

        zeros = 0 * new_dx
        dsdt = torch.stack([new_dx, new_dy, new_dphi, new_du, new_dv, new_dr] + [zeros, zeros, zeros, new_dT], dim=-1)
        new_x = x + dsdt * dt
        new_xx = new_x.clone()
        new_xx[:, 2] = torch.clamp(new_x[:, 2], -args.s_phimax, args.s_phimax)
        new_xx[:, 3] = torch.clamp(new_x[:, 3], args.s_umin, args.s_umax)
        new_xx[:, 4] = torch.clamp(new_x[:, 4], -args.s_vmax, args.s_vmax)
        new_xx[:, 5] = torch.clamp(new_x[:, 5], -args.s_rmax, args.s_rmax)
        
        x = new_xx
    return new_xx

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

        u[..., 0] = torch.clip(u[..., 0] * args.thrust_max, -args.thrust_max, args.thrust_max)
        u[..., 1] = torch.clip(u[..., 1] * args.delta_max, -args.delta_max, args.delta_max)
        u = u.cuda()

        new_x = dynamics_s(x, u, num=args.stl_sim_steps)
        xs.append(new_x)
        us.append(u)
        x = new_x
        tt2=time.time()
        if ti > 0:
            dt_minus += tt2-tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)  # (N, 2) -> (N, T, 2)
    return xs, us, dt_minus

def initialize_x_cycle(n, is_cbf=False):
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
    if is_cbf:
        y = y * 1.2
    res = torch.cat([x, y, phi, u, v, r, obs_x, obs_y, obs_r, obs_T], dim=1)
    return res

def mux(scene_type, x0, x1, x2, x3):
    return (scene_type==0).float() * x0 + (scene_type==1).float() * x1 + (scene_type==2).float() * x2 + (scene_type==3).float() * x3


def initialize_x(n):
    x_list = []
    total_n = 0
    while(total_n<n):
        x_init = initialize_x_cycle(n)
        safe_bloat = args.bloat_d
        dd = 5
        n_res = 100
        crit_list = []
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

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        input_dim = 10
        output_dim = 2
        self.net = build_relu_nn(input_dim, output_dim, args.net_hiddens, activation_fn=nn.ReLU)

    def forward(self, x, k1=None, k2=None, k3=None, k4=None, k5=None):
        num_samples = x.shape[0]
        x_enc = x.clone()
        x_enc[:, 0] = 0
        x_enc[:, 1:6] = x[:, 1:6]
        x_enc[:, 6] = x[:, 6] - x[:, 0]
        x_enc[:, 7] = x[:, 7]
        x_enc[:, 8] = x[:, 8]
        x_enc[:, 9] = x[:, 9]
        u = self.net(x_enc).reshape(num_samples, -1)
        
        if args.test_pid:
            uref0 = - k1 * (x_enc[:, 3] - 4)
            uref1 = - k2 * (x_enc[:, 1] - 0) - k3 * (x_enc[:, 2] - 0) - k4 * (x_enc[:, 3] - 0) - k5* (x_enc[:, 4] - 0)
            u0 = torch.clip(torch.tanh(u[..., 0]) * args.thrust_max * 0 + uref0, -args.thrust_max, args.thrust_max)
            u1 = torch.clip(torch.tanh(u[..., 1]) * args.delta_max * 0 + uref1, -args.delta_max, args.delta_max)
        else:
            uref0 = - 5 * (x_enc[:, 3] - 4)
            uref1 = - 3 * (x_enc[:, 1] - 0) - 5 * (x_enc[:, 2] - 0)
            u0 = torch.clip(torch.tanh(u[..., 0]) * args.thrust_max + uref0, -args.thrust_max, args.thrust_max)
            u1 = torch.clip(torch.tanh(u[..., 1]) * args.delta_max + uref1, -args.delta_max, args.delta_max)
        uu = torch.stack([u0, u1], dim=-1)
        return uu

class CBF(nn.Module):
    def __init__(self, args):
        super(CBF, self).__init__()
        self.args = args
        input_dim = 10
        output_dim = 1
        self.net = build_relu_nn(input_dim, output_dim, args.cbf_hiddens, activation_fn=nn.ReLU)

    def forward(self, x):
        num_samples = x.shape[0]
        x_enc = x.clone()
        x_enc[:, 0] = 0
        x_enc[:, 1:6] = x[:, 1:6]
        x_enc[:, 6] = x[:, 6] - x[:, 0]
        x_enc[:, 7] = x[:, 7]
        x_enc[:, 8] = x[:, 8]
        x_enc[:, 9] = x[:, 9]
        v = torch.tanh(self.net(x_enc))
        tau = args.smoothing_factor
        v_prior1 = torch.clip(torch.norm(x[..., :2] - x[..., 6:8], dim=-1)**2 - x[..., 8]**2, -10, 10) 
        v_prior3 = args.river_w*((args.river_width/2)**2 - (x[..., 1])**2)  
        v_prior = torch.minimum(v_prior1, v_prior3).reshape(x.shape[0], 1)
        return v_prior * args.cbf_prior_w + v * args.cbf_nn_w

def mask_mean(x, mask):
    # TODO comment
    return torch.mean(x * mask) / torch.clip(torch.mean(mask), 1e-4)

def get_masks(x):
    dist1 = torch.norm(x[..., :2] - x[..., 6:8], dim=-1) - x[..., 8]
    dist3 = args.river_width/2 - torch.abs(x[..., 1]) 
    safe_mask = torch.logical_and(dist1>=args.cbf_pos_bloat, dist3>=args.cbf_pos_bloat).float()
    dang_mask = torch.logical_or(dist1<0, dist3<0).float()
    mid_mask = (1 - safe_mask) * (1 - dang_mask)
    return safe_mask, dang_mask, mid_mask

def check_safety(x):
    dist1 = torch.norm(x[..., :2] - x[..., 6:8], dim=-1) - x[..., 8]
    dist3 = args.river_width/2 - torch.abs(x[..., 1]) 
    acc = torch.all(torch.logical_and(dist1>=0, dist3>=0), dim=-1).float()
    inl = torch.all(dist3>=0, dim=-1).float()
    return acc, inl

def check_safety_stl(x):
    dist1 = torch.norm(x[..., :2] - x[..., 6:8], dim=-1) - x[..., 8]
    dist3 = args.river_width/2 - torch.abs(x[..., 1]) 
    acc = torch.all(torch.logical_and(torch.logical_and(dist1>=0, dist3>=0), x[..., 9]>=0), dim=-1).float()
    inl = torch.all(dist3>=0, dim=-1).float()
    return acc, inl

def train_traj_cbf(x_init, eta):
    net = Net(args).cuda()
    cbf = CBF(args).cuda()
    
    if args.alternative or args.alternative2:
        net_optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        cbf_optimizer = torch.optim.Adam(cbf.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(net.parameters()) + list(cbf.parameters()), lr=args.lr)

    relu = nn.ReLU()
    N = args.num_samples 
    T = args.nt
    Nd = args.num_dense_sample
    gamma = args.cbf_gamma
    alpha = args.cbf_alpha

    # state-wise training for cbf classification      
    x_dense = initialize_x_cycle(args.num_dense_sample, is_cbf=True).cuda()

    only_traj_cls = not(args.both_state_cls or args.dense_state_cls)
    only_traj_dec = not(args.both_state_dec or args.dense_state_dec)

    for epi in range(args.epochs):
        eta.update()
        
        # episodic case
        x = x_init.clone()
        us = []
        segs = [x]
        X_DIM = x.shape[-1]
        for ti in range(args.nt):
            u = net(x)
            new_x = dynamics_s(x, u, num=args.num_sim_steps)
            segs.append(new_x)
            us.append(u)
            x = new_x.detach()
        segs = torch.stack(segs, dim=1)  # (N, T, 3)
        us = torch.stack(us, dim=1)  # (N, T, 2)
        all_x_epi = segs
        
        # dense evolution
        curr_x_dense = x_dense
        curr_u_dense = net(curr_x_dense)
        next_x_dense = dynamics_s(curr_x_dense, curr_u_dense, num=args.num_sim_steps)
        all_x_dense = torch.stack([curr_x_dense, next_x_dense], dim=1)

        safe_mask_dense, dang_mask_dense, mid_mask_dense = get_masks(all_x_dense)
        all_v_dense = cbf(all_x_dense.reshape(Nd * 2, X_DIM).detach()).reshape(Nd, 2)
        curr_v_dense = cbf(curr_x_dense).reshape(Nd, 1)
        next_v_dense = cbf(next_x_dense).reshape(Nd, 1)

        # dense classification loss
        loss_cbf_cls_dense = mask_mean(relu(gamma - all_v_dense), safe_mask_dense) + mask_mean(relu(all_v_dense + gamma), dang_mask_dense)

        # dense decreasing loss
        loss_cbf_dec_dense = torch.mean(relu((1 - alpha) * curr_v_dense - next_v_dense))

        # episodic evolution
        safe_mask_epi, dang_mask_epi, mid_mask_epi = get_masks(all_x_epi)
        all_v_epi = cbf(all_x_epi.reshape(N * (T+1), X_DIM)).reshape(N, T+1)
        curr_x_epi = all_x_epi[:, :-1].reshape(N*T, -1)
        next_x_epi = all_x_epi[:, 1:].reshape(N*T, -1)
        curr_v_epi = cbf(curr_x_epi).reshape(N, T)
        next_v_epi = cbf(next_x_epi).reshape(N, T)

        # episodic classification loss
        loss_cbf_cls_epi = mask_mean(relu(gamma - all_v_epi), safe_mask_epi) + mask_mean(relu(all_v_epi + gamma), dang_mask_epi)

        # episodic decreasing loss
        loss_cbf_dec_epi = torch.mean(relu((1 - alpha) * curr_v_epi - next_v_epi))

        if args.both_state_cls:
            loss_cbf_cls = args.dense_ratio * loss_cbf_cls_dense + (1 - args.dense_ratio) * loss_cbf_cls_epi
        elif args.dense_state_cls:
            loss_cbf_cls = loss_cbf_cls_dense
        else:
            loss_cbf_cls = loss_cbf_cls_epi
        
        if args.both_state_dec:
            loss_cbf_dec = args.dense_ratio * loss_cbf_dec_dense + (1 - args.dense_ratio) * loss_cbf_dec_epi
        elif args.dense_state_dec:
            loss_cbf_dec = loss_cbf_dec_dense
        else:
            loss_cbf_dec = loss_cbf_dec_epi
        loss_cbf_cls = loss_cbf_cls * args.cbf_cls_w
        loss_cbf_dec = loss_cbf_dec * args.cbf_dec_w
        u_loss = torch.mean(torch.norm(us, dim=-1)) * args.u_loss + torch.mean(torch.norm(curr_u_dense, dim=-1)) * args.u_loss
        loss = loss_cbf_cls + loss_cbf_dec + u_loss
        
        if args.alternative:
            if epi % (args.alternative_freq * 2) < args.alternative_freq:
                cbf_optimizer.zero_grad()
                loss.backward()
                cbf_optimizer.step()
            else:
                net_optimizer.zero_grad()
                loss.backward()
                net_optimizer.step()
        elif args.alternative2:
            if epi < args.epochs//2:
                cbf_optimizer.zero_grad()
                loss.backward()
                cbf_optimizer.step()
            else:
                net_optimizer.zero_grad()
                loss.backward()
                net_optimizer.step()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc, inl = check_safety(segs)
        acc_avg = torch.mean(acc)
        
        inl_avg = torch.mean(inl)

        if epi % args.print_freq == 0:
            print("%s|%03d loss:%.3f cls:%.3f dec:%.3f u:%.3f acc:%.4f in:%.3f safe%.3f mid:%.3f dang:%.3f s:%.3f m:%.3f d:%.3f| dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), loss_cbf_cls.item(), loss_cbf_dec.item(), u_loss.item(), #loss_debug.item(),
                acc_avg.item(),  inl_avg.item(),
                torch.mean(safe_mask_dense).item(), torch.mean(mid_mask_dense).item(), torch.mean(dang_mask_dense).item(), 
                torch.mean(safe_mask_epi).item(), torch.mean(mid_mask_epi).item(), torch.mean(dang_mask_epi).item(), 
                eta.interval_str(), eta.elapsed_str(), eta.eta_str()))

        # Save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/net_%05d.ckpt"%(args.model_dir, epi))
            torch.save(cbf.state_dict(), "%s/cbf_%05d.ckpt"%(args.model_dir, epi))
        
        if epi % args.viz_freq == 0 or epi == args.epochs - 1:
            init_np = to_np(x_init)
            seg_np = to_np(segs)
            v_np = to_np(all_v_epi)
            acc_np = to_np(acc)
            sim_visualization(epi, init_np, seg_np, acc_np, v_np=v_np)


def sim_visualization(epi, init_np, seg_np, acc_np, v_np=None):
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
            offset = 6
            ax.add_patch(Ellipse([seg_np[idx, 0, offset], seg_np[idx, 0, offset + 1]], seg_np[idx, 0, offset + 2] * 2, seg_np[idx, 0, offset + 2] * 2, 
                                    label="obstacle", color="gray", alpha=0.8))
            ax.add_patch(Ellipse([seg_np[idx, 0, 0], seg_np[idx, 0, 1]], 0.5, 0.5, 
                                        label="ego", color="blue", alpha=0.8))
            plt.plot(seg_np[idx, :, 0], seg_np[idx, :, 1], label="trajectory", color="blue", linewidth=2, alpha=0.5)
            for ti in range(0, args.nt, 2):
                ax.text(seg_np[idx, ti, 0]+0.25, seg_np[idx, ti, 1]+0.25, "%.1f"%(seg_np[idx, ti, -1]), fontsize=6)
            if v_np is not None:
                plt.plot(seg_np[idx, :, 0], v_np[idx, :] * 1.8, label="CBF value (x1.8)", color="red", linewidth=2, alpha=0.3)
            if idx==0:
                plt.legend(fontsize=6, loc="lower right")
            ax.axis("scaled")
            plt.xlim(0-bloat, args.canvas_w+bloat)
            plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)

    figname="%s/iter_%05d.png"%(args.viz_dir, epi)
    plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def solve_mpc(x_init):
    import casadi

    x_init = to_np(x_init)

    u = None
    # avoid obstacle
    mpc_t1 = time.time()
    opti = casadi.Opti()
    mpc_max_iters = 10000
    quiet = True
    
    x = opti.variable(args.nt + 1, 6)  # x, vx, dy
    u = opti.variable(args.nt, 2)  # a
    gamma = opti.variable(args.nt, 4)
    
    obs = x_init[0, 6:9]
    obs_t = x_init[0, 9]
    bloat = 0.1

    # args.thrust_max, args.delta_max
    # s_phimax
    # s_umin, s_umax
    # s_vmax
    # s_rmax
    for i in range(6):
        x[0, i] = x_init[0, i]

    for ti in range(args.nt):
        # TODO small dts multi-steps
        x[ti+1, 0] = x[ti, 0] + (x[ti, 3] * casadi.cos(x[ti, 2]) - x[ti, 4] * casadi.sin(x[ti, 2])) * args.dt
        x[ti+1, 1] = x[ti, 1] + (x[ti, 3] * casadi.sin(x[ti, 2]) + x[ti, 4] * casadi.cos(x[ti, 2])) * args.dt
        x[ti+1, 2] = x[ti, 2] + x[ti, 5] * args.dt
        x[ti+1, 3] = x[ti, 3] + u[ti, 0] * args.dt
        x[ti+1, 4] = x[ti, 4] + u[ti, 1] * 0.01 * args.dt
        x[ti+1, 5] = x[ti, 5] + u[ti, 1] * 0.5 * args.dt

        opti.subject_to(u[ti, 0] <= args.thrust_max)
        opti.subject_to(u[ti, 0] >= -args.thrust_max)

        opti.subject_to(u[ti, 1] <= args.delta_max)
        opti.subject_to(u[ti, 1] >= -args.delta_max)

        opti.subject_to(x[ti+1, 2] <= args.s_phimax)
        opti.subject_to(x[ti+1, 2] >= -args.s_phimax)

        opti.subject_to(x[ti+1, 3] <= args.s_umax)
        opti.subject_to(x[ti+1, 3] >= args.s_umin)

        opti.subject_to(x[ti+1, 4] <= args.s_vmax)
        opti.subject_to(x[ti+1, 4] >= -args.s_vmax)

        opti.subject_to(x[ti+1, 5] <= args.s_rmax)
        opti.subject_to(x[ti+1, 5] >= -args.s_rmax)

        opti.subject_to(x[ti+1, 1] <= args.river_width/2 + gamma[ti, 0])
        opti.subject_to(x[ti+1, 1] >= -args.river_width/2 - gamma[ti, 1])

        # avoid collision
        opti.subject_to((x[ti+1, 0]-obs[0])**2+(x[ti+1, 1]-obs[1])**2 >= bloat + obs[2]**2 - gamma[ti, 2])

        # get back to centerline
        if obs_t < (ti+1)*args.dt:
            opti.subject_to((x[ti+1, 1])**2 <= args.track_thres**2 + gamma[ti, 3])

    loss = casadi.sumsqr(gamma) * 100000 + casadi.sumsqr(u) + casadi.sumsqr(x[:, 1]) * 100
    opti.minimize(loss)

    p_opts = {"expand": True}
    s_opts = {"max_iter": mpc_max_iters, "tol": 1e-5}
    if quiet:
        p_opts["print_time"] = 0
        s_opts["print_level"] = 0
        s_opts["sb"] = "yes"
    opti.solver("ipopt", p_opts, s_opts)
    try:
        sol1 = opti.solve()
    except:
        do_nothing=1
    x_np = opti.debug.value(x)
    u_np = opti.debug.value(u)
    u_np[:, 0] = np.clip(u_np[:, 0], -args.thrust_max, args.thrust_max)
    u_np[:, 1] = np.clip(u_np[:, 1], -args.delta_max, args.delta_max)
    mpc_t2 = time.time()

    print("%.4f seconds"%(mpc_t2-mpc_t1))

    return to_torch(u_np[None, :]).cpu()

def gradient_solve(tti, x_init, stl, multi_test=False):
    relu = torch.nn.ReLU()
    # u_lat = torch.zeros(x_init.shape[0], args.nt).cuda().requires_grad_()
    u_lat = torch.zeros(x_init.shape[0], args.nt, 2).requires_grad_()
    x_init = x_init.cpu()
    optimizer = torch.optim.Adam([u_lat], lr=args.grad_lr)
    tt1=time.time()
    prev_loss = None
    for i in range(args.grad_steps):
        u0 = torch.tanh(u_lat[..., 0]) * args.thrust_max
        u1 = torch.tanh(u_lat[..., 1]) * args.delta_max
        u = torch.stack([u0, u1], dim=-1)
        seg = dynamics(x_init, u, include_first=True)
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        
        dist_loss = torch.mean(seg[..., 1]**2)
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
    return u.detach().cuda()

def solve_planner(tti, iii, x_init):
    from lib_pwlplan import plan, Node
    from gurobipy import GRB
    from utils import default_line, pid_control
    tt1=time.time()

    def func1(m, PWLs, di):
        for i in range(args.nt):
            m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i][0][1] <= 0.2)
            m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i][0][1] >= -0.2)
            if i>=1:
                m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i-1][0][1] <= 0.2)
                m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i-1][0][1] >= -0.2)
            m.addConstr(PWLs[0][i+1][0][0] - PWLs[0][i][0][0] >= 0.3)
            m.addConstr(PWLs[0][i+1][1] - PWLs[0][i][1] == args.dt)
        return 1000 * sum((PWLs[0][i+1][0][1]-PWLs[0][i][0][1])**2 for i in range(nt)) + \
                100*sum((PWLs[0][i+1][0][1])**2 for i in range(nt))


    nt = args.nt
    dt = args.dt
    tmax = (nt+1)*dt
    vmax = 10
    bloat_r = 0.0

    x_np = to_np(x_init)
    obs_A0, obs_b0 = xyr_2_Ab(x_np[0, 6+0], x_np[0, 6+1], x_np[0, 6+2]+bloat_r, num_edges=8)
    A_map, b_map = xxyy_2_Ab([-0.5, 20, -args.river_width/2, args.river_width/2])
    A_track, b_track = xxyy_2_Ab([-0.5, 20, -args.track_thres-bloat_r, args.track_thres+bloat_r])
    
    in_map = Node("mu", info={"A":A_map, "b":b_map})
    within_track = Node("mu", info={"A":A_track, "b":b_track})
    avoid0 = Node("negmu", info={"A":obs_A0, "b":obs_b0})
    always_in_map = Node("A", deps=[in_map], info={"int":[0, tmax]})
    always_avoid0 = Node("A", deps=[avoid0], info={"int":[0, tmax]})
    
    t_remain = x_np[0, -1]
    diverged = np.abs(x_np[0, 1]) > args.track_thres
    if diverged:
        specs = [Node("and", deps=[always_in_map, always_avoid0])]
    else:
        specs = [Node("and", deps=[always_in_map, always_avoid0])]

    tt2=time.time()
    x0s = [np.array([x_init[0, 0].item(), x_init[0, 1].item()])]
    PWL, u_out = plan(x0s, specs, bloat=0.01, MIPGap=0.05, num_segs=args.nt, tmax=tmax, vmax=vmax, extra_fn_list=[func1], return_u=True, quiet=True)
    tt3=time.time()
    
    if PWL[0] is None:
        print("Failed")
        pwl_torch = default_line(x_init, args.nt)
    else:
        pwl_torch = torch.tensor([PWL[0][i][0] for i in range(nt+1)]).unsqueeze(0).cuda()

    ## rollout
    x_sim = x_init.cpu()
    pwl_torch_cpu = pwl_torch.cpu()
    u_segs=[]
    for i in range(args.nt):
        u_out = pid_control(x_sim, pwl_torch_cpu)
        x_sim = dynamics_s(x_sim, u_out, num=args.stl_sim_steps)
        u_segs.append(u_out)
    u_segs = torch.stack(u_segs, dim=1).cuda()

    tt4=time.time()
    return u_segs, pwl_torch

def solve_cem(ti, x_input, stl, args):
    def dynamics_step_func(x, u):
        return dynamics_s(x, u, num=args.stl_sim_steps)
    
    def reward_func(trajs):
        return stl(trajs, args.smoothing_factor, d={"hard":True})[:, 0]

    u_min = torch.tensor([-args.thrust_max, -args.delta_max]).cuda()
    u_max = torch.tensor([args.thrust_max, args.delta_max]).cuda()
    u, _, info = solve_cem_func(
        x_input, state_dim=x_input.shape[-1], nt=args.nt, action_dim=2,
        num_iters=500, n_cand=10000, n_elites=100, policy_type="direct",
        dynamics_step_func=dynamics_step_func, reward_func=reward_func,
        transform=None, u_clip=(u_min, u_max), seed=None, args=None, 
        extra_dict=None, quiet=False, device="gpu", visualize=False
    )
    return u


def ship_backup_demo(x_init, net_stl, net_cbf, rl_policy, stl, stl_safe, stl_list_debug, args):
    # find the dangerous initial states

    nt = args.test_nt
    n_trials = args.num_trials
    n_obs = 20
    obs_dL = 12

    reset_T = rand_choice_tensor(list(range(15, 16)), (n_trials, n_obs)) * args.dt
    obs_list=[]
    obs_x = x_init[0:1, 6]
    for i in range(n_obs):
        if i>0:
            obs_x = obs_x + obs_dL
            obs_y = 0 * obs_x + 0.2
        else:
            obs_x  = obs_x * 0 + x_init[0:1, 0] + 4
            obs_y = 0 * obs_x + 0.25
        obs_r = obs_x * 0 + uniform_tensor(args.obs_rmin, args.obs_rmax, (1, )).cuda()
        if i==1:
            obs_r = 1.5*obs_r
        reset_T[:, i] = rand_choice_tensor([20, 25], (1, )).cuda() * args.dt

        obs = torch.cat([obs_x, obs_y, obs_r], dim=-1)
        obs_list.append(obs)

    obs_map = torch.stack(obs_list, dim=0)  # (M, 3)


    x_init[0:n_trials, 1] = uniform_tensor(args.obs_rmin-0.3, args.river_width/2-0.2, (n_trials,)).cuda() * rand_choice_tensor([-1,1.0], (n_trials, )).cuda()
    x_init[0:n_trials, 2] = 0
    x_init[0:n_trials, 3] = 5.0
    x_init[0:n_trials, 4:6] = 0.0
    x_init[0:n_trials, 6] = obs_map[0:1, 0]
    x_init[0:n_trials, 7] = obs_map[0:1, 1]
    x_init[0:n_trials, 8] = obs_map[0:1, 2]
    x_init[0:n_trials, 9] = -5 * args.dt


    x = x_init[0:n_trials].cpu()
    base_i = [0] * n_trials
    cd = [0] * n_trials

    x_input = x.cuda()
    x_input[:, 6] = x_input[:, 6] - x_input[:, 0]
    x_input[:, 0] = 0
    x_input[:, 6][torch.where(x_input[:, 6]>args.obs_xmax)] = -5

    u = net_stl(x_input)

    # let the ordinary control run once, and find unsafe cases
    seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
    acc_tmp = (stl(seg_out, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
    acc_safe_tmp = (stl_safe(seg_out, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()

    acc_rate = torch.mean(acc_tmp)
    safe_rate = torch.mean(acc_safe_tmp)
    oot_rate = torch.mean((torch.any(seg_out[:, :, -1]<0, dim=1)).float(), dim=0)

    print("safe:%.3f acc:%.3f oot:%.3f"%(safe_rate, acc_rate, oot_rate))

    # statistics
    seg_list=[]
    metrics_str=["acc", "reward", "score", "t", "safety", "intime"]
    metrics = {xx:[] for xx in metrics_str}

    crash = np.zeros((nt, n_trials))
    timeout = np.zeros((nt, n_trials))
    prev_x_input = None
    history=[]
    cd_list=np.zeros((nt, n_trials))

    acc_list=np.zeros((nt, n_trials))
    # start the loop
    for ti in range(nt):
        if ti % 10 == 0:
            print("ti=",ti)
        print("t:%03d x:%.2f y:%.2f ph:%.2f | %.2f %.2f %.2f | %.2f %.2f r=%.2f T=%.2f || %d"%(
            ti, x[0,0], x[0,1], x[0,2],
            x[0,3], x[0,4], x[0,5],
            x[0,6], x[0,7], x[0,8], x[0,9], cd[0]
            ))
        shall_update = [False] * n_trials
        updated_obs = [False] * n_trials
        x_input = x.cuda()
        # update the obstacles if needed
        for i in range(n_trials):
            t1_debug = time.time()
            if obs_map[base_i[i],0]-x[i,0]<-1:
                base_i[i]+=1
                x[i,6:6+3] = obs_map[base_i[i]]
                updated_obs[i]=True
            
            if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.abs(x[i,1]) > args.river_width/2:
                print(x[i],torch.norm(x[i, :2]-x[i, 6:8], dim=-1),x[i, 8])
                crash[ti, i] = 1

            if x[i,9]<0:
                timeout[ti, i] = 1

            # get updated x_input
            dx = x_input[i, 0]
            x_input[i, 6] = x_input[i, 6] - dx
            if x_input[i, 6] > args.obs_xmax:
                x_input[i, 6] = -5
            else:
                if prev_x_input is not None and (prev_x_input[i, 6] == -5 or updated_obs[i]):
                    x[i, 9] = reset_T[i, base_i[i]]
                    print(ti, i, x[i,9])
            x_input[i, 0] = x_input[i, 0] - dx

        u_nn = net_stl(x_input)
        u_real=[]
        for i in range(n_trials):
            cd[i] = 0  # for updated_obs
            u_real.append(u_nn[i])
        u_real = torch.stack(u_real, dim=0)
        seg_out = dynamics(x.cpu(), u_real.cpu(), include_first=True)
        stl_score_pre = (stl(seg_out, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        
        k_list = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

        if args.not_use_backup==False:
            # when under unsafe
            for i in range(n_trials):
                if stl_score_pre[i]<1 and cd[i]<=0:
                    max_plan_t = 10
                    u_real[i] = 0.0
                    for plan_t in [1,4,len(k_list)-1]:
                        print(plan_t)
                        nt0 = plan_t
                        k0 = 1
                        k1 = k_list[plan_t]
                        a0 = np.linspace(-args.thrust_max, -args.thrust_max, k0)
                        a1 = np.linspace(-args.delta_max, args.delta_max, k1)
                        A0, A1 = np.meshgrid(a0,a1)
                        A0, A1 = A0.flatten(), A1.flatten()
                        li0 = [A0] * nt0
                        li1 = [A1] * nt0
                        a_seq0 = np.array(np.meshgrid(*li0)).reshape(nt0, -1).T
                        a_seq1 = np.array(np.meshgrid(*li1)).reshape(nt0, -1).T
                        a_seq = np.stack([a_seq0, a_seq1], axis=-1)
                        a_seq = torch.from_numpy(a_seq).float().cuda()
                        
                        x_input_mul = torch.tile(x[i:i+1].cuda(), [a_seq.shape[0], 1])
                        seg0 = dynamics(x_input_mul, a_seq, include_first=True)
                        x_new = seg0[:, -1]
                        x_new_input = x_new.clone()
                        x_new_input[:, 6] = x_new_input[:, 6] - x_new_input[:, 0]
                        x_new_input[:, 0] = 0
                        u_output = net_stl(x_new_input).detach()
                        seg1 = dynamics(x_new, u_output[:, :args.nt-nt0], include_first=False)
                        seg = torch.cat([seg0, seg1], dim=1)

                        score_safe = torch.all(
                            torch.logical_and(
                                torch.norm(seg0[:, :, :2]-seg0[:, :, 6:8], dim=-1)>seg0[:, 0:1, 8]+0.1, 
                                torch.abs(seg0[:, :, 1]) < args.river_width/2), dim=1
                        )
                        safe_idx = torch.where(score_safe)[0]
                        
                        if safe_idx.shape[0]>0:
                            
                            score_stl_smooth = stl(seg[safe_idx], args.smoothing_factor)[:, :1]
                            max_idx = torch.argmax(score_stl_smooth, dim=0)[0]
                            score_stl = (stl(seg[safe_idx], args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                            u_real[i] = torch.cat([a_seq[safe_idx[max_idx]], u_output[safe_idx[max_idx], :args.nt-nt0]], dim=0)
                            cd[i] = plan_t
                            cd_list[ti, i] = cd[i]
                            if score_stl[max_idx] >0:
                                # find feasible
                                print("find feasible")
                                break
                        else:
                            if len(seg_list)>0:
                                print("PREV")
                                for ttti in range(args.nt+1):
                                    print("i=%02d t=%02d x:%.2f y:%.2f th:%.2f | %.2f %.2f %.2f | x=%.2f y=%.2f r=%.2f T=%.2f | %.2f %.2f"%(
                                        i, ttti, seg_list[-1][i,ttti,0], seg_list[-1][i,ttti,1], seg_list[-1][i,ttti,2], seg_list[-1][i,ttti,3],
                                        seg_list[-1][i,ttti,4], seg_list[-1][i,ttti,5], seg_list[-1][i,ttti,6], seg_list[-1][i,ttti,7],
                                        seg_list[-1][i,ttti,8], seg_list[-1][i,ttti,9], 
                                        u_prev[i,min(ttti,args.nt-1),0], u_prev[i,min(ttti,args.nt-1),1]
                                    ))
                            print("NOW")
                            # for ii in range(seg0.shape[0]):
                            for ii in [0, 2*(3**2), 2*(3**3), -1]:
                                for ttti in range(seg0.shape[1]):
                                    print("i=%02d t=%02d x:%.2f y:%.2f th:%.2f | %.2f %.2f %.2f | x=%.2f y=%.2f r=%.2f T=%.2f | %.2f %.2f"%(
                                        ii, ttti, seg0[ii, ttti, 0], seg0[ii, ttti, 1], seg0[ii, ttti, 2], 
                                        seg0[ii, ttti, 3], seg0[ii, ttti, 4], seg0[ii, ttti, 5], 
                                        seg0[ii, ttti, 6], seg0[ii, ttti, 7], seg0[ii, ttti, 8], seg0[ii, ttti,9], 
                                        a_seq[ii,min(ttti,seg0.shape[1]-2),0], a_seq[ii,min(ttti,seg0.shape[1]-2),1]
                                    ))

                            raise NotImplementError

                        if i==2:
                            if ti==32:
                                print(safe_idx[max_idx], score_safe[safe_idx[max_idx]], )
                                print("Choose", safe_idx[max_idx], max_idx)
                                for ttti in range(plan_t):
                                    ss=seg0[safe_idx[max_idx],ttti]
                                    print("i=%02d t=%02d x:%.2f y:%.2f th:%.2f | %.2f %.2f %.2f | x=%.2f y=%.2f r=%.2f T=%.2f | %.2f %.2f"%(
                                        i, ttti, ss[0], ss[1], ss[2], ss[3],
                                        ss[4], ss[5], ss[6], ss[7],
                                        ss[8], ss[9],
                                        a_seq[safe_idx[max_idx],min(ttti,args.nt-1),0], a_seq[safe_idx[max_idx],min(ttti,args.nt-1),1]
                                    ))

                                
                                print(torch.norm(seg0[safe_idx[max_idx], :, :2]-seg0[safe_idx[max_idx], :, 6:8], dim=-1))
                                print(seg0[safe_idx[max_idx], 0:1, 8])
                                print(seg0[safe_idx[max_idx], :, 1])

        
        seg_out = dynamics(x.cpu(), u_real.cpu(), include_first=True)

        # evaluation
        debug_dt = time.time() - t1_debug

        seg_total = seg_out.clone()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        acc_list[ti,:] = acc[:,0]

        safety = 1 - np.mean(np.any(crash, axis=0), axis=0)
        intime = 1 - np.mean(np.any(timeout, axis=0), axis=0) 

        metrics["t"].append(debug_dt)
        metrics["safety"].append(safety)
        metrics["intime"].append(intime)
        metrics["acc"].append(acc_avg)

        history.append(x.clone())
        seg_list.append(seg_out.detach().cpu())
        for i in range(n_trials):
            x[i] = seg_out[i, 1].detach().cpu()

        prev_x_input = x_input.clone()
        u_prev = u_real.clone()

        print("%03d Acc:%.2f Safe:%.2f InTime:%.2f"%(ti, acc_avg, safety, intime))
    
    history = torch.stack(history, dim=1)
    seg_list = torch.stack(seg_list, dim=1)

    print("ACC:%.3f SAFE:%.3f Trajlen:%.3f InTime:%.3f"%(
        np.mean(np.array(metrics["acc"])), metrics["safety"][-1], 
        np.mean(np.sum(np.cumsum(crash, axis=0)<=0, axis=0), axis=0),
        metrics["intime"][-1]))

    # visualizations
    if args.no_viz:
        return
    
    # visualization
    bloat = 1.0
    ratio = 0.6  # make space for ship
    ship_ratio = 1 - ratio
    extend_x = 10
    r = np.sqrt(2)/2
    bk = np.sqrt(1 - r**2) 
    poly_ship = np.array([
        [1, 0],
        [0, r],
        [-bk, r],
        [-bk, -r],
        [0, -r]
    ])
    poly_ship = poly_ship * ship_ratio
    fs_list=[]
    for ti in range(nt):
        if ti % args.sim_freq == 0 or ti == nt - 1:
            print("Viz", ti)  #, 1-np.mean(collide[ti]), 1-np.mean(real_collide[ti]))
            ax = plt.gca()
            for obs_i in range(n_obs):
                ax.add_patch(
                    Ellipse([obs_map[obs_i, 0], obs_map[obs_i, 1]], obs_map[obs_i, 2] * 2 * ratio, obs_map[obs_i, 2] * 2 * ratio, 
                                        label="obstacle" if obs_i==0 else None, color="gray", alpha=0.8))

            i_cnt = 0
            for i in range(n_trials):
                i_cnt+=1
                s = to_np(history[i, ti])
                poly_ship_t = np.array(poly_ship)
                poly_ship_t[:, 0] = poly_ship[:, 0] * np.cos(s[2]) - poly_ship[:, 1] * np.sin(s[2])
                poly_ship_t[:, 1] = poly_ship[:, 0] * np.sin(s[2]) + poly_ship[:, 1] * np.cos(s[2])
                poly_ship_t[:, 0] += s[0]
                poly_ship_t[:, 1] += s[1]
                ax.add_patch(Polygon(poly_ship_t, label="ego ship" if i_cnt==1 else None, color="blue", alpha=0.8))
                plt.plot(history[i, :ti+1, 0], history[i, :ti+1, 1], 
                            color="lightsalmon", label="past-traj" if i_cnt==1 else None, linewidth=2, alpha=1.0)
                plt.plot(seg_list[i, ti, :, 0], seg_list[i, ti, :, 1], label="NN-traj" if i_cnt==1 else None, color="purple" if acc_list[ti][i]<=0 else "green", linewidth=1, alpha=1)
                plt.plot(seg_list[i, ti, :int(cd_list[ti, i])+1, 0], seg_list[i, ti, :int(cd_list[ti, i])+1, 1], 
                            color="red", label="backup-traj" if i_cnt==1 else None, linewidth=1, alpha=1)

                if args.plan:
                    plt.plot(to_np(pwl_list_list[ti][i, :, 0])+s[0], to_np(pwl_list_list[ti][i, :, 1]), color="red", alpha=0.5)


            # plot visible range
            idx = 0 # camera idx
            s = history[idx, ti]
            plt.legend(fontsize=10, loc="lower right", ncol=2)
            ax.axis("scaled")
            plt.xlim(-1, 30)
            plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)
            figname="%s/t_%03d.png"%(args.viz_dir, ti)
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
            fs_list.append(figname)
            plt.close()


def test_ship(x_init, net_stl, net_cbf, rl_policy, stl, stl_safe, stl_list_debug):
    avoid, in_river, finally_track, in_T, track = stl_list_debug
    metrics_str=["acc", "reward", "score", "t", "safety"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.ship_env import ShipEnv
    args.mode="ship2"
    the_env = ShipEnv(args)

    # a long map of 2d obstacles
    if net_cbf is not None:
        test_cbf = True
        state_str = "TEST_CBF"
    else:
        test_cbf = False
        state_str = "TEST_STL"
    nt = args.test_nt
    n_obs = 100
    n_trials = args.num_trials
    update_freq = args.mpc_update_freq
    if args.diff_test:
        reset_T = 20 * args.dt
        # TODO reset_T could be (n_obs, n_trials) array (range 10-20)
        reset_T = rand_choice_tensor(list(range(15, 25)), (n_trials, n_obs)) * args.dt
    else:
        reset_T = 20 * args.dt
    debug_t1 = time.time()
    
    obs_list=[]
    obs_x = x_init[0:1, 6].item()
    for i in range(n_obs):
        obs_x = obs_x + uniform_tensor(15, 25, (1, 1))
        obs_y = obs_x * 0 + 0
        if args.diff_test:
            obs_r = obs_x * 0 + uniform_tensor(args.obs_rmin, args.obs_rmax, (1, 1))
        else:
            obs_r = obs_x * 0 + 1
        if args.obs_specific:
            if obs_r<0.8:
                reset_T[:, i] = 15 * args.dt
            elif obs_r<1.0:
                reset_T[:, i] = 20 * args.dt
            else:
                reset_T[:, i] = 25 * args.dt
        obs = torch.cat([obs_x, obs_y, obs_r], dim=-1)
        obs_list.append(obs)
    obs_map = torch.stack(obs_list, dim=1)[0]  # (M, 3)

    x_init[0:n_trials, 1] = x_init[0:n_trials, 1] * 0.5
    x_init[0:n_trials, 2] = 0
    x_init[0:n_trials, 6] = obs_map[0:1, 0]
    x_init[0:n_trials, 7] = x_init[0:1, 7]
    x_init[0:n_trials, 8] = x_init[0:1, 8]
    x_init[0:n_trials, 9] = 15 * args.dt
    x = x_init[0:n_trials].cpu()
    base_i = [0] * n_trials  # look at the first and the second one
    fs_list = []
    history = []
    seg_list = []

    # statistics
    safety = 0
    move_distance = 0
    cnt = [0] * n_trials
    seg = None
    collide = np.zeros((nt, n_trials))
    real_collide = np.zeros((nt, n_trials))
    cbf_record = [0] * n_trials
    prev_x_input = None
    x_input_list = []
    pwl_list_list = []
    for ti in range(nt):
        if ti % 10 == 0:
            print(ti)
        shall_update = [False] * n_trials
        updated_obs = [False] * n_trials
        x_input = x.cuda()

        for i in range(n_trials):
            if obs_map[base_i[i], 0] - x[i, 0] < -1:
                base_i[i] += 1
                x[i, 6:6+3] = obs_map[base_i[i]]
                x_input[i, 6:6+3] = obs_map[base_i[i]].cuda()
                updated_obs[i] = True
                if test_cbf:
                    cbf_record[i] = x[i, 6]
            if test_cbf:
                if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.abs(x[i,1]) > args.river_width/2 or (ti-1>=0 and collide[ti-1, i] == 1):
                    collide[ti, i] = 1
            else:
                if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.abs(x[i,1]) > args.river_width/2 or (ti-1>=0 and collide[ti-1, i] == 1) or x[i,9]<0:
                    collide[ti, i] = 1

                if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.abs(x[i,1]) > args.river_width/2 or (ti-1>=0 and real_collide[ti-1, i] == 1):
                    real_collide[ti, i] = 1

            if cnt[i] % update_freq == 0 or updated_obs[i]:
                shall_update[i] = True
                cnt[i] = 0
                if test_cbf:
                    dx = x_input[i, 0]
                else:
                    dx = x_input[i, 0]
                x_input[i, 6] = x_input[i, 6] - dx
                
                if x_input[i, 6] > args.obs_xmax:
                    x_input[i, 6] = -5
                else:  # real obstacle coming
                    if prev_x_input is not None and (prev_x_input[i, 6] == -5 or updated_obs[i]):
                        # reset T
                        if args.diff_test:
                            x[i, 9] = reset_T[i, base_i[i]]
                        else:
                            x[i, 9] = reset_T

                x_input[i, 0] = x_input[i, 0] - dx
        x_input_list.append(x_input)
        debug_tt1=time.time()
        dt_minus = 0
        if test_cbf:
            u = net_cbf(x_input).reshape(n_trials, 1, 2)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.plan:
            u_list = []
            pwl_list = []
            for iii in range(x_input.shape[0]):
                u_item, pwl_torch = solve_planner(ti, i, x_input[iii:iii+1])
                u_list.append(u_item)
                pwl_list.append(pwl_torch)
            u = torch.cat(u_list, dim=0)
            pwl_list = torch.cat(pwl_list, dim=0)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.mpc:
            u_list=[]
            for iii in range(x_input.shape[0]):
                u_item = solve_mpc(x_input[iii:iii+1])
                u_list.append(u_item)
            u = torch.cat(u_list, dim=0)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.cem:
            u_list=[]
            for iii in range(x_input.shape[0]):
                u_item = solve_cem(ti, x_input[iii], stl, args)
                u_list.append(u_item)
            u = torch.stack(u_list, dim=0)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.rl or args.mbpo or args.pets:
            _, u, dt_minus = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=True)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.grad:
            u = gradient_solve(ti, x_input, stl)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        else:
            u = net_stl(x_input)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
            if args.finetune:
                seg_tmp = seg_out.cuda()
                acc_tmp = (stl(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                err_idx = torch.where(acc_tmp<1)[0]

                acc_tmp1 = (avoid(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_tmp2 = (in_river(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_tmp3 = (finally_track(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                acc_tmp4 = (in_T(seg_tmp, args.smoothing_factor, d={"hard":True})[0, :]>=0).float()
                acc_tmp5 = (track(seg_tmp, args.smoothing_factor, d={"hard":True})[0, :]>=0).float()

                if ti==2:
                    print(acc_tmp, acc_tmp1, acc_tmp2, acc_tmp3)
                    for ttti in range(args.nt):
                        print(ttti, acc_tmp4[ttti], acc_tmp5[ttti])

                if err_idx.shape[0]>0:
                    ft_n = err_idx.shape[0]
                    print(ti, "[Before] Acc=%.2f, %d do not satisfy STL"%(torch.mean(acc_tmp), ft_n))
                    u_output_fix = u.clone()
                    for iii in range(ft_n):
                        sel_ = err_idx[iii]
                        u_fix = back_solve(x_input[sel_:sel_+1], u[sel_:sel_+1], net_stl, stl, stl_safe, x[sel_:sel_+1], seg_tmp[sel_:sel_+1])
                        u_output_fix[sel_:sel_+1] = u_fix
                    u = u_output_fix
                    seg_fix = dynamics(x.cpu(), u.cpu(), include_first=True)
                    acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                    err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                    ft_n_fix = err_idx_fix.shape[0]
                    print(ti, "[After]  Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
                    if ti==2:
                        for _t in range(seg_fix.shape[1]):
                            print("%02d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f || %.2f %.2f"%(
                                _t, seg_fix[0, _t, 0], seg_fix[0, _t, 1], seg_fix[0, _t, 2], seg_fix[0, _t, 3], seg_fix[0, _t, 4],
                                seg_fix[0, _t, 5], seg_fix[0, _t, 6], seg_fix[0, _t, 7], seg_fix[0, _t, 8], seg_fix[0, _t, 9],
                                u[0, min(_t, args.nt-1), 0], u[0, min(_t, args.nt-1), 1]
                                ))
                    seg_out = seg_fix
        debug_tt2=time.time()

        if args.plan:
            pwl_list_list.append(pwl_list)

        seg_total = seg_out.clone()
        # EVALUATION
        debug_dt = debug_tt2 - debug_tt1
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(the_env.generate_reward_batch(to_np(seg_total[:,0])))
        
        safety = 1-np.mean(collide[ti])
        metrics["t"].append(debug_dt - dt_minus)
        metrics["safety"].append(safety)
        metrics["acc"].append(acc_avg)
        metrics["score"].append(score_avg)
        metrics["reward"].append(reward)

        if seg is None:
            seg = seg_out
        else:
            seg = seg_list[-1].clone()
            for i in range(n_trials):
                if shall_update[i]:
                    seg[i] = seg_out[i].detach().cpu()
        seg_list.append(seg.detach().cpu())
        history.append(x.clone())
        for i in range(n_trials):
            x[i] = seg[i, cnt[i]+1].detach().cpu()
        for i in range(n_trials):
            cnt[i] += 1
        
        prev_x_input = x_input.clone()

    print(metrics["acc"])

    print("Real  safe:%.3f"%(np.mean(1-real_collide[-1])))
    print("Total safe:%.3f"%(np.mean(1-collide[-1])))
    print("Total acc: %.3f"%(np.mean(np.array(metrics["acc"]))))
    eval_proc(metrics, "e4_ship_track", args)
    if args.no_viz:
        return

    history = torch.stack(history, dim=1)
    seg_list = torch.stack(seg_list, dim=1)
    
    # visualization
    bloat = 1.0
    ratio = 0.6  # make space for ship
    ship_ratio = 1 - ratio
    extend_x = 10
    r = np.sqrt(2)/2
    bk = np.sqrt(1 - r**2) 
    poly_ship = np.array([
        [1, 0],
        [0, r],
        [-bk, r],
        [-bk, -r],
        [0, -r]
    ])
    poly_ship = poly_ship * ship_ratio
    for ti in range(nt):
        if ti % args.sim_freq == 0 or ti == nt - 1:
            print(state_str, "Viz", ti, 1-np.mean(collide[ti]), 1-np.mean(real_collide[ti]))
            ax = plt.gca()
            for obs_i in range(n_obs):
                ax.add_patch(
                    Ellipse([obs_map[obs_i, 0], obs_map[obs_i, 1]], obs_map[obs_i, 2] * 2 * ratio, obs_map[obs_i, 2] * 2 * ratio, 
                                        label="obstacle" if obs_i==0 else None, color="gray", alpha=0.8))

            i_cnt = 0
            for i in range(n_trials):
                if collide[ti, i]:
                    continue
                else:
                    i_cnt+=1
                s = to_np(history[i, ti])
                poly_ship_t = np.array(poly_ship)
                poly_ship_t[:, 0] = poly_ship[:, 0] * np.cos(s[2]) - poly_ship[:, 1] * np.sin(s[2])
                poly_ship_t[:, 1] = poly_ship[:, 0] * np.sin(s[2]) + poly_ship[:, 1] * np.cos(s[2])
                poly_ship_t[:, 0] += s[0]
                poly_ship_t[:, 1] += s[1]
                ax.add_patch(Polygon(poly_ship_t, label="ego ship" if i_cnt==1 else None, color="blue", alpha=0.8))
                plt.plot(seg_list[i, ti, :, 0], seg_list[i, ti, :, 1], label="trajectory" if i_cnt==1 else None, color="blue", linewidth=0.5, alpha=0.5)

                if args.plan:
                    plt.plot(to_np(pwl_list_list[ti][i, :, 0])+s[0], to_np(pwl_list_list[ti][i, :, 1]), color="red", alpha=0.5)


            # plot visible range
            idx = 0 # camera idx
            s = history[idx, ti]
            plt.legend(fontsize=8, loc="lower right")
            ax.axis("scaled")
            plt.xlim(s[0]-bloat, s[0]+args.canvas_w+extend_x+bloat)
            plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)

            debug_input = to_np(x_input_list[ti])
            plt.title("Simulation (%04d/%04d) Safe:%.2f%%"%(ti, nt, 1-np.mean(collide[ti])))
            figname="%s/t_%03d.png"%(args.viz_dir, ti)
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
            fs_list.append(figname)
            plt.close()
    
    print("Real  safe:%.3f"%(np.mean(1-real_collide[-1])))
    print("Total safe:%.3f"%(np.mean(1-collide[-1])))
    print("Total acc: %.3f"%(np.mean(np.array(metrics["acc"]))))

    os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
    generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)
    debug_t2 = time.time()
    print("Finished in %.2f seconds"%(debug_t2 - debug_t1))

def viz_cbf(net_cbf):
    # (x, y, phi, u, v, r, d, r, T)
    # iterate (x, y)
    nx=200
    ny=200
    x = torch.linspace(0, 10, nx)
    y = torch.linspace(-args.river_width/2, args.river_width/2, ny) * 1.2
    xy = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2)
    xs = xy[:, 0]
    ys = xy[:, 1]
    phis = torch.ones_like(xs) * 0
    us = torch.ones_like(xs) * (args.s_umin + args.s_umax) /2
    vs = torch.ones_like(xs) * 0
    rs = torch.ones_like(xs) * 0
    obs_xs = torch.ones_like(xs) * 5
    obs_ys = torch.ones_like(xs) * 0
    obs_rs = torch.ones_like(xs) * 1
    obs_ts = torch.ones_like(xs) * 15 * args.dt

    x_input = torch.stack([
        xs, ys, phis, us, vs, rs, obs_xs, obs_ys, obs_rs, obs_ts
    ], dim=-1).cuda()

    for ss_i in range(11):
        s_str = "g0207-091431_Ship3_T20_3e-4_256x3_CBF_ori_Long/models/cbf_%05d.ckpt"%(ss_i * 10000)
        net_cbf.load_state_dict(torch.load(utils.find_path(s_str)))
        v_output = net_cbf(x_input)
        print(ss_i, v_output.shape, torch.min(v_output), torch.max(v_output))
        xs_np = to_np(xs).reshape((nx, ny))
        ys_np = to_np(ys).reshape((nx, ny))
        vs_np = to_np(v_output).reshape((nx, ny))
        from matplotlib import ticker, cm
        cs = plt.contourf(xs_np, ys_np, vs_np,
                    #   locator = ticker.LogLocator(),
                    levels = [-2.0, -1.0, -0.5, -0.1, 0, 0.1, 0.5, 1.0, 2.0],
                    linewidths=2,
                    linestyles="--",
                    # cmap ="bone"
                    )
    
        cbar = plt.colorbar(cs)
        plt.axis("scaled")
        plt.savefig("%s/debug_%05d.png"%(args.viz_dir, ss_i * 1000))
        plt.close()


def test_pid(x_init, net_cbf):
    k4=10
    k5=10
    for k1 in [5]:
        for k2 in [10, 20, 30, 50, 100, 200, 500, 1000, 2000]:
            for k3 in [10, 20, 30, 50, 100, 200, 500, 1000, 2000]:
                x = x_init.clone()
                us = []
                segs = [x]
                X_DIM = x.shape[-1]
                for ti in range(args.nt * 2):
                    u = net_cbf(x, k1, k2, k3, k4, k5)
                    new_x = dynamics_s(x, u, num=args.num_sim_steps)
                    segs.append(new_x)
                    us.append(u)
                    x = new_x.detach()
                segs = torch.stack(segs, dim=1)  # (N, T, 3)
                us = torch.stack(us, dim=1)  # (N, T, 2)
                all_x_epi = segs

                init_np = to_np(x_init)
                seg_np = to_np(segs)
                print("k: %.1f %.1f %.1f rmse: %.2f %.2f"%(k1, k2, k3, torch.mean(torch.square(segs[:, :, 1])), torch.mean(torch.square(segs[:, args.nt:, 1]))))
                num_viz = 100
                for i in range(num_viz):
                    plt.plot(seg_np[i, :, 0], seg_np[i, :, 1], color="blue", alpha=0.5, linewidth=1)
                plt.axis("scaled")
                plt.xlim(0, 25)
                plt.ylim(-3, 3)
                plt.savefig("%s/pid_debug_%.1f_%.1f_%.1f.png"%(args.viz_dir, k1, k2, k3))
                plt.close()


def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)    
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    
    avoid = Always(0, args.nt, AP(lambda x: torch.norm(x[..., :2] - x[..., 6:6+2], dim=-1)**2 - x[..., 6+2]**2 - args.bloat_d**2, comment="Avoid Obs"))
    in_river = Always(0, args.nt, AP(lambda x: (args.river_width/2)**2-x[..., 1]**2, comment="In River"))
    diverge = AP(lambda x: - args.track_thres**2 + x[..., 1]**2, comment="Diverge")
    in_T = AP(lambda x: x[..., 9], comment="In Time")
    no_obs = AP(lambda x: -x[..., 6]-3, comment="No Obs")
    track = Always(0, args.nt, Not(diverge))
    track_if_no_obs = Imply(no_obs, Eventually(0, args.nt, track))
    back_to_track_if_obs = Imply(Not(no_obs), Until(0, args.nt, in_T, track))
    finally_track = Until(0, args.nt, in_T, track)
    if args.solve2:
        finally_track_2 = Imply(diverge, Or(
            Always(0, args.nt, in_T),
            Eventually(0, args.nt, Always(0,args.nt,And(in_T, Not(diverge))))
        ))
        stl = ListAnd([avoid, in_river, finally_track_2])
    else:
        stl = ListAnd([avoid, in_river, finally_track])

    
    stl_safe = ListAnd([avoid, in_river])

    stl_list_debug = [avoid, in_river, finally_track, in_T, track]

    x_init = initialize_x(args.num_samples).float().cuda()

    if args.test_pid:
        net_cbf = Net(args).cuda()
        test_pid(x_init, net_cbf)
        return

    if args.test:
        net_stl = Policy(args).cuda()
        net_cbf = Net(args).cuda()
        if args.policy_pretrained_path is not None:
            net_cbf.load_state_dict(torch.load(utils.find_path(args.policy_pretrained_path)))
            if args.viz_cbf:
                net_cbf_v = CBF(args).cuda()
                net_cbf_v.load_state_dict(torch.load(utils.find_path(args.cbf_pretrained_path)))
                viz_cbf(net_cbf_v)
                return
            test_ship(x_init, None, net_cbf, None, stl, stl_safe, stl_list_debug)
        elif args.net_pretrained_path is not None or args.rl or args.mbpo or args.pets:
            rl_policy = None
            if args.rl:
                from stable_baselines3 import SAC, PPO, A2C
                rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
            elif args.mbpo or args.pets:
                rl_policy = get_mbrl_models(get_exp_dir()+"/"+args.rl_path, args, args.mbpo)
            if args.net_pretrained_path is not None:
                net_stl.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
            if args.backup:
                ship_backup_demo(x_init, net_stl, None, rl_policy, stl, stl_safe, stl_list_debug, args)
            else:
                test_ship(x_init, net_stl, None, rl_policy, stl, stl_safe, stl_list_debug)
        return

    if args.train_cbf:
        train_traj_cbf(x_init, eta)
        exit()

    print(stl)
    stl.update_format("word")
    print(stl)
    relu = nn.ReLU()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epi in range(args.epochs):
        eta.update()
        x0 = x_init.detach()
        u = net(x0)
        seg = dynamics(x0, u, include_first=True)
        score = stl(seg, args.smoothing_factor)[:, :1]
        acc = (stl(seg, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc)

        safe_metric, _ = check_safety_stl(seg) 
        safe_avg = torch.mean(safe_metric)

        dist_loss = torch.mean(seg[..., 1]**2)
        dist_loss = dist_loss * args.dist_w

        loss = torch.mean(relu(0.5-score)) + dist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if epi % args.print_freq == 0:
            print("%s|%03d  loss:%.3f dist:%.3f acc:%.4f safe:%.3f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), dist_loss.item(), acc_avg.item(), safe_avg.item(), 
                eta.interval_str(), eta.elapsed_str(), eta.eta_str()))
        
        # save models
        if epi % args.save_freq == 0:
            torch.save(net.state_dict(), "%s/model_%05d.ckpt"%(args.model_dir, epi))
        
        # viz simulations
        if epi % args.viz_freq == 0 or epi == args.epochs - 1:
            init_np = to_np(x_init)
            seg_np = to_np(seg)
            acc_np = to_np(acc)
            sim_visualization(epi, init_np, seg_np, acc_np)


def back_solve(x_input, output_0, policy, stl, stl_safe, x, seg_old):

    nt0 = 7
    k = 2

    a0 = np.linspace(-args.thrust_max, args.thrust_max, k)
    a1 = np.linspace(-args.delta_max, args.delta_max, k)
    A0, A1 = np.meshgrid(a0,a1)
    A0, A1 = A0.flatten(), A1.flatten()
    li0 = [A0] * nt0
    li1 = [A1] * nt0
    a_seq0 = np.array(np.meshgrid(*li0)).T.reshape(-1, nt0)
    a_seq1 = np.array(np.meshgrid(*li1)).T.reshape(-1, nt0)
    a_seq = np.stack([a_seq0, a_seq1], axis=-1)
    a_seq = torch.from_numpy(a_seq).float().cuda()

    x_input_mul = torch.tile(x.cuda(), [a_seq.shape[0], 1])
    seg0 = dynamics(x_input_mul, a_seq, include_first=True)
    x_new = seg0[:, -1]

    x_new_input = x_new.clone()
    x_new_input[:, 6] = x_new_input[:, 6] - x_new_input[:, 0]
    x_new_input[:, 0] = 0

    u_output = policy(x_new_input).detach()

    seg1 = dynamics(x_new, u_output[:, :args.nt-nt0], include_first=False)
    

    seg = torch.cat([seg0, seg1], dim=1)
    score = stl(seg, args.smoothing_factor)[:, :1]

    score_old = stl(seg_old, args.smoothing_factor)[:, :1]

    score_safe = stl_safe(seg0, args.smoothing_factor)[:, :1]

    idx = torch.argmax(score, dim=0).item()
    score_max = score[idx]
    if score_max>score_old:
        u_merged = torch.cat([a_seq[idx, :],u_output[idx, :args.nt-nt0]], dim=0).unsqueeze(0)
        return u_merged
    else:
        return output_0


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    add = parser.add_argument
    add("--exp_name", '-e', type=str, default=None)
    add("--gpus", type=str, default="0")
    add("--seed", type=int, default=1007)
    add("--num_samples", type=int, default=50000)
    add("--epochs", type=int, default=50000)
    add("--lr", type=float, default=3e-5)
    add("--nt", type=int, default=20)
    add("--dt", type=float, default=0.15)
    add("--print_freq", type=int, default=100)
    add("--viz_freq", type=int, default=1000)
    add("--save_freq", type=int, default=1000)
    add("--sim_freq", type=int, default=1)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)

    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--stl_sim_steps", type=int, default=2)
    add("--n_obs", type=int, default=3)
    add("--obs_rmin", type=float, default=0.6)
    add("--obs_rmax", type=float, default=1.2)
    add("--river_width", type=float, default=4.0)
    add("--range_x", type=float, default=15.0)
    add("--thrust_max", type=float, default=0.5)
    add("--delta_max", type=float, default=3.0)
    add("--s_phimax", type=float, default=0.5)
    add("--s_umin", type=float, default=3.0)
    add("--s_umax", type=float, default=5.0)
    add("--s_vmax", type=float, default=0.3)
    add("--s_rmax", type=float, default=0.5)

    add("--canvas_h", type=float, default=10.0)
    add("--canvas_w", type=float, default=10.0)

    # CBF's configurations
    add("--train_cbf", action='store_true', default=False)
    add("--net_hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--cbf_hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--num_sim_steps", type=int, default=1)
    add("--cbf_pos_bloat", type=float, default=0.1)
    add("--cbf_neg_bloat", type=float, default=0.1)
    add("--cbf_gamma", type=float, default=0.1)
    add("--cbf_alpha", type=float, default=0.2)
    add("--cbf_cls_w", type=float, default=1)
    add("--cbf_dec_w", type=float, default=1)
    add("--cbf_prior_w", type=float, default=0.0)
    add("--cbf_nn_w", type=float, default=1.0)

    add("--dense_state_cls", action='store_true', default=False)
    add("--dense_state_dec", action='store_true', default=False)
    add("--num_dense_sample", type=int, default=10000)

    add("--alternative", action='store_true', default=False)
    add("--alternative2", action='store_true', default=False)
    add("--alternative_freq", type=int, default=50)

    add("--policy_pretrained_path", type=str, default=None)
    add("--qp", action='store_true', default=False)

    add("--both_state_cls", action='store_true', default=False)
    add("--both_state_dec", action='store_true', default=False)
    add("--dense_ratio", type=float, default=0.5)

    add("--mpc_update_freq", type=int, default=1)

    add("--u_loss", type=float, default=0.0)

    add("--river_w", type=float, default=10.0)
    add("--num_trials", type=int, default=1000)

    add("--track_thres", type=float, default=0.3)
    add("--tmax", type=int, default=25)
    add("--obs_ymin", type=float, default=-0.0)
    add("--obs_ymax", type=float, default=0.0)
    add("--obs_xmin", type=float, default=-1.0)
    add("--obs_xmax", type=float, default=8.0)

    add("--viz_cbf", action='store_true', default=False)
    add("--cbf_pretrained_path", type=str, default=None)
    add("--bloat_d", type=float, default=0.0)
    add("--origin_sampling", action='store_true', default=False)
    add("--origin_sampling2", action='store_true', default=False)
    add("--origin_sampling3", action='store_true', default=False)
    add("--dist_w", type=float, default=0.0)

    add("--test_pid", action='store_true', default=False)
    add("--diff_test", action='store_true', default=False)
    add("--obs_specific", action='store_true', default=False)

    add("--mpc", action='store_true', default=False)
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

    add("--test_nt", type=int, default=200)

    add("--finetune", action="store_true", default=False)
    add("--solve2", action="store_true", default=False)

    add("--backup", action='store_true', default=False)
    add("--not_use_backup", action='store_true', default=False)

    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)
    add("--cem", action='store_true', default=False)
    args = parser.parse_args()

    args.origin_sampling3 = True
    args.obs_specific = True
    args.diff_test = True

    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))