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

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        # input  (state, obs)
        # input (x, y, phi, u, v, r, dx1, dy1, r1, dx2, dy2, r2, dx3, dy3, r3)
        # output (acc)
        input_dim = 6 + 3 * args.n_obs
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
        # print(new_du)
        zeros = 0 * new_dx
        dsdt = torch.stack([new_dx, new_dy, new_dphi, new_du, new_dv, new_dr] + [zeros] * (3 *args.n_obs), dim=-1)
        new_x = x + dsdt * dt
        new_xx = new_x.clone()
        new_xx[:, 2] = torch.clamp(new_x[:, 2], -args.s_phimax, args.s_phimax)
        new_xx[:, 3] = torch.clamp(new_x[:, 3], args.s_umin, args.s_umax)
        new_xx[:, 4] = torch.clamp(new_x[:, 4], -args.s_vmax, args.s_vmax)
        new_xx[:, 5] = torch.clamp(new_x[:, 5], -args.s_rmax, args.s_rmax)
        
        # new_xx[:, 0:2] = new_x[:, 0:2]
        # new_xx[:, 6:] = new_x[:, 6:]
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
        if ti>0:
            dt_minus += tt2 - tt1
    xs = torch.stack(xs, dim=1)
    us = torch.stack(us, dim=1)  # (N, 2) -> (N, T, 2)
     
    return xs, us, dt_minus

def initialize_x_cycle(n, is_cbf=False):
    if is_cbf:
        x = uniform_tensor(0, 0, (n, 1))
        y = uniform_tensor(-args.river_width/2 * 1.1, args.river_width/2 * 1.1, (n, 1))
    else:
        x = uniform_tensor(0, 0, (n, 1))
        y = uniform_tensor(-args.river_width/2, args.river_width/2, (n, 1))
    phi = uniform_tensor(-args.s_phimax, args.s_phimax, (n, 1))
    u = uniform_tensor(args.s_umin, args.s_umax, (n, 1))
    v = uniform_tensor(-args.s_vmax, args.s_vmax, (n, 1))
    r = uniform_tensor(-args.s_rmax, args.s_rmax, (n, 1))
    
    obs = []
    for i in range(args.n_obs):
        gap = args.range_x / args.n_obs
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


def initialize_x(n):
    x_list = []
    total_n = 0
    while(total_n<n):
        x_init = initialize_x_cycle(n)

        safe_bloat = 1.5
        dd = 5
        n_res = 100
        crit_list = []
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
        x_val = x_init[valids_indices]
        total_n += x_val.shape[0]
        x_list.append(x_val)
    x_list = torch.cat(x_list, dim=0)[:n]
    return x_list

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        input_dim = 6 + 3 * args.n_obs
        output_dim = 2
        self.net = build_relu_nn(input_dim, output_dim, args.net_hiddens, activation_fn=nn.ReLU)

    def forward(self, x):
        num_samples = x.shape[0]
        if args.relative:
            x_enc = x.clone()
            gap = args.range_x / args.n_obs
            is_changed = (x[:, 6] < x[:, 0]).float()
            x_enc[:, 0] = 0
            x_enc[:, 1:6] = x[:, 1:6]
            x_enc[:, 6] = (x[:, 6] - x[:, 0]) * (1-is_changed) + (x[:, 9]-x[:,0]) * is_changed
            x_enc[:, 7] = x[:, 7] * (1-is_changed) + x[:, 10] * is_changed
            x_enc[:, 8] = x[:, 8] * (1-is_changed) + x[:, 8] * is_changed
            x_enc[:, 9] = (x[:, 9] - x[:, 0]) * (1-is_changed) + (x[:, 9]-x[:,0] + gap) * is_changed
            x_enc[:, 10] = x[:, 10]
            x_enc[:, 11] = x[:, 11]

            u = self.net(x_enc).reshape(num_samples, -1)
        else:
            u = self.net(x).reshape(num_samples, -1)
        u0 = torch.tanh(u[..., 0]) * args.thrust_max
        u1 = torch.tanh(u[..., 1]) * args.delta_max
        uu = torch.stack([u0, u1], dim=-1)
        return uu

class CBF(nn.Module):
    def __init__(self, args):
        super(CBF, self).__init__()
        self.args = args
        input_dim = 6 + 3 * args.n_obs
        output_dim = 1
        self.net = build_relu_nn(input_dim, output_dim, args.cbf_hiddens, activation_fn=nn.ReLU)

    def forward(self, x):
        num_samples = x.shape[0]
        if args.relative:
            x_enc = x.clone()
            gap = args.range_x / args.n_obs
            is_changed = (x[:, 6] < x[:, 0]).float()
            x_enc[:, 0] = 0
            x_enc[:, 1:6] = x[:, 1:6]
            x_enc[:, 6] = (x[:, 6] - x[:, 0]) * (1-is_changed) + (x[:, 9]-x[:,0]) * is_changed
            x_enc[:, 7] = x[:, 7] * (1-is_changed) + x[:, 10] * is_changed
            x_enc[:, 8] = x[:, 8] * (1-is_changed) + x[:, 8] * is_changed
            x_enc[:, 9] = (x[:, 9] - x[:, 0]) * (1-is_changed) + (x[:, 9]-x[:,0] + gap) * is_changed
            x_enc[:, 10] = x[:, 10]
            x_enc[:, 11] = x[:, 11]

            v = torch.tanh(self.net(x_enc))
        else:
            v = torch.tanh(self.net(x))
        tau = args.smoothing_factor
        v_prior1 = torch.clip(torch.norm(x[..., :2] - x[..., 6:8], dim=-1)**2 - x[..., 8]**2, -10, 10) 
        v_prior2 = torch.clip(torch.norm(x[..., :2] - x[..., 6+3:8+3], dim=-1)**2 - x[..., 8+3]**2, -10, 10)
        v_prior3 = args.river_w*((args.river_width/2)**2 - (x[..., 1])**2)  
        v_prior = torch.minimum(torch.minimum(v_prior1, v_prior2), v_prior3).reshape(x.shape[0], 1)
        return v_prior * args.cbf_prior_w + v * args.cbf_nn_w

def mask_mean(x, mask):
    return torch.mean(x * mask) / torch.clip(torch.mean(mask), 1e-4)


def get_masks(x):
    dist1 = torch.norm(x[..., :2] - x[..., 6:8], dim=-1) - x[..., 8]
    dist2 = torch.norm(x[..., :2] - x[..., 6+3:8+3], dim=-1) - x[..., 8+3]
    dist3 = args.river_width/2 - torch.abs(x[..., 1]) 
    safe_mask = torch.logical_and(
        torch.logical_and(dist1>=args.cbf_pos_bloat, dist2>=args.cbf_pos_bloat),
        dist3>=args.cbf_pos_bloat
    ).float()
    dang_mask = torch.logical_or(
        torch.logical_or(dist1<0, dist2<0),
        dist3<0
    ).float()
    mid_mask = (1 - safe_mask) * (1 - dang_mask)
    return safe_mask, dang_mask, mid_mask

def check_safety(x):
    dist1 = torch.norm(x[..., :2] - x[..., 6:8], dim=-1) - x[..., 8]
    dist2 = torch.norm(x[..., :2] - x[..., 6+3:8+3], dim=-1) - x[..., 8+3]
    dist3 = args.river_width/2 - torch.abs(x[..., 1]) 
    acc = torch.all(torch.logical_and(torch.logical_and(dist1>=0, dist2>=0), dist3>=0), dim=-1).float()
    inl = torch.all(dist3>=0, dim=-1).float()
    return acc, inl

def train_debug(x_init, eta):
    net = Net(args).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    for epi in range(args.epochs):
        eta.update()
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

        loss = torch.mean(segs[..., 1] ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        inl_avg = torch.mean((torch.abs(segs[..., 1])<args.river_width/2).float())
        if epi % 20 == 0:
            print("test")
            print(to_np(segs[0, :, 0]))
            print(to_np(segs[0, :, 1]))

        if epi % args.print_freq == 0:
            print("%s|%03d loss:%.3f in:%.3f | dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(),  inl_avg.item(),
                eta.interval_str(), eta.elapsed_str(), eta.eta_str()))


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
            for obs_i in range(args.n_obs):
                offset = 6 + 3 * obs_i
                ax.add_patch(Ellipse([seg_np[idx, 0, offset], seg_np[idx, 0, offset + 1]], seg_np[idx, 0, offset + 2] * 2, seg_np[idx, 0, offset + 2] * 2, 
                                        label="obstacle" if obs_i==0 else None, color="gray", alpha=0.8))
            ax.add_patch(Ellipse([seg_np[idx, 0, 0], seg_np[idx, 0, 1]], 0.5, 0.5, 
                                        label="ego", color="blue", alpha=0.8))
            plt.plot(seg_np[idx, :, 0], seg_np[idx, :, 1], label="trajectory", color="blue", linewidth=2, alpha=0.5)
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
    
    obs0 = x_init[0, 6:9]
    obs1 = x_init[0, 9:12]
    bloat = 0.1

    # args.thrust_max, args.delta_max
    # s_phimax
    # s_umin, s_umax
    # s_vmax
    # s_rmax
    for i in range(6):
        x[0, i] = x_init[0, i]

    for ti in range(args.nt):
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

        opti.subject_to((x[ti+1, 0]-obs0[0])**2+(x[ti+1, 1]-obs0[1])**2 >= bloat + obs0[2]**2 - gamma[ti, 2])
        opti.subject_to((x[ti+1, 0]-obs1[0])**2+(x[ti+1, 1]-obs1[1])**2 >= bloat + obs1[2]**2 - gamma[ti, 3])

    loss = casadi.sumsqr(gamma) * 1000000 + casadi.sumsqr(u)
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
        loss = torch.mean(relu(0.5-score))
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


def solve_planner(tti, x_init):
    from lib_pwlplan import plan, Node
    from gurobipy import GRB
    from utils import default_line, pid_control
    tt1=time.time()

    def func1(m, PWLs, di):
        if args.motion:
            _phi = m.addVars(args.nt+1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the__phi")
            _u = m.addVars(args.nt+1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the__u")
            _v = m.addVars(args.nt+1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the__v")
            _r = m.addVars(args.nt+1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="the__r")

        else:
            heading = x_np[0, 2]
            cos_th = np.cos(x_np[0, 2])
            sin_th = np.sin(x_np[0, 2])
            for i in range(args.nt):
                
                m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i][0][1] <= 0.2)
                m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i][0][1] >= -0.2)
                if i>=1:
                    m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i-1][0][1] <= 0.2)
                    m.addConstr(PWLs[0][i+1][0][1] - PWLs[0][i-1][0][1] >= -0.2)
 
                m.addConstr(PWLs[0][i+1][0][0] - PWLs[0][i][0][0] >= 0.3)
                m.addConstr(PWLs[0][i+1][1] - PWLs[0][i][1] == args.dt)
            return 1000 * sum((PWLs[0][i+1][0][1]-PWLs[0][i][0][1])**2 for i in range(nt))

    nt = args.nt
    dt = args.dt
    tmax = (nt+1)*dt
    vmax = 10
    bloat_r = 0.1

    x_np = to_np(x_init)
    obs_A0, obs_b0 = xyr_2_Ab(x_np[0, 6+0], x_np[0, 6+1], x_np[0, 6+2]+bloat_r, num_edges=8)
    obs_A1, obs_b1 = xyr_2_Ab(x_np[0, 9+0], x_np[0, 9+1], x_np[0, 9+2]+bloat_r, num_edges=8)

    A_map, b_map = xxyy_2_Ab([-0.5, 10.5, -args.river_width/2, args.river_width/2])

    in_map = Node("mu", info={"A":A_map, "b":b_map})
    avoid0 = Node("negmu", info={"A":obs_A0, "b":obs_b0})
    avoid1 = Node("negmu", info={"A":obs_A1, "b":obs_b1})
    always_in_map = Node("A", deps=[in_map], info={"int":[0, tmax]})
    always_avoid0 = Node("A", deps=[avoid0], info={"int":[0, tmax]})
    always_avoid1 = Node("A", deps=[avoid1], info={"int":[0, tmax]})

    tt2=time.time()

    specs = [Node("and", deps=[
        always_in_map, 
        always_avoid0, always_avoid1])]
    x0s = [np.array([x_init[0, 0].item(), x_init[0, 1].item()])]
    PWL, u_out = plan(x0s, specs, bloat=0.01, MIPGap=0.05, num_segs=args.nt, tmax=tmax, vmax=vmax, extra_fn_list=[func1], return_u=True, quiet=True)
    
    tt3=time.time()
    
    if PWL[0] is None:
        print("Failed")
        pwl_torch = default_line(x_init, args.nt)
    else:
        pwl_torch = torch.tensor([PWL[0][i][0] for i in range(nt+1)]).unsqueeze(0).cuda()
    
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


def test_ship(x_init, net_stl, net_cbf, rl_policy, stl):
    metrics_str=["acc", "reward", "score", "t", "safety"]
    metrics = {xx:[] for xx in metrics_str}
    from envs.ship_env import ShipEnv
    args.mode="ship1"
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
    debug_t1 = time.time()

    gap = args.range_x / args.n_obs
    
    obs_list=[]
    x_init[0:1, 6+0] = gap
    x_init[0:1, 6+3] = gap * 2
    obs_x = x_init[0:1, 6+3:6+4].cpu()
    obs_list.append(x_init[0:1, 6+0:6+3].cpu())
    obs_list.append(x_init[0:1, 6+3:6+6].cpu())
    for i in range(2, n_obs):
        obs_x = obs_x + gap
        obs_y = uniform_tensor(-args.river_width/2, args.river_width/2, (1, 1))
        obs_r = uniform_tensor(1, args.obs_rmax, (1, 1))
        obs = torch.cat([obs_x, obs_y, obs_r], dim=-1)
        obs_list.append(obs)
    obs_map = torch.stack(obs_list, dim=1)[0]  # (M, 3)

    x_init[0:n_trials, 6] = x_init[0:1, 6]
    x_init[0:n_trials, 7] = x_init[0:1, 7]
    x_init[0:n_trials, 8] = x_init[0:1, 8]
    x_init[0:n_trials, 9] = x_init[0:1, 9]
    x_init[0:n_trials, 10] = x_init[0:1, 10]
    x_init[0:n_trials, 11] = x_init[0:1, 11]


    x = x_init[0:n_trials].cpu()
    base_i = [0] * n_trials  # look at the first and the second one
    fs_list = []
    history = []
    seg_list = []
    pwl_list_list = []

    # statistics
    safety = 0
    move_distance = 0
    cnt = [0] * n_trials
    seg = None
    collide = np.zeros((nt, n_trials))
    cbf_record = [0] * n_trials
    for ti in range(nt):
        if ti % 10 == 0:
            print("ti=",ti)
        shall_update = [False] * n_trials
        updated_obs = [False] * n_trials
        x_input = x.cuda()

        for i in range(n_trials):
            if obs_map[base_i[i], 0] < x[i, 0]:
                base_i[i] += 1
                x[i, 6:6+3] = x[i, 6+3:6+6]
                x[i, 6+3:6+6] = obs_map[base_i[i]+1]
                updated_obs[i] = True
                if test_cbf:
                    cbf_record[i] = x[i, 6]
            
            if torch.norm(x[i, :2]-x[i, 6:8], dim=-1)<x[i, 8] or torch.norm(x[i, :2]-x[i, 6+3:8+3], dim=-1)<x[i, 8+3] or torch.abs(x[i,1]) > args.river_width/2 or (ti-1>=0 and collide[ti-1, i] == 1):
                collide[ti, i] = 1

            if cnt[i] % update_freq == 0 or updated_obs[i]:
                shall_update[i] = True
                cnt[i] = 0
                if test_cbf:
                    dx = cbf_record[i]
                else:
                    dx = x_input[i, 0]
                x_input[i, 9] = x_input[i, 9] - dx
                x_input[i, 6] = x_input[i, 6] - dx
                x_input[i, 0] = x_input[i, 0] - dx
        debug_tt1=time.time()
        dt_minus = 0
        if test_cbf:
            u = net_cbf(x_input).reshape(n_trials, 1, 2)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.rl or args.mbpo or args.pets:
            _, u, dt_minus = get_rl_xs_us(x_input, rl_policy, args.nt, include_first=True)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.mpc:
            u_list=[]
            for iii in range(x_input.shape[0]):
                u_item = solve_mpc(x_input[iii:iii+1])
                u_list.append(u_item)
            u = torch.cat(u_list, dim=0)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.plan:
            u_list=[]
            pwl_list=[]
            for iii in range(x_input.shape[0]):
                u_item, pwl_torch = solve_planner(ti, x_input[iii:iii+1])
                u_list.append(u_item)
                pwl_list.append(pwl_torch)
            u = torch.cat(u_list, dim=0)
            pwl_list = torch.cat(pwl_list, dim=0)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.grad:
            u = gradient_solve(ti, x_input, stl)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        elif args.cem:
            us=[]
            for i in range(x_input.shape[0]):
                u = solve_cem(ti, x_input[i], stl, args)
                us.append(u)
            u = torch.stack(us, dim=0)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
        else:
            u = net_stl(x_input)
            seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)
            if args.finetune:
                seg_tmp = seg_out.cuda()
                acc_tmp = (stl(seg_tmp, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                err_idx = torch.where(acc_tmp<1)[0]
                if err_idx.shape[0]>0:
                    ft_n = err_idx.shape[0]
                    print(ti, "[Before] Acc=%.2f, %d do not satisfy STL"%(torch.mean(acc_tmp), ft_n))
                    u_output_fix = u.clone()
                    for iii in range(ft_n):
                        sel_ = err_idx[iii]
                        u_fix = back_solve(x_input[sel_:sel_+1], u[sel_:sel_+1], net_stl, stl)
                        u_output_fix[sel_:sel_+1] = u_fix
                    u = u_output_fix
                    seg_fix = dynamics(x_input, u, include_first=True)
                    acc_tmp_fix = (stl(seg_fix, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
                    err_idx_fix = torch.where(acc_tmp_fix<1)[0]
                    ft_n_fix = err_idx_fix.shape[0]
                    print(ti, "[After]  Acc=%.2f, %d do not satisfy STL %s"%(torch.mean(acc_tmp_fix), ft_n_fix, err_idx_fix))
                    seg_out = dynamics(x.cpu(), u.cpu(), include_first=True)

        debug_tt2=time.time()
        seg_total = seg_out.clone()

        if args.plan:
            pwl_list_list.append(pwl_list)

        # EVALUATION
        debug_dt = debug_tt2 - debug_tt1 - dt_minus
        score = stl(seg_total, args.smoothing_factor)[:, :1]
        score_avg= torch.mean(score).item()
        acc = (stl(seg_total, args.smoothing_factor, d={"hard":True})[:, :1]>=0).float()
        acc_avg = torch.mean(acc).item()
        reward = np.mean(the_env.generate_reward_batch(to_np(seg_total[:,0])))
        
        safety = 1-np.mean(collide[ti])
        metrics["t"].append(debug_dt)
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

    print("Safety: %.3f"%(safety))
    print("Acc:    %.3f"%(np.mean(np.array(metrics["acc"]))))
    eval_proc(metrics, "e3_ship_safe", args)
    if args.no_viz==False:

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
                print(state_str, "Viz", ti, 1-np.mean(collide[ti]))
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
                    # ax.add_patch(Ellipse([s[0], s[1]], 0.5, 0.5, label="ego ship", color="blue", alpha=0.8))
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
                view_left = min(s[0], s[6]-s[6+2])
                view_right = s[6+3] + s[6+5]
                RY = args.river_width/2
                ax.add_patch(Rectangle([view_left, -RY], view_right-view_left, 2*RY, label="view of field", color="yellow", alpha=0.3))
                plt.legend(fontsize=8, loc="lower right")
                ax.axis("scaled")
                plt.xlim(s[0]-bloat, s[0]+args.canvas_w+extend_x+bloat)
                plt.ylim(-args.canvas_h/2-bloat, args.canvas_h/2+bloat)

                plt.title("Simulation (%04d/%04d) Safe:%.2f%%"%(ti, nt, 1-np.mean(collide[ti])))
                figname="%s/t_%03d.png"%(args.viz_dir, ti)
                plt.savefig(figname, bbox_inches='tight', pad_inches=0.1)
                fs_list.append(figname)
                plt.close()

        os.makedirs("%s/animation"%(args.viz_dir), exist_ok=True)
        generate_gif('%s/animation/demo.gif'%(args.viz_dir), 0.1, fs_list)
    debug_t2 = time.time()
    print("Finished in %.2f seconds"%(debug_t2 - debug_t1))


def back_solve(x_input, output_0, policy, stl):
    nt0 = 4
    k = 3

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


def main():
    utils.setup_exp_and_logger(args, test=args.test)
    eta = utils.EtaEstimator(0, args.epochs, args.print_freq)    
    net = Policy(args).cuda()
    if args.net_pretrained_path is not None:
        net.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
    avoid_func = lambda obs_i: Always(0, args.nt, AP(
        lambda x: torch.norm(x[..., :2] - x[..., 6+3*obs_i:6+3*obs_i+2], dim=-1)**2 - x[..., 6+3*obs_i+2]**2, comment="AVOID OBS%d"%(obs_i)))

    avoid_list = [avoid_func(obs_i) for obs_i in range(args.n_obs)] + [Always(0, args.nt, AP(lambda x: args.river_width/2 - torch.norm(x[..., 1:2], dim=-1), comment="IN MAP"))]

    stl = ListAnd(avoid_list)

    x_init = initialize_x(args.num_samples).float().cuda()

    if args.test:
        net_stl = Policy(args).cuda()
        net_cbf = Net(args).cuda()
        if args.policy_pretrained_path is not None:
            net_cbf.load_state_dict(torch.load(utils.find_path(args.policy_pretrained_path)))
            test_ship(x_init, None, net_cbf, None, stl)
        elif args.net_pretrained_path is not None or args.rl or args.mbpo or args.pets:
            rl_policy = None
            if args.rl:
                from stable_baselines3 import SAC, PPO, A2C
                rl_policy = SAC.load(get_exp_dir()+"/"+args.rl_path, print_system_info=False)
            elif args.mbpo or args.pets:
                rl_policy = get_mbrl_models(get_exp_dir()+"/"+args.rl_path, args, args.mbpo)
            if args.net_pretrained_path is not None:
                net_stl.load_state_dict(torch.load(utils.find_path(args.net_pretrained_path)))
            test_ship(x_init, net_stl, None, rl_policy, stl)
        return

    if args.train_debug:
        train_debug(x_init, eta)

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

        loss = torch.mean(relu(0.5-score))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if epi % args.print_freq == 0:
            print("%s|%03d  loss:%.3f acc:%.4f dT:%s T:%s ETA:%s" % (
                args.exp_dir_full.split("/")[-1], epi, loss.item(), acc_avg.item(), 
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
    add("--save_freq", type=int, default=5000)
    add("--sim_freq", type=int, default=1)
    add("--smoothing_factor", type=float, default=500.0)
    add("--test", action='store_true', default=False)
    add("--net_pretrained_path", '-P', type=str, default=None)

    add("--hiddens", type=int, nargs="+", default=[256, 256, 256])
    add("--stl_sim_steps", type=int, default=2)
    add("--n_obs", type=int, default=2)
    add("--obs_rmax", type=float, default=1.0)
    add("--river_width", type=float, default=4.0)
    add("--range_x", type=float, default=15.0)
    add("--thrust_max", type=float, default=0.5)
    add("--delta_max", type=float, default=3.0)
    add("--s_phimax", type=float, default=0.5)
    add("--s_umin", type=float, default=3.0)
    add("--s_umax", type=float, default=5.0)
    add("--s_vmax", type=float, default=0.3)
    add("--s_rmax", type=float, default=0.5)

    add("--canvas_h", type=float, default=4.0)
    add("--canvas_w", type=float, default=15.0)

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

    add("--relative", action='store_true', default=False)
    add("--train_debug", action='store_true', default=False)

    add("--river_w", type=float, default=10.0)
    add("--num_trials", type=int, default=1000)

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
    add("--motion", action="store_true", default=False)
    add("--finetune", action="store_true", default=False)

    add("--test_nt", type=int, default=200)

    add("--pets", action="store_true", default=False)
    add("--mbpo", action="store_true", default=False)
    add("--cem", action='store_true', default=False)
    args = parser.parse_args()
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.4f seconds"%(t2 - t1))