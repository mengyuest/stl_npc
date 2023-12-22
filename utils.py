import os
from os.path import join as ospj
import sys
import time
import shutil
from datetime import datetime, timedelta
import random
import numpy as np
import torch
import torch.nn as nn
import imageio

from lib_stl_core import softmax_pairs, softmin_pairs, softmax, softmin

def generate_gif(gif_path, duration, fs_list):
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in fs_list:
            image = imageio.imread(filename)
            writer.append_data(image)

# TODO get the experiment directory
def get_exp_dir(just_local=False):
    os.makedirs("exps_stl", exist_ok=True)
    return "./exps_stl"

def find_path(path):
    return "exps_stl/%s"%(path)

class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq, num_workers=1):
        self.start_iter = start_iter
        num_workers = 1 if num_workers is None else num_workers
        self.end_iter = end_iter//num_workers
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0
        self.num_workers = num_workers

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        if self.curr_iter % (max(1,self.check_freq//self.num_workers)) == 0:
            self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
            self.eta_t = self.interval * (self.end_iter - self.curr_iter)
    
    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)


# TODO create the exp directory
def setup_exp_and_logger(args, set_gpus=True, just_local=False, test=False):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.stdout = logger = Logger()
    EXP_ROOT_DIR = get_exp_dir(True)
    if test:
        if (hasattr(args, "rl") and args.rl) or (hasattr(args, "mbpo") and args.mbpo) or (hasattr(args, "pets") and args.pets):
            tuples = args.rl_path.split("/")
        else:
            tuples = args.net_pretrained_path.split("/")
        if ".ckpt" in tuples[-1] or ".zip" in tuples[-1]:
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-3])
        elif ".pth" in tuples[-1]:  # mbrl case
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-2])
        else:
            EXP_ROOT_DIR = ospj(EXP_ROOT_DIR, tuples[-1])
        args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "test_%s" % (logger._timestr))
    else:
        if args.exp_name.startswith("exp") and "debug" not in str.lower(args.exp_name) and "dbg" not in str.lower(args.exp_name):
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, args.exp_name)
        else:
            args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.viz_dir = os.path.join(args.exp_dir_full, "viz")
    args.src_dir = os.path.join(args.exp_dir_full, "src")
    args.model_dir = os.path.join(args.exp_dir_full, "models")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.src_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.src_dir, fname))

    logger.create_log(args.exp_dir_full)
    with open(ospj(args.exp_dir_full, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(sys.argv))
    np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)

    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    return args

class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")
        self.log = None

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        if self.log is not None:
            self.log.write(message)

    def flush(self):
        pass

def eval_proc(metrics, e_name, args, metrics_avg=None):
    if metrics_avg is None:
        metrics_avg = {xx:np.mean(np.array(metrics[xx])) for xx in metrics}
    if args.rl:
        method="RL-Raw"
        row_i=1
        if args.rl_stl:
            method="RL-STL"
            row_i=2
        if args.rl_acc:
            method="RL-acc"
            row_i=3
    elif args.mbpo:
        method="mbpo"
        row_i=9
    elif args.pets:
        method="pets"
        row_i=10
    elif args.mpc:
        method="MPC"
        row_i=4
    elif args.plan:
        method="STL-planner"
        row_i=5
    elif args.grad:
        method="SGD"
        row_i=6
    elif args.cem:
        method="CEM"
        row_i=11
    else:
        if args.finetune==False:
            method="Ours"
            row_i=7
        else:
            method="Ours-ft"
            row_i=8
    
    if "ship" in e_name:
        metrics_avg["safety"] = metrics["safety"][-1]

    new_metrics_avg = {k:[metrics_avg[k]] for k in metrics_avg}
    metrics_avg["method"] = method
    np.savez("%s/result.npz"%(args.exp_dir_full), data=metrics, data_avg=metrics_avg, method=method)

    # uniform data source
    eval_dir = os.path.join(args.exp_dir_full, "..", "..", args.eval_path)
    if os.path.exists(eval_dir)==False:
        os.makedirs(eval_dir, exist_ok=True)
    m_ = str.lower(method.replace("-", "_"))
    np.savez("%s/result_%s_%s.npz"%(eval_dir, e_name, m_), data=metrics, data_avg=metrics_avg, method=method, exp_name=args.exp_dir_full)

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()

def build_relu_nn(input_dim, output_dim, hiddens, activation_fn, last_fn=None):
    n_neurons = [input_dim] + hiddens + [output_dim]
    layers = []
    for i in range(len(n_neurons)-1):
        layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
        layers.append(activation_fn())
    if last_fn is not None:
        layers[-1] = last_fn()
    else:
        del layers[-1]
    return nn.Sequential(*layers)

def build_relu_nn1(input_output_dim, hiddens, activation_fn, last_fn=None):
    return build_relu_nn(input_output_dim[0], input_output_dim[1], hiddens, activation_fn, last_fn=last_fn)

def soft_step(x):
    return (torch.tanh(500 * x) + 1)/2

def soft_step_hard(x):
    hard = (x>=0).float()
    soft = (torch.tanh(500 * x) + 1)/2
    return soft + (hard - soft).detach()

def xxyy_2_Ab(x_input):
    xmin, xmax, ymin, ymax = x_input
    A = np.array([
            [-1, 1, 0, 0],
            [0, 0, -1, 1]
        ]).T
    b = np.array([-xmin, xmax, -ymin, ymax])
    return A, b

def xyr_2_Ab(x, y, r, num_edges=8):
    thetas = np.linspace(0, np.pi*2, num_edges+1)[:-1]
    A = np.stack([np.cos(thetas), np.sin(thetas)], axis=-1)
    b = r + x * np.cos(thetas) + y * np.sin(thetas)
    return A, b

# 3d case
def xxyyzz_2_Ab(x_input):
    xmin, xmax, ymin, ymax, zmin, zmax = x_input
    A = np.array([
            [-1, 1, 0, 0, 0, 0],
            [0, 0, -1, 1, 0, 0],
            [0, 0, 0, 0, -1, 1]
        ]).T
    b = np.array([-xmin, xmax, -ymin, ymax, -zmin, zmax])
    return A, b


def xyzr_2_Ab(x, y, z, r, num_edges=8):
    thetas = np.linspace(0, np.pi, num_edges, endpoint=False)
    phis = np.linspace(0, np.pi*2, num_edges, endpoint=False)
    thetas, phis  = np.meshgrid(thetas, phis)
    thetas = thetas.flatten()
    phis = phis.flatten()
    A = np.stack([np.sin(thetas) * np.cos(phis), np.sin(thetas) * np.sin(phis), np.cos(thetas)], axis=-1)
    b = r + x * np.sin(thetas) * np.cos(phis) + y * np.sin(thetas) * np.sin(phis) + z * np.cos(thetas)
    
    inner_val = A[:,0]*x+A[:,1]*y+A[:,2]*z-b
    A[inner_val>0] = -A[inner_val>0]
    b[inner_val>0] = -b[inner_val>0]
    return A, b

# geometry checking
def cross_product(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1


def inner_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def pts_in_poly(traj, polygon, args, obses_1=None, obses_2=None):
    # https://math.stackexchange.com/questions/4183023/check-if-point-is-inside-a-convex-polygon-i-need-an-example-for-a-formular
    # https://inginious.org/course/competitive-programming/geometry-pointinconvex#:~:text=A%20convex%20polygon%20is%20a,of%20each%20of%20the%20segments.
    BATCH = True
    
    n_lines = polygon.shape[0]
    if BATCH:
        # polygon_uns = obses_1  #polygon.unsqueeze(0).unsqueeze(1)
        # polygon_roll = obses_2  #torch.roll(polygon_uns, shifts=-1, dims=2)
        xp = traj[..., 0:1]  # (N, T, 1)
        yp = traj[..., 1:2]  # (N, T, 1) 
        cross_vec = cross_product(obses_1[..., 0] - xp , obses_1[..., 1] - yp, obses_2[..., 0] - xp, obses_2[..., 1] - yp)
    else:
        cross_list = []
        for i in range(n_lines):
            xp = traj[..., 0]
            yp = traj[..., 1]
            x1 = polygon[i, 0]
            y1 = polygon[i, 1]
            x2 = polygon[(i+1) % n_lines, 0]
            y2 = polygon[(i+1) % n_lines, 1]
            cross = cross_product(x1 - xp , y1 - yp, x2 - xp, y2 - yp)
            cross_list.append(cross)
        
        # (N, T, m)  Any traj-pt, ALL poly-edge, cross_loss > 0 => IN_POLY
        # in poly if all cross are greater than zero (bloat to -0.1)
        cross_vec = torch.stack(cross_list, dim=-1)

    res = softmin(cross_vec+0.01, tau=args.smoothing_factor, d=None, dim=-1)
    return res[..., 0]


def seg_int_poly(traj, polygon, args, obses_1=None, obses_2=None):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    n_lines = polygon.shape[0]
    relu = nn.ReLU()
    
    ALLOCATE = False
    BATCH = True
    if BATCH:
        px = obses_1[..., 0]
        py = obses_1[..., 1]
        rx = obses_2[..., 0] - px
        ry = obses_2[..., 1] - py

        qx = traj[:, :-1, 0:1]
        qy = traj[:, :-1, 1:2]
        sx = traj[:, 1:, 0:1] - qx
        sy = traj[:, 1:, 1:2] - qy

        qpx = qx - px
        qpy = qy - py
        r_X_s = cross_product(rx, ry, sx, sy)
        q_p_X_r = cross_product(qpx, qpy, rx, ry)
        q_p_X_s = cross_product(qpx, qpy, sx, sy)
        r_X_s_clipped = r_X_s.clone()
        r_X_s_clipped[r_X_s==0] = 1e-4

        t = q_p_X_s / r_X_s_clipped
        u = q_p_X_r / r_X_s_clipped

        case_1 = torch.logical_and(r_X_s==0, q_p_X_r==0)
        case_2 = torch.logical_and(r_X_s==0, q_p_X_r!=0)
        case_3 = torch.logical_and(r_X_s!=0, torch.logical_and(t*(1-t)>=0, u*(1-u)>=0))

        # case-1 (co-linear) check: tmin, tmax = min(t0, t1), max(t0, t1); tmin<1 AND tmax>0 => INTERSECT(overlap)
        qp_D_r = inner_product(qpx, qpy, rx, ry)
        r_D_r = inner_product(rx, ry, rx, ry)
        s_D_r = inner_product(sx, sy, rx, ry)
        t0 = qp_D_r / r_D_r
        t1 = t0 + s_D_r / r_D_r
        tmin, tmax = torch.minimum(t0, t1), torch.maximum(t0, t1)
        loss_1 = softmin_pairs(1-tmin, tmax, tau=args.smoothing_factor, d=None)
        # case-2 (parallel, but non-intersect) no need to check

        # case-3 (non-parallel) check: 0<=t<=1 AND 0<=u<=1 => INTERSECT
        loss_3 = softmin_pairs(t*(1-t), u*(1-u), tau=args.smoothing_factor, d=None)

        other_case = torch.logical_and(torch.logical_not(case_1), torch.logical_not(case_3)).float()

        seg_loss_vec = loss_1 * (case_1.float()) + loss_3 * (case_3.float()) - 1 * other_case
    else:
        if ALLOCATE:
            seg_loss_list = torch.ones(traj.shape[0], traj.shape[1], n_lines).float().cuda()
        else:
            seg_loss_list = []
        relu = nn.ReLU()
        for i in range(n_lines):
            # one line (from polygon edge): p -> p + r
            px = polygon[i, 0]
            py = polygon[i, 1]
            rx = polygon[(i+1) % n_lines, 0] - px
            ry = polygon[(i+1) % n_lines, 1] - py

            # another line (from trajectory line): q -> q + s
            qx = traj[:, :-1, 0]
            qy = traj[:, :-1, 1]
            sx = traj[:, 1:, 0] - qx
            sy = traj[:, 1:, 1] - qy

            # check intersection: p + tr = q + us
            r_X_s = cross_product(rx, ry, sx, sy)
            q_p_X_r = cross_product(qx - px, qy - py, rx, ry)
            q_p_X_s = cross_product(qx - px, qy - py, sx, sy)
            r_X_s_clipped = r_X_s.clone()
            r_X_s_clipped[r_X_s==0] = 1e-4
            
            t = q_p_X_s / r_X_s_clipped
            u = q_p_X_r / r_X_s_clipped

            case_1 = torch.logical_and(r_X_s==0, q_p_X_r==0)
            case_2 = torch.logical_and(r_X_s==0, q_p_X_r!=0)
            case_3 = torch.logical_and(r_X_s!=0, torch.logical_and(t*(1-t)>=0, u*(1-u)>=0))
            
            # case-1 (co-linear) check: tmin, tmax = min(t0, t1), max(t0, t1); tmin<1 AND tmax>0 => INTERSECT(overlap)
            qp_D_r = inner_product(qx - px, qy - py, rx, ry)
            r_D_r = inner_product(rx, ry, rx, ry)
            s_D_r = inner_product(sx, sy, rx, ry)
            t0 = qp_D_r / r_D_r
            t1 = t0 + s_D_r / r_D_r
            tmin, tmax = torch.minimum(t0, t1), torch.maximum(t0, t1)
            # loss_1 = (torch.relu(1 - tmin) + torch.relu(tmax))/2
            loss_1 = softmin_pairs(1-tmin, tmax, tau=args.smoothing_factor, d=None)

            # case-2 (parallel, but non-intersect) no need to check

            # case-3 (non-parallel) check: 0<=t<=1 AND 0<=u<=1 => INTERSECT
            # loss_3 = (torch.relu(t*(1-t)) + torch.relu(u*(1-u)))/2
            loss_3 = softmin_pairs(t*(1-t), u*(1-u), tau=args.smoothing_factor, d=None)

            other_case = torch.logical_and(torch.logical_not(case_1), torch.logical_not(case_3)).float()

            seg_loss = loss_1 * (case_1.float()) + loss_3 * (case_3.float()) - 1 * other_case
            
            if ALLOCATE:
                seg_loss_list[:, :-1, i] = seg_loss
            else:
                seg_loss_list.append(seg_loss)
            
        
        # (N, T-1, m)  Any traj-seg, Any poly-edge, seg_loss > 0 => INTERSECT
        if ALLOCATE:
            seg_loss_vec = seg_loss_list
        else:
            seg_loss_vec = torch.stack(seg_loss_list, dim=-1)  
    # res = torch.max(seg_loss_vec, dim=-1)[0]
    res = softmax(seg_loss_vec, tau=args.smoothing_factor, d=None, dim=-1)
    if ALLOCATE==False:
        res = torch.cat([res, -1*torch.ones_like(res[:,-1:])], dim=1)
    # print('seg-res', res.shape)
    return res[:, :, 0]


def check_pts_collision(traj, polygon):
    # https://math.stackexchange.com/questions/4183023/check-if-point-is-inside-a-convex-polygon-i-need-an-example-for-a-formular
    # https://inginious.org/course/competitive-programming/geometry-pointinconvex#:~:text=A%20convex%20polygon%20is%20a,of%20each%20of%20the%20segments.
    n_lines = polygon.shape[0]
    cross_list = []
    relu = nn.ReLU()
    for i in range(n_lines):
        xp = traj[..., 0]
        yp = traj[..., 1]
        x1 = polygon[i, 0]
        y1 = polygon[i, 1]
        x2 = polygon[(i+1) % n_lines, 0]
        y2 = polygon[(i+1) % n_lines, 1]

        cross = cross_product(x1 - xp , y1 - yp, x2 - xp, y2 - yp)
        cross_list.append(cross)
    
    # (N, T, m)  Any traj-pt, ALL poly-edge, cross_loss > 0 => IN_POLY
    # in poly if all cross are greater than zero (bloat to -0.1)
    cross_vec = torch.stack(cross_list, dim=-1)
    loss_all = relu(torch.min(cross_vec + 0.01, dim=-1)[0])
    loss = torch.mean(loss_all)
    return loss, loss_all


def check_seg_collision(traj, polygon):
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    n_lines = polygon.shape[0]
    seg_loss_list = []
    relu = nn.ReLU()
    for i in range(n_lines):
        # one line (from polygon edge): p -> p + r
        px = polygon[i, 0]
        py = polygon[i, 1]
        rx = polygon[(i+1) % n_lines, 0] - px
        ry = polygon[(i+1) % n_lines, 1] - py

        # another line (from trajectory line): q -> q + s
        qx = traj[:, :-1, 0]
        qy = traj[:, :-1, 1]
        sx = traj[:, 1:, 0] - qx
        sy = traj[:, 1:, 1] - qy

        # check intersection: p + tr = q + us
        r_X_s = cross_product(rx, ry, sx, sy)
        q_p_X_r = cross_product(qx - px, qy - py, rx, ry)
        q_p_X_s = cross_product(qx - px, qy - py, sx, sy)
        r_X_s_clipped = r_X_s.clone()
        r_X_s_clipped[r_X_s==0] = 1e-4
        
        t = q_p_X_s / r_X_s_clipped
        u = q_p_X_r / r_X_s_clipped

        case_1 = torch.logical_and(r_X_s==0, q_p_X_r==0)
        case_2 = torch.logical_and(r_X_s==0, q_p_X_r!=0)
        case_3 = torch.logical_and(r_X_s!=0, torch.logical_and(t*(1-t)>=0, u*(1-u)>=0))
        
        # case-1 (co-linear) check: tmin, tmax = min(t0, t1), max(t0, t1); tmin<1 AND tmax>0 => INTERSECT(overlap)
        qp_D_r = inner_product(qx - px, qy - py, rx, ry)
        r_D_r = inner_product(rx, ry, rx, ry)
        s_D_r = inner_product(sx, sy, rx, ry)
        t0 = qp_D_r / r_D_r
        t1 = t0 + s_D_r / r_D_r
        tmin, tmax = torch.minimum(t0, t1), torch.maximum(t0, t1)
        loss_1 = (torch.relu(1 - tmin) + torch.relu(tmax))/2

        # case-2 (parallel, but non-intersect) no need to check

        # case-3 (non-parallel) check: 0<=t<=1 AND 0<=u<=1 => INTERSECT
        loss_3 = (torch.relu(t*(1-t)) + torch.relu(u*(1-u)))/2

        seg_loss = loss_1 * (case_1.float()) + loss_3 * (case_3.float())
        seg_loss_list.append(seg_loss)
    
    # (N, T-1, m)  Any traj-seg, Any poly-edge, seg_loss > 0 => INTERSECT
    seg_loss_vec = torch.stack(seg_loss_list, dim=-1)  
    loss_all = torch.mean(relu(seg_loss_vec), dim=-1)
    loss = torch.mean(loss_all)
    return loss, loss_all


### PID ###
def default_line(x, nt):
    x_delta = 10
    ref_v = 5
    segs = torch.zeros(x.shape[0], nt+1, 2).cuda()
    segs[:, 0, :] = x[:, 0:2]
    for i in range(nt):
        segs[:, i+1, 0] = segs[:, i, 0] + x_delta / nt
        segs[:, i+1, 1] = x[:, 1] * 0.8
    return segs

def find_cloest_seg(x, ref_segs):
    segs = torch.zeros(1, 5).to(x.device)
    min_i = torch.argmin(x[0,0]-ref_segs[0,:-1,0])
    segs[0, 0] = ref_segs[0, min_i, 0]
    segs[0, 1] = ref_segs[0, min_i, 1]
    segs[0, 2] = ref_segs[0, min_i+1, 0]
    segs[0, 3] = ref_segs[0, min_i+1, 1]
    segs[0, 4] = ref_v = 5
    return segs

def get_dy_dtheta_dv(x, ref_seg):
    ref_x0, ref_y0, ref_x1, ref_y1, ref_v = ref_seg[:, 0], ref_seg[:, 1], ref_seg[:, 2], ref_seg[:, 3], ref_seg[:, 4]
    ref_theta = torch.arctan2(ref_seg[:, 3]-ref_seg[:, 1], ref_seg[:, 2]-ref_seg[:, 0])
    ego_x = x[:, 0]
    ego_y = x[:, 1]
    ref_dx = (ego_x - ref_x0) * torch.cos(ref_theta) + (ego_y - ref_y0) * torch.sin(ref_theta)
    ref_dy = -(ego_x - ref_x0) * torch.sin(ref_theta) + (ego_y - ref_y0) * torch.cos(ref_theta)
    ref_dphi = x[:, 2] - ref_theta
    ego_dv = (x[:,3]**2+x[:,4]**2)**0.5
    ref_dv = ego_dv - ref_v
    return ref_dy, ref_dphi, ref_dv

def pid_control(x, ref_segs):
    COEFF = [3.69, -18.19, -28.74, -10.81, -59.80, -0.40]
    ref_seg = find_cloest_seg(x, ref_segs)
    ref_dy, ref_dphi, ref_dv = get_dy_dtheta_dv(x, ref_seg)
    u0 = COEFF[0] * ref_dy + COEFF[1] * ref_dphi + COEFF[2] * ref_dv 
    u1 = COEFF[3] * ref_dy + COEFF[4] * ref_dphi + COEFF[5] * ref_dv
    u = torch.stack([u0, u1], dim=-1)
    return u