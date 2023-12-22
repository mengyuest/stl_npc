import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

def dynamics_per_step(x, u):
    # next_x = x + u * 0.1
    next_x = x * 1.0
    next_x[..., 0] = x[..., 0] + u[..., 0] * np.cos(u[..., 1]) * 0.1
    next_x[..., 1] = x[..., 1] + u[..., 0] * np.sin(u[..., 1]) * 0.1
    return next_x

def dynamics_per_step_torch(x, u):
    # next_x = x + u * 0.1
    next_x = x * 1.0
    next_x[..., 0] = x[..., 0] + u[..., 0] * torch.cos(u[..., 1]) * 0.1
    next_x[..., 1] = x[..., 1] + u[..., 0] * torch.sin(u[..., 1]) * 0.1
    return next_x

def get_reward(trajs):
    # goal = np.array([-2, 1])
    wt = (np.linspace(0, np.pi * 2, trajs.shape[1]) + np.arctan2(2, 2))
    goal_x = -1 + 2 * np.sqrt(2) * np.cos(wt)
    goal_y = 3 + 2 * np.sqrt(2) * np.sin(wt)
    goal = np.stack([goal_x, goal_y], axis=-1)
    reward = -np.sum(np.linalg.norm(trajs - goal, axis=-1), axis=-1)
    return reward

def get_reward_torch(trajs):
    # goal = np.array([-2, 1])
    wt = torch.linspace(0, np.pi * 2, trajs.shape[1]) + np.arctan2(2, 2)
    goal_x = -1 + 2 * np.sqrt(2) * torch.cos(wt)
    goal_y = 3 + 2 * np.sqrt(2) * torch.sin(wt)
    goal = torch.stack([goal_x, goal_y], dim=-1).to(trajs.device)
    reward = -torch.sum(torch.norm(trajs - goal, dim=-1), dim=-1)
    return reward

def to_np(x):
    return x.detach().cpu().numpy()

def solve_cem_func(x0_init, state_dim, nt, action_dim, num_iters, n_cand, n_elites, policy_type, 
              dynamics_step_func, reward_func, transform=None, u_clip=None, seed=None, args=None, 
              extra_dict=None, quiet=False, print_freq=10, device="numpy", visualize=False):
    # x0 (state_dim)
    assert device in ["numpy", "cpu", "gpu"]
    assert x0_init.shape[0] == state_dim and len(x0_init.shape)==1
    assert policy_type in ["direct", "wx+b"] or policy_type.startswith("mlp")
    assert torch.is_tensor(x0_init)==(device!="numpy")
    if u_clip is not None:
        u_min = u_clip[0]
        u_max = u_clip[1]
        assert torch.is_tensor(u_min)==(device!="numpy")
        assert torch.is_tensor(u_max)==(device!="numpy")

    if device=="numpy":
        x0_init = np.repeat(x0_init[None, :], n_cand, axis=0)
    else: 
        x0_init = x0_init.repeat([n_cand, 1])

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    params = {}
    if device=="numpy":
        zeros = np.zeros
        ones = np.ones
    else:
        zeros = torch.zeros
        ones = torch.ones
    if policy_type == "direct":
        params["u"] = {"mean": zeros((nt, action_dim)), "std": ones((nt, action_dim))}
    elif policy_type == "wx+b":
        params["w"]={"mean": zeros((action_dim, state_dim)), "std": ones((action_dim, state_dim))}
        params["b"]={"mean": zeros((action_dim, 1)), "std": ones((action_dim, 1))}
    elif policy_type.startswith("mlp"):
        try:
            dim_list=[state_dim] + [int(xx) for xx in policy_type.split("_")[1:]] + [action_dim]
        except Exception as err:
            print(err)
            print("Unrecognized mlp format: %s, try sth like mlp_64"%(policy_type))
        for i in range(len(dim_list) - 1):
            params["w%0d"%(i)] = {"mean":zeros((dim_list[i+1], dim_list[i])), "std":ones((dim_list[i+1], dim_list[i]))}
            params["b%0d"%(i)] = {"mean":zeros((dim_list[i+1], 1)), "std":ones((dim_list[i+1], 1))}
    else:
        raise NotImplementedError
    
    if device=="gpu":
        params = {k: {"mean": params[k]["mean"].cuda(), "std": params[k]["std"].cuda()} for k in params}
    
    reward_list = []
    best_reward = None
    info = {"reward":[]}
    eps = 1e-5
    prev_reward = None
    for iter_i in range(num_iters):
        # sampling
        if device=="numpy":
            inst_p = {k: np.random.normal(loc=params[k]["mean"], scale=params[k]["std"], size=[n_cand,]+list(params[k]["mean"].shape)) for k in params}
        else:
            inst_p = {k: torch.normal(
                mean = params[k]["mean"][None, :].repeat([n_cand,] + [1 for xxx in list(params[k]["mean"].shape)]), 
                std = params[k]["std"][None, :].repeat([n_cand,] + [1 for xxx in list(params[k]["std"].shape)])) for k in params}
            if device=="gpu":
                inst_p = {k: inst_p[k].cuda() for k in inst_p}

        # generate trajectories based on the dynamics
        x0 = x0_init
        trajs = [x0]
        us = []
        for ti in range(nt):
            if policy_type=="direct":
                u = inst_p["u"][:, ti]
            elif policy_type=="wx+b":
                u = inst_p["w"] @ x0[..., None] + inst_p["b"]
                u = u.squeeze(-1)
            else:
                if device=="numpy":
                    relu = lambda x: np.maximum(x, 0)
                else:
                    relu = torch.nn.ReLU()
                n_layers = len(params)//2
                z = x0[..., None]
                for layer_i in range(n_layers):
                    z = inst_p["w%d"%(layer_i)] @ z + inst_p["b%d"%(layer_i)]
                    if layer_i != n_layers-1:
                        z = relu(z)
                u = z.squeeze(-1)
            # clip the control
            if u_clip is not None:
                if device=="numpy":
                    u = np.clip(u, u_min, u_max)  # TODO need test
                else:
                    u = torch.clip(u, u_min, u_max)
            us.append(u)
            new_x0 = dynamics_step_func(x0, u)
            x0 = new_x0
            trajs.append(x0)
        if device=="numpy":
            trajs = np.stack(trajs, axis=1)
            us = np.stack(us, axis=1)
        else:
            trajs = torch.stack(trajs, axis=1)
            us = torch.stack(us, axis=1)

        # evaluate the performance 
        if transform is not None:
            trajs_aug = transform(trajs, extra_dict)
        else:
            trajs_aug = trajs

        reward = reward_func(trajs_aug)  # should be in (N,)

        # pick the elite and do updates
        if device=="numpy":
            idx = np.argsort(-reward)[:n_elites]
        else:
            idx = torch.argsort(-reward)[:n_elites]
        for k in params:
            if device=="numpy":
                params[k]["mean"] = np.mean(inst_p[k][idx], axis=0)
                params[k]["std"] = np.clip(np.std(inst_p[k][idx], axis=0), a_min=0.01, a_max=None)
            else:
                params[k]["mean"] = torch.mean(inst_p[k][idx], dim=0)
                params[k]["std"] = torch.clip(torch.std(inst_p[k][idx], dim=0), min=0.01, max=None)
        
        if device=="numpy":
            elite_reward = np.mean(reward[idx])
            total_reward = np.mean(reward)
            current_best_reward = np.max(reward)
        else:
            elite_reward = torch.mean(reward[idx]).item()
            total_reward = torch.mean(reward).item()
            current_best_reward = torch.max(reward)
        
        if best_reward is None or current_best_reward > best_reward:
            best_at = iter_i
            best_reward = current_best_reward
            info["best_reward"] = best_reward
            u_best = us[idx[0]]

        stop_criterion = prev_reward is not None and abs(prev_reward-elite_reward)<=eps
        if not quiet:
            if iter_i % (num_iters // print_freq) == 0 or iter_i in [0, num_iters-1] or stop_criterion:
                print("CEM-iter [%04d/%04d] best_r:%.3f elite_r:%.3f total_r:%.3f best_elite_r:%.3f@[%04d]"%(
                    iter_i, num_iters, best_reward, elite_reward, total_reward, best_reward, best_at
                ))
                if visualize:
                    u_mean = us[0]
                    if device=="numpy":
                        for ii in range(idx.shape[0]):
                            plt.plot(trajs[idx[ii],:,0], trajs[idx[ii],:,1], color="blue", alpha=0.3)
                    else:
                        for ii in range(idx.shape[0]):
                            plt.plot(to_np(trajs[idx[ii],:,0]), to_np(trajs[idx[ii],:,1]), color="blue", alpha=0.3)
                    plt.axis("scaled")
                    plt.xlim(-6, 6)
                    plt.ylim(-6, 6)
                    plt.savefig("viz_cem/cem_iter%03d.png"%(iter_i))
                    plt.close()
        info["reward"].append(elite_reward)
        # corner case, early stop ....
        
        if stop_criterion:
            break
        
        prev_reward = elite_reward

    return u_best, params, info

def main():
    os.makedirs("viz_cem",exist_ok=True)
    x0 = torch.from_numpy(np.array([1, 5])).float().cuda()
    u_min = torch.from_numpy(np.array([-3, -3])).float().cuda()
    u_max = torch.from_numpy(np.array([3, 3])).float().cuda()
    u, params, info = solve_cem_func(
        x0_init=x0, state_dim=2, nt=20, action_dim=2, num_iters=5000, n_cand=1000, n_elites=100, 
        policy_type="direct",
        # policy_type="wx+b",
        # policy_type="mlp_16",
        dynamics_step_func=dynamics_per_step_torch, reward_func=get_reward_torch, transform=None, u_clip=(u_min, u_max), seed=1007, args=None, extra_dict=None, quiet=False, print_freq=10,
        device="gpu",
        visualize=True,
    )

    for ti in range(u.shape[0]):
        print(ti, u[ti])
    reward_list = info["reward"]
    plt.plot(range(len(reward_list)), reward_list)
    plt.savefig("viz_cem/cem_reward_curve.png", bbox_inches='tight', pad_inches=0.1)  
    plt.close()

if __name__ == "__main__":
    t1=time.time()
    main()
    t2=time.time()
    print("Finished in %.3f seconds"%(t2 - t1))
