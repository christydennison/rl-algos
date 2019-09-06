import torch
import time
import gym
import numpy as np
from rlalgos.base import *
from rcall import meta
torch.set_num_threads(1)


def gaussian_logprob(action, mu, log_std):
    std = torch.exp(log_std)
    log_prob = torch.sum(-0.5 * (((action - mu)/(std + 1e-8))**2 + 2 * log_std + LOG_PROB_CONST), dim=-1).unsqueeze(-1)
    return log_prob


def act(pi, act_limit, obs, deterministic=False, grad_logger=None):
    res = pi(obs)
    mu, log_std = torch.chunk(res, 2, dim=-1)
    if not deterministic:
        std = torch.exp(log_std)
        normal_noise = torch.randn_like(mu)
        with torch.no_grad():
            sample = mu.detach() + normal_noise * std.detach()
        log_prob = gaussian_logprob(sample, mu, log_std)
    else:
        sample = mu
        log_prob = torch.tensor([0.0]).float()  # 50%
    return act_limit * sample, log_prob, mu, log_std


class GradLogger():
    def __init__(self):
        self.epoch_logs = collections.defaultdict(lambda: collections.defaultdict(list)) # {epoch => {var_name: [grad, grad, grad]}}
        self.epoch = 0

    def set_current_epoch(self, epoch):
        self.epoch = epoch

    def add(self, name, grad):
        self.epoch_logs[self.epoch][name].append(grad)

    def get_norms(self, epoch, name):
        return torch.norm(torch.cat(self.epoch_logs[epoch][name])).item()


def train(args):
    env, test_env, act_limit, obs_dim, act_dim = train_base(args)

    v = Net(obs_dim, [1], activation=torch.nn.Tanh)
    pi = Net(obs_dim, [act_dim, act_dim], activation=torch.nn.Tanh)
    pi_prev = Net(obs_dim, [act_dim, act_dim], activation=torch.nn.Tanh)
    pi_prev.load_state_dict(pi.state_dict())
    pi_prev.eval()
    grad_logger = GradLogger()
    rank = get_rank()

    def curried_act(obs, random=False, deterministic=False):
        if random:
            return torch.tensor(np.random.uniform(-act_limit, act_limit, act_dim)).float(), torch.tensor(0.0).float(), torch.tensor(0).float(), torch.tensor(0).float()
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic=deterministic, grad_logger=grad_logger)

    agent = Agent(args, env, test_env, curried_act)
    logfile, paramsfile = get_filenames(args)
    log = DataLogger(logfile, args, rank)

    start = time.time()
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=args.pi_lr)
    v_optimizer = torch.optim.Adam(v.parameters(), lr=args.vf_lr)

    step = 0
    for epoch in range(args.epochs):
        rank_print(rank, f"--------Epoch {epoch}--------")
        pi_losses = []
        v_losses = []
        act_mean = []
        max_std = 0
        min_std = 0
        entropy = []
        grad_logger.set_current_epoch(epoch)

        trajectories, ep_lens, obs, obs_sp, actions, rewards, costs, log_probs = agent.run_trajectories()
        rtg = torch.cat([cumulative_sum(traj.rewards, args.gamma) for traj in trajectories])
        step += ep_lens.sum()

        v_s_res = v(obs)
        v_sp_res = v(obs_sp)

        # loop over lengths and slice so we only do 1 FP with v_s/v_sp
        adv_unscaled = []
        traj_index = 0
        for tau in trajectories:
            traj_len = len(tau.s)
            traj_start = traj_index
            traj_end = traj_index + traj_len
            adv_tau = compute_advantage(args, v_s_res[traj_start:traj_end], v_sp_res[traj_start:traj_end], rewards[traj_start:traj_end])
            adv_unscaled.append(adv_tau)
            traj_index += traj_len

        adv = normalize(torch.cat(adv_unscaled))
        g = adv.clone()
        g[adv >= 0] *= (1 + args.clip_ratio)
        g[adv < 0] *= (1 - args.clip_ratio)

        _, _, pi_prev_mus, pi_prev_log_stds = act(pi_prev, act_limit, obs)
        pi_prev_log_probs = log_probs


        for i_train in range(args.train_iters):

            pi_mus, pi_log_stds = torch.chunk(pi(obs), 2, dim=-1)
            pi_log_probs = gaussian_logprob(actions/act_limit, pi_mus, pi_log_stds)

            max_std = torch.max(pi_log_stds)
            min_std = torch.min(pi_log_stds)

            # break early if kl > target_kl
            with torch.no_grad():
                kl = gaussian_kl_divergence(pi_prev_log_stds, pi_log_stds, pi_prev_mus, pi_mus)
            ave_kl = mpi_avg(kl)
            if ave_kl > args.target_kl * 1.5:
                rank_print(rank, f"Breaking early at optimization step {i_train} with KL div {ave_kl}")
                break

            ratio = torch.exp(pi_log_probs - pi_prev_log_probs)
            entropy.append(-pi_log_probs.detach().numpy())

            # train pi with advantages pre-calculated
            pi_loss = -torch.mean(torch.min(adv * ratio, g))  # ascent -> descent

            pi_optimizer.zero_grad()
            pi_loss.backward()
            average_gradients(pi)
            pi_optimizer.step()
            pi_losses.append(pi_loss.clone().detach())


        for i_train in range(args.train_iters):

            # train V with fresh data
            v_s_res = v(obs)
            v_loss = torch.mean((v_s_res - rtg)**2)

            v_optimizer.zero_grad()
            v_loss.backward()
            average_gradients(v)
            v_optimizer.step()
            v_losses.append(v_loss.clone().detach())


        # set to optimized pi's params at end of optimization
        pi_prev.load_state_dict(pi.state_dict())


        ep_pi_losses = np.array(pi_losses)
        ep_v_losses = np.array(v_losses)
        ep_lens_test = []
        ep_rew_test = []
        ep_entropy = np.array(entropy)
        ep_rews = np.array([traj.rewards.sum() for traj in trajectories])

        for _ in range(args.test_iters):
            test_ep_len, test_ep_rew, _ = agent.test(render=False)
            ep_lens_test.append(test_ep_len)
            ep_rew_test.append(test_ep_rew)

        log.log_tabular("ExpName", args.exp_name)
        log.log_tabular("AverageReturn", ep_rews.mean().item())
        log.log_tabular("TestReturn", np.array(ep_rew_test).mean())
        log.log_tabular("MaxReturn", ep_rews.max().item())
        log.log_tabular("MinReturn", ep_rews.min().item())
        log.log_tabular("StdReturn", ep_rews.std().item())
        log.log_tabular("AverageEpLen", ep_lens.mean())
        log.log_tabular("TestEpLen", np.array(ep_lens_test).mean())
        log.log_tabular("PiLoss", ep_pi_losses.mean() if len(ep_pi_losses) > 0 else 0)
        log.log_tabular("VLoss", ep_v_losses.mean())
        log.log_tabular("PiLogStdMax", max_std.item())
        log.log_tabular("PiLogStdMin", min_std.item())
        log.log_tabular("Entropy", ep_entropy.mean())
        log.log_tabular("Time", time.time() - start)
        log.log_tabular("Steps", step)
        log.log_tabular("Epoch", epoch)
        log.dump_tabular()

        if rank == 0:
            torch.save(pi, paramsfile)
    agent.done()


def test(args):
    env, test_env, act_limit, obs_dim, act_dim = train_base(args)
    pi = Net(obs_dim, [act_dim, act_dim], activation=torch.nn.Tanh)

    def curried_act(obs, random=False, deterministic=False):
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic=True)

    agent = Agent(args, env, test_env, curried_act)
    _, paramsfile = get_filenames(args)
    torch.load(paramsfile)
    agent.test(render=True)


def main():
    '''
    spinup.ppo(env_fn, actor_critic=<function mlp_actor_critic>, ac_kwargs={}, seed=0,
    steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001,
    train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs={}, save_freq=10)
    '''
    parser = base_argparser()
    parser.add_argument("--pi_lr", type=float, default=0.0003)
    parser.add_argument("--vf_lr", type=float, default=0.001)
    parser.add_argument("--train_iters", type=int, default=80)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--target_kl", type=float, default=0.01)
    args = parser.parse_args()
    # scale_hypers(args)

    args.start_steps = 0

    if args.test:
        test(args)
    else:
        if args.remote:
            name = '-'.join([*args.exp_name.split('_')])
            meta.call(
                backend=args.backend,
                fn=train,
                kwargs=dict(args=args),
                log_relpath=name,
                job_name=name,
                update=args.update,
                num_gpu=0,
                num_cpu=16,
                mpi_proc_per_machine=3,
                mpi_machines=1,
            )
        else:
            mpi_fork(args.ncpu)
            train(args)


if __name__ == "__main__":
    main()

