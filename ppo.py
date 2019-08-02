import torch
import time
import gym
import numpy as np
from base import *


def act(pi, act_limit, obs, deterministic=False):
    res = pi(obs)
    mu, log_std = torch.chunk(res, 2, dim=-1)
    if not deterministic:
        std = torch.exp(log_std)
        with torch.no_grad():
            unsquashed_sample = torch.normal(mean=mu, std=std)
        log_prob = 0.5 * torch.sum(((unsquashed_sample - mu) / std)**2 + 2 * log_std, dim=-1)
        sample = unsquashed_sample
    else:
        sample = np.tanh(mu)
        log_prob = -0.7  # not used, 50%
    return act_limit * sample, log_prob


def cumulative_sum(data, discount):
    longest_T = len(max(data, key=len))
    discounts = torch.tensor([discount**i for i in range(longest_T)])
    discounted = []
    for datum in data:
        for t in range(len(datum)):
            to_end = datum[t:]
            discount_slice = discounts[:len(to_end)]
            discounted.append(torch.sum(discount_slice * to_end))
    return torch.tensor(discounted)


def reward_to_go(args, rewards):
    return cumulative_sum(rewards, args.gamma)


def compute_advantage(args, v_s_res, v_sp_res, rewards):
    delta = rewards + args.gamma * v_sp_res - v_s_res
    adv_unscaled = cumulative_sum(delta, args.gamma * args.lam)
    std = adv_unscaled.std()
    mu = adv_unscaled.mean()
    adv_unit_scaled = (adv_unscaled - mu) / (std + 1e-8)
    return adv_unit_scaled


def train(args):
    env, act_limit, obs_dim, act_dim = train_base(args)

    v = Net(obs_dim, 1)
    pi = Net(obs_dim, act_dim * 2)
    pi_prev = Net(obs_dim, act_dim * 2)
    pi_prev.load_from_model(pi)
    pi = Net(obs_dim, act_dim * 2)  ## mean and std output
    v.eval()
    pi.eval()
    pi_prev.eval()

    def curried_act(obs, random=False, deterministic=False):
        if random:
            return torch.tensor(np.random.uniform(-act_limit, act_limit, act_dim)).float(), None
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic=deterministic)

    agent = Agent(args, env, curried_act)
    log = DataLogger(args.logdir)

    start = time.time()
    v_mse_loss = torch.nn.MSELoss()
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=args.pi_lr)
    v_optimizer = torch.optim.Adam(v.parameters(), lr=args.vf_lr)

    for epoch in range(args.epochs):
        print(f"--------Epoch {epoch}--------")
        step = 0
        epoch_rews = []
        trajectories = []

        while step < args.steps_per_epoch:
            steps = agent.run_trajectory()
            step += steps
            trajectories.append(agent.replay_buffer.get())
            agent.replay_buffer.clear()

        for _ in range(args.train_iters):
            obs = torch.cat([traj[0] for traj in trajectories])
            obs_sp = torch.cat([traj[1] for traj in trajectories])
            log_probs = torch.cat([traj[5] for traj in trajectories])
            rewards = torch.cat([traj[3] for traj in trajectories])

            v_s_res = v(obs)
            v_sp_res = v(obs_sp)
            pi_prev_log_probs = act(pi_prev, act_limit, obs)[1]
            pi_curr_log_probs = log_probs

            # pi loss
            adv = compute_advantage(args, v_s_res, v_sp_res, rewards)
            g = adv.clone()
            g[adv >= 0] *= (1 + args.target_kl)
            g[adv < 0] *= (1 - args.target_kl)
            ratio = pi_prev_log_probs/pi_curr_log_probs

            pi_loss = -torch.mean(torch.min(adv * ratio, g)) / len(trajectories)  # ascent -> descent

            # v loss
            rtg = reward_to_go(args, rewards)
            v_loss = v_mse_loss(v_s_res, rtg)

            v_optimizer.zero_grad()
            pi_optimizer.zero_grad()

            v_loss.backward()
            pi_loss.backward()

            v_optimizer.step()
            pi_optimizer.step()


            epoch_rews.append(rewards.numpy())

        ep_rew = np.array(epoch_rews)
        log.log_tabular("ExpName", args.exp_name)
        log.log_tabular("AverageReturn", ep_rew.mean())
        log.log_tabular("StdReturn", ep_rew.std())
        log.log_tabular("MaxReturn", ep_rew.max())
        log.log_tabular("MinReturn", ep_rew.min())
        log.log_tabular("Time", time.time() - start)
        log.log_tabular("Steps", step * (1 + epoch))
        log.log_tabular("Epoch", epoch)
        log.dump_tabular()

    torch.save(pi, args.paramsdir)
    agent.done()


def test(args):
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    pi = Net(obs_dim, act_dim * 2)  ## mean and std output
    pi.eval()
    torch.load(args.paramsdir)
    def curried_act(obs, random=False, deterministic=True):
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic=True)
    agent = Agent(args, env, curried_act)
    agent.test(steps=2000)


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

    if args.test:
        test(args)
    else:
        if not args.logdir:
            print("ERROR! Must have logdir specified")
        else:
            train(args)


if __name__ == "__main__":
    main()

