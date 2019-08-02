import torch
import time
import gym
import numpy as np
from base import *


def act(pi, act_limit, obs, deterministic=False):
    res = pi(obs)
    mu, log_std = torch.chunk(res, 2, dim=-1)
    if not deterministic:
        # dist = torch.distributions.normal.Normal(mu, torch.exp(log_std))
        # unsquashed_sample = dist.sample()
        # sample = torch.tanh(unsquashed_sample)
        # log_prob = dist.log_prob(unsquashed_sample)
        std = torch.exp(log_std)
        with torch.no_grad():
            unsquashed_sample = torch.normal(mean=mu, std=std)
        log_prob = 0.5 * torch.sum(((unsquashed_sample - mu) / std)**2 + 2 * log_std, dim=-1)
        sample = torch.tanh(unsquashed_sample)
    else:
        sample = np.tanh(mu)
        log_prob = None  # not used
    return act_limit * sample, log_prob


# def get_log_probs(pi, acts, obs):
#     res = pi.forward(obs)  # do we really need to recompute this?
#     mu, log_std = torch.chunk(res, 2)
#     dist = torch.distributions.normal.Normal(mu, torch.exp(log_std))
#     samples = -2.0 * torch.cosh(acts)  # inverting tanh
#     return dist.log_prob(samples)


def train(args):
    env, act_limit, obs_dim, act_dim = train_base(args)

    q0 = Net(obs_dim + act_dim, 1)
    q1 = Net(obs_dim + act_dim, 1)
    v = Net(obs_dim, 1)
    v_targ = Net(obs_dim, 1)
    v_targ.load_from_model(v)
    pi = Net(obs_dim, act_dim * 2)  ## mean and std output
    q0.eval()
    q1.eval()
    v.eval()
    v_targ.eval()
    pi.eval()

    def curried_act(obs, random=False, deterministic=False):
        if random:
            return torch.tensor(np.random.uniform(-act_limit, act_limit, act_dim)).float(), None
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic=deterministic)

    agent = Agent(args, env, curried_act)
    log = DataLogger(args.logdir)

    start = time.time()
    q_mse_loss = torch.nn.MSELoss()
    v_mse_loss = torch.nn.MSELoss()
    q_optimizer = torch.optim.Adam(list(q0.parameters()) + list(q1.parameters()), lr=args.lr)
    v_optimizer = torch.optim.Adam(v.parameters(), lr=args.lr)
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"--------Epoch {epoch}--------")
        step = 0
        epoch_rews = []

        while step < args.steps_per_epoch:
            steps = agent.run_trajectory()
            step += steps

            for _ in range(args.max_ep_len):
                old_obs, new_obs, acts, rews, dones, _ = agent.replay_buffer.sample(args.batch_size)

                old_obs_acts = torch.cat([old_obs, acts], dim=1)
                neg_done_floats = (1 - dones).float()
                q0_res = q0(old_obs_acts)
                q1_res = q1(old_obs_acts)
                v_targ_res = v_targ(new_obs)
                v_res = v(old_obs)

                fresh_acts, log_probs = act(pi, act_limit, old_obs)  # fresh
                obs_acts_fresh = torch.cat([old_obs, fresh_acts], dim=1)
                entropy_bonus = args.alpha * log_probs.unsqueeze(1)

                with torch.no_grad():
                    q_target = rews + args.gamma * neg_done_floats * v_targ_res
                    v_target = torch.min(q0_res, q1_res) - entropy_bonus
                    q0_fresh_res = q0(obs_acts_fresh)

                q0_loss = q_mse_loss(q0_res, q_target)
                q1_loss = q_mse_loss(q1_res, q_target)
                q_loss = q0_loss + q1_loss
                v_loss = v_mse_loss(v_res, v_target)
                pi_loss = -1 * torch.mean(q0_fresh_res - entropy_bonus)  # gradient ascent -> descent

                q_optimizer.zero_grad()
                v_optimizer.zero_grad()
                pi_optimizer.zero_grad()

                q_loss.backward()
                v_loss.backward()
                pi_loss.backward()

                q_optimizer.step()
                v_optimizer.step()
                pi_optimizer.step()

                v_targ_state_dict = v_targ.state_dict()
                for name, params in v.state_dict().items():
                    v_targ_state_dict[name] = args.polyak * v_targ_state_dict[name] + (1 - args.polyak) * params

                epoch_rews.append(rews.numpy())

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

    pi.save(args.paramsdir)
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
    spinup.sac(env_fn, actor_critic=<function mlp_actor_critic>, ac_kwargs={}, seed=0,
    steps_per_epoch=5000, epochs=100, replay_size=1000000, gamma=0.99, polyak=0.995,
    lr=0.001, alpha=0.2, batch_size=100, start_steps=10000, max_ep_len=1000, logger_kwargs={}, save_freq=1)
    '''
    parser = base_argparser()
    parser.add_argument("--polyak", type=int, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.2)
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