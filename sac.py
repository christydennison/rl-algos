import torch
import time
import gym
import numpy as np
from base import *


LOG_STD_MAX = 2
LOG_STD_MIN = -20

def act(pi, act_limit, obs, act_dim, deterministic=False):
    res = pi(obs)
    mu, log_std = torch.chunk(res, 2, dim=-1)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    if not deterministic:
        std = torch.exp(log_std)
        normal_noise = torch.randn_like(mu)
        unsquashed_sample = mu + normal_noise * std
        log_prob = -0.5 * (normal_noise**2 + 2 * log_std)

        # from https://github.com/openai/jachiam-sandbox/blob/master/Standalone-RL/myrl/algos/sac_new/sac.py#L51
        log_prob -= 2*(np.log(2) - unsquashed_sample - torch.nn.functional.softplus(-2 * unsquashed_sample))

        sample = torch.tanh(unsquashed_sample)
    else:
        sample = np.tanh(mu)
        log_prob = torch.tensor([0.0] * act_dim).float()  # 50%
    return act_limit * sample, log_prob, torch.tanh(mu), torch.tanh(log_std)


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
            return torch.tensor(np.random.uniform(-act_limit, act_limit, act_dim)).float(), torch.tensor([0.0] * act_dim).float(), torch.tensor(0).float(), torch.tensor(0).float()
        return act(pi, act_limit, torch.tensor(obs).float(), act_dim=act_dim, deterministic=deterministic)

    agent = Agent(args, env, curried_act)
    logfile, paramsfile = get_filenames(args)
    log = DataLogger(logfile, args)

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
        entropy_bonuses = []
        entropy = []
        pi_losses = []
        q0_losses = []
        q1_losses = []
        v_losses = []
        act_mean = []

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

                fresh_acts, log_probs, _, log_stds = act(pi, act_limit, old_obs, act_dim=act_dim)  # fresh
                obs_acts_fresh = torch.cat([old_obs, fresh_acts], dim=1)
                entropy_bonus = torch.sum(args.alpha * log_probs, dim=-1).unsqueeze(1)

                with torch.no_grad():
                    q_target = rews + args.gamma * neg_done_floats * v_targ_res
                    v_target = torch.min(q0_res, q1_res) - entropy_bonus
                    q0_fresh_res = q0(obs_acts_fresh)

                b_inv = 1.0/args.batch_size
                q0_loss = b_inv * torch.sum((q0_res - q_target)**2) #q_mse_loss(q0_res, q_target)
                q1_loss = b_inv * torch.sum((q1_res - q_target)**2) # q_mse_loss(q1_res, q_target)
                q_loss = q0_loss + q1_loss
                v_loss = b_inv * torch.sum((v_res - v_target)**2) # v_mse_loss(v_res, v_target)
                pi_loss = -b_inv * torch.sum(q0_fresh_res - entropy_bonus)  # gradient ascent -> descent

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

                epoch_rews.append(np.sum(rews.numpy()))
                entropy_bonuses.append(entropy_bonus.detach().numpy())
                # entropy.append(torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=-1)).detach().numpy())
                entropy.append(torch.sum(log_stds, dim=-1).detach().numpy())
                pi_losses.append(pi_loss.detach().item())
                q0_losses.append(q0_loss.detach().item())
                q1_losses.append(q1_loss.detach().item())
                v_losses.append(v_loss.detach().item())
                act_mean.append(torch.sum(torch.mean(acts, dim=0)))

            ep_rew = np.array(epoch_rews)
            ep_entropy_bonus = np.array(entropy_bonuses)
            ep_entropy = np.array(entropy)
            ep_pi_losses = np.array(pi_losses)
            ep_q0_losses = np.array(q0_losses)
            ep_q1_losses = np.array(q1_losses)
            ep_v_losses = np.array(v_losses)
            ep_act_mean = np.array(act_mean)

            log.log_tabular("ExpName", args.exp_name)
            log.log_tabular("AverageReturn", ep_rew.mean())
            log.log_tabular("StdReturn", ep_rew.std())
            log.log_tabular("MaxReturn", ep_rew.max())
            log.log_tabular("MinReturn", ep_rew.min())
            log.log_tabular("EntropyBonus", ep_entropy_bonus.mean())
            log.log_tabular("Entropy", ep_entropy.mean())
            log.log_tabular("PiLoss", ep_pi_losses.mean())
            log.log_tabular("Q0Loss", ep_q0_losses.mean())
            log.log_tabular("Q1Loss", ep_q1_losses.mean())
            log.log_tabular("VLoss", ep_v_losses.mean())
            log.log_tabular("ActMean", ep_act_mean.mean())
            log.log_tabular("Time", time.time() - start)
            log.log_tabular("Steps", epoch * args.steps_per_epoch + step)
            log.log_tabular("Epoch", epoch)
            # log out entropy
            # plot out action values
            # not working? bump up batch size
            # nuts and bolts talk
            log.dump_tabular()

        # save params at end of epoch
        torch.save(pi, paramsfile)
    agent.done()


def test(args):
    env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    pi = Net(obs_dim, act_dim * 2)  ## mean and std output
    pi.eval()
    _, paramsfile = get_filenames(args)
    torch.load(paramsfile)
    def curried_act(obs, _, deterministic):
        return act(pi, act_limit, torch.tensor(obs).float(), act_dim, deterministic)
    agent = Agent(args, env, curried_act)
    agent.test(steps=2000)


def main():
    '''
    spinup.sac(env_fn, actor_critic=<function mlp_actor_critic>, ac_kwargs={}, seed=0,
    steps_per_epoch=5000, epochs=100, replay_size=1000000, gamma=0.99, polyak=0.995,
    lr=0.001, alpha=0.2, batch_size=100, start_steps=10000, max_ep_len=1000, logger_kwargs={}, save_freq=1)
    '''
    parser = base_argparser()
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args()
    scale_hypers(args)

    if args.test:
        test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()