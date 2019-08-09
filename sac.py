import torch
import time
import gym
import numpy as np
from base import *


LOG_PROB_CONST2 = np.log(2)


def act(pi, act_limit, obs, deterministic=False):
    res = pi(obs)
    mu, log_std = torch.chunk(res, 2, dim=-1)
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    if not deterministic:
        std = torch.exp(log_std)
        normal_noise = torch.randn_like(mu)
        # mu.register_hook(lambda grad: print("mu", grad))
        # log_std.register_hook(lambda grad: print("log_std", grad))
        unsquashed_sample = mu + normal_noise * std
        # unsquashed_sample.register_hook(lambda grad: print("unsquashed_sample", grad))
        sample = torch.tanh(unsquashed_sample)

        log_prob = -0.5 * torch.sum(((unsquashed_sample - mu)/(std + 1e-8))**2 + 2 * log_std + LOG_PROB_CONST, dim=-1)

        # from https://github.com/openai/jachiam-sandbox/blob/master/Standalone-RL/myrl/algos/sac_new/sac.py#L51
        log_prob -= torch.sum(2 * (LOG_PROB_CONST2 - unsquashed_sample - torch.nn.functional.softplus(-2 * unsquashed_sample)), dim=-1)
        # log_prob -= torch.sum(torch.log(1 - sample**2), dim=-1)
        log_prob = log_prob.unsqueeze(-1)
        # log_prob.register_hook(lambda grad: print("log_prob", grad))

        # sample.register_hook(lambda grad: print("sample", grad))
    else:
        sample = np.tanh(mu)
        log_prob = torch.tensor([0.0]).float()  # 50%
    return act_limit * sample, log_prob, torch.tanh(mu), torch.tanh(log_std)


def train(args):
    env, test_env, act_limit, obs_dim, act_dim = train_base(args)

    q0 = Net(obs_dim + act_dim, [1])
    q1 = Net(obs_dim + act_dim, [1])
    v = Net(obs_dim, [1])
    v_targ = Net(obs_dim, [1])
    v_targ.load_state_dict(v.state_dict())
    pi = Net(obs_dim, [act_dim, act_dim])  ## mean and std output
    v_targ.eval()  # don't save gradients for targ

    def curried_act(obs, random=False, deterministic=False):
        if random:
            return torch.tensor(np.random.uniform(-act_limit, act_limit, act_dim)).float(), torch.tensor(0.0).float(), torch.tensor(0).float(), torch.tensor(0).float()
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic=deterministic)

    agent = Agent(args, env, test_env, curried_act)
    logfile, paramsfile = get_filenames(args)
    log = DataLogger(logfile, args)

    start = time.time()
    # q0_optimizer = torch.optim.Adam(q0.parameters(), lr=args.lr)
    # q1_optimizer = torch.optim.Adam(q1.parameters(), lr=args.lr)
    # v_optimizer = torch.optim.Adam(v.parameters(), lr=args.lr)
    qv_optimizer = torch.optim.Adam(list(q0.parameters()) + list(q1.parameters()) + list(v.parameters()), lr=args.lr)
    pi_optimizer = torch.optim.Adam(pi.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        print(f"--------Epoch {epoch}--------")
        step = 0
        epoch_rews = []
        entropy_bonuses = []
        # entropy = []
        pi_losses = []
        q0_losses = []
        q1_losses = []
        v_losses = []
        act_mean = []
        ep_lens = []
        step_ranges = []

        while step < args.steps_per_epoch:
            traj_steps, ep_rew = agent.run_trajectory()
            epoch_rews.append(ep_rew)
            ep_lens.append(traj_steps)
            step += traj_steps

            for _ in range(args.max_ep_len):
                old_obs, new_obs, acts, rews, dones, _, steps_for_sample = agent.replay_buffer.sample(args.batch_size)
                step_ranges.append(steps_for_sample)

                old_obs_acts = torch.cat([old_obs, acts], dim=1)
                neg_done_floats = (1 - dones).float()
                q0_res = q0(old_obs_acts)
                q1_res = q1(old_obs_acts)
                v_targ_res = v_targ(new_obs)
                v_res = v(old_obs)
                # q0_res.register_hook(lambda grad: print("q0_res", grad))
                # q1_res.register_hook(lambda grad: print("q1_res", grad))
                # v_targ_res.register_hook(lambda grad: print("v_targ_res", grad))
                # v_res.register_hook(lambda grad: print("v_res", grad))

                fresh_acts, log_probs, _, _ = act(pi, act_limit, old_obs)  # fresh
                obs_acts_fresh = torch.cat([old_obs, fresh_acts], dim=1)
                q0_fresh_res = q0(obs_acts_fresh)
                # q0_fresh_res.register_hook(lambda grad: print("q0_fresh_res", grad))
                entropy_bonus = -args.alpha * log_probs
                # entropy_bonus.register_hook(lambda grad: print("entropy_bonus", grad))

                with torch.no_grad():
                    q_target = (rews + args.gamma * neg_done_floats * v_targ_res)#.detach()
                    v_target = (torch.min(q0_res, q1_res) - entropy_bonus)#.detach()

                q0_loss = 0.5 * torch.mean((q0_res - q_target)**2) #q_mse_loss(q0_res, q_target)
                q1_loss = 0.5 * torch.mean((q1_res - q_target)**2) # q_mse_loss(q1_res, q_target)
                v_loss = 0.5 * torch.mean((v_res - v_target)**2) # v_mse_loss(v_res, v_target)
                qv_loss = q0_loss + q1_loss + v_loss
                # qv_loss.register_hook(lambda grad: print("qv_loss", grad))
                pi_loss = -torch.mean(q0_fresh_res + entropy_bonus)  # gradient ascent -> descent
                # pi_loss.register_hook(lambda grad: print("pi_loss", grad))

                qv_optimizer.zero_grad()
                qv_loss.backward()
                qv_optimizer.step()

                pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_optimizer.step()

                # q0_optimizer.zero_grad()
                # q0_loss.backward()
                # q0_optimizer.step()

                # q1_optimizer.zero_grad()
                # q1_loss.backward()
                # q1_optimizer.step()

                # v_optimizer.zero_grad()
                # v_loss.backward()
                # v_optimizer.step()

                # v_targ_state_dict = v_targ.state_dict()
                # for name, params in v.state_dict().items():
                #     v_targ_state_dict[name] = args.polyak * v_targ_state_dict[name] + (1 - args.polyak) * params
                for target_param, param in zip(v_targ.parameters(), v.parameters()):
                    target_param.data.copy_(target_param.data * args.polyak + param.data * (1.0 - args.polyak))

                entropy_bonuses.append(entropy_bonus.clone().detach().numpy())
                # entropy.append(torch.mean(torch.sum(torch.exp(log_probs) * log_probs, dim=-1)).detach().numpy())
                # entropy.append(torch.sum(log_stds, dim=-1).item())
                pi_losses.append(pi_loss.clone().item())
                q0_losses.append(q0_loss.clone().item())
                q1_losses.append(q1_loss.clone().item())
                v_losses.append(v_loss.clone().item())
                act_mean.append(torch.sum(torch.mean(act_limit - torch.abs(acts), dim=0)))  # how close to the edges of the act limit are the actions

        ep_rew = np.array(epoch_rews)
        ep_entropy_bonus = np.array(entropy_bonuses)
        # ep_entropy = np.array(entropy)
        ep_pi_losses = np.array(pi_losses)
        ep_q0_losses = np.array(q0_losses)
        ep_q1_losses = np.array(q1_losses)
        ep_v_losses = np.array(v_losses)
        ep_act_mean = np.array(act_mean)
        ep_lens_mean = np.array(ep_lens)
        ep_step_ranges = np.array(step_ranges)

        test_ep_len, test_ep_rew = agent.test(render=False)

        log.log_tabular("ExpName", args.exp_name)
        log.log_tabular("AverageReturn", ep_rew.mean())
        log.log_tabular("TestReturn", test_ep_rew)
        log.log_tabular("MaxReturn", ep_rew.max())
        log.log_tabular("MinReturn", ep_rew.min())
        log.log_tabular("StdReturn", ep_rew.std())
        log.log_tabular("AverageEpLen", ep_lens_mean.mean())
        log.log_tabular("TestEpLen", test_ep_len)
        log.log_tabular("EntropyBonus", ep_entropy_bonus.mean())
        # log.log_tabular("Entropy", ep_entropy.mean())
        log.log_tabular("PiLoss", ep_pi_losses.mean())
        log.log_tabular("Q0Loss", ep_q0_losses.mean())
        log.log_tabular("Q1Loss", ep_q1_losses.mean())
        log.log_tabular("VLoss", ep_v_losses.mean())
        # log.log_tabular("ActMean", ep_act_mean.mean())
        log.log_tabular("Time", time.time() - start)
        log.log_tabular("StepRangeMin", ep_step_ranges.min())
        log.log_tabular("StepRangeMax", ep_step_ranges.max())
        log.log_tabular("Steps", epoch * args.steps_per_epoch + step)
        log.log_tabular("Epoch", epoch)
        # plot out action values
        log.dump_tabular()

        # save params at end of epoch
        torch.save(pi, paramsfile)
    agent.done()


def test(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]
    pi = Net(obs_dim, [act_dim, act_dim])  ## mean and std output
    pi.eval()
    _, paramsfile = get_filenames(args)
    torch.load(paramsfile)
    def curried_act(obs, random, deterministic):
        return act(pi, act_limit, torch.tensor(obs).float(), deterministic)
    agent = Agent(args, env, test_env, curried_act)
    agent.test(render=True)


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