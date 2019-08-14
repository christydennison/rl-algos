import torch
import gym
import pandas as pd
import os
import collections
import random
import numpy as np


LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_PROB_CONST = np.log(2 * np.pi)

Transition = collections.namedtuple("Transition", ["old_obs", "new_obs", "action", "reward", "done", "log_prob", "step"])

dtype = torch.float


class SplitNet(torch.nn.Module):
    def __init__(self, input_size, output_sizes, activation=None):
        super(SplitNet, self).__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.activation = activation
        self._build()
    
    def _build(self):
        layers = []
        for output_size in self.output_sizes:
            seq_layers = []
            seq_layers.append(torch.nn.Linear(self.input_size, output_size))
            if self.activation is not None:
                seq_layers.append(self.activation())
            layers.append(torch.nn.Sequential(*seq_layers))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        res = []
        # apply in parallel
        for layer in self.layers:
            res.append(layer(x))
        return torch.cat(res, dim=-1)


class Net(torch.nn.Module):
    def __init__(self, input_size, output_sizes, hidden_sizes=(400, 300), activation=torch.nn.ReLU, output_activation=None):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.output_activation = output_activation
        self._build()

    def _build(self):
        layers = []
        start_size = self.input_size

        for size in self.hidden_sizes:
            layer = torch.nn.Linear(start_size, size)
            layers.append(layer)
            layers.append(self.activation())
            start_size = size

        if len(self.output_sizes) == 1:
            layer = torch.nn.Linear(start_size, self.output_sizes[0])
        else:
            layer = SplitNet(start_size, self.output_sizes, self.output_activation)
        layers.append(layer)

        if self.output_activation:
            layers.append(self.output_activation())
        self.seq = torch.nn.Sequential(*layers).float()

    def get_params(self):
        return self.state_dict()

    def forward(self, *inp):
        return self.seq(*inp)


class ReplayBuffer():
    def __init__(self, maxsize):
        self.data = []
        self.maxsize = maxsize

    def add(self, data):
        self.data.append(data)

    @property
    def size(self):
        return len(self.data)

    def _split(self, transitions):
        old_obs, new_obs, acts, rews, dones, log_probs, steps = [], [], [], [], [], [], []
        for transition in transitions:
            old_obs.append(torch.tensor(transition.old_obs).unsqueeze(0))
            new_obs.append(torch.tensor(transition.new_obs).unsqueeze(0))
            acts.append(torch.tensor(transition.action).unsqueeze(0))
            rews.append(torch.tensor(transition.reward))
            dones.append(torch.tensor(transition.done))
            log_probs.append(torch.tensor(transition.log_prob).unsqueeze(0))
            steps.append(transition.step)
        return torch.cat(old_obs).float(), torch.cat(new_obs).float(), torch.cat(acts).float(), torch.tensor(rews).unsqueeze(1).float(), torch.tensor(dones).unsqueeze(1), torch.tensor(log_probs).unsqueeze(1).float(), np.array(steps)

    def sample(self, batch_size):
        transitions = random.sample(self.data, batch_size)
        self.purge()
        return self._split(transitions)

    def get(self):
        return self._split(self.data)

    def clear(self):
        self.data = []

    def purge(self):
        if self.size > self.maxsize:
            size_diff = self.maxsize - self.size
            print(f"size_diff {size_diff}")
            for _ in range(size_diff):
                self.data.pop(0)



class Agent():
    def __init__(self, args, env, test_env, act_fn):
        self.args = args
        self.act_fn = act_fn
        self.env = env
        self.test_env = test_env
        self.max_ep_len = self.args.max_ep_len or self.env.spec.max_episode_steps
        print("Max Episode Length", self.max_ep_len)
        self.replay_buffer = ReplayBuffer(maxsize=args.replay_size)
        self.started = False  # this only flips once per lifetime of Agent
        self.total_steps = 0
        self.start_steps = args.start_steps if args.start_steps is not None else 0

    def get_env_dims(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]

    def run_trajectory(self):
        done = False
        old_obs = self.env.reset()
        steps = 0
        ep_rew = 0

        while not(done or steps >= self.max_ep_len):
            if not self.started and self.start_steps <= self.total_steps:
                self.started = True
                print(f"Start steps completed with {self.total_steps} steps")

            action, log_prob, _, _ = self.act_fn(old_obs, random=not self.started)
            action_detached = action.detach().numpy()

            try:
                new_obs, reward, done, _ = self.env.step(action_detached)
            except Exception:
                print(f"Exception when trying to step at step {steps} with done signal {done}")
            done = False if steps == self.max_ep_len else done

            self.replay_buffer.add(Transition(old_obs, new_obs, action_detached, reward, done, log_prob.item(), self.total_steps))
            steps += 1
            self.total_steps += 1
            old_obs = new_obs
            ep_rew += reward

        return steps, ep_rew

    def run_trajectories(self):
        steps = 0
        while steps < self.args.steps_per_epoch:
            steps += self.run_trajectory()
        print(f"Batch collected {steps} samples")
        return steps

    def test(self, render=False, record=False):
        if record:
            self.test_env = gym.wrappers.Monitor(self.test_env, self.args.dir, force=True)
        observation = self.test_env.reset()
        total_reward = 0
        steps = 0
        done = False
        while not(done or steps >= self.max_ep_len):
            if render:
                self.test_env.render()

            with torch.no_grad():
                action, _, _, _ = self.act_fn(observation, random=False, deterministic=True)

            observation, reward, done, _ = self.test_env.step(action.detach().numpy())
            total_reward += reward
            steps += 1
        if render:
            print(f"Ep Length: {steps}")
            print(f"Ep Reward: {total_reward}")
        return steps, total_reward

    def done(self):
        self.test(render=True)
        self.env.close()



class DataLogger():
    def __init__(self, filename, args):
        self.filename = filename
        self.unsaved = {}
        self.use_headers = True
        self.index_key = 'Epoch'
        self.args = args
        self.print_count = 0

    def log_tabular(self, key, val):
        self.unsaved[key] = [val]

    def dump_tabular(self):
        df = pd.DataFrame(self.unsaved, index=self.unsaved[self.index_key])
        if self.print_count % 10 == 0:
            print(df.round(3))
            self.print_count += 1
        else:
            print(df.round(3).to_string().split("\n")[1])
        df.to_csv(self.filename, sep='\t', mode='a', header=self.use_headers)
        self.unsaved = {}
        self.use_headers = False

    def get_data(self):
        return pd.read_table(self.filename)


def train_base(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    act_limit = env.action_space.high[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    print("*** IS DISCRETE: ", discrete)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, test_env, act_limit, obs_dim, act_dim


def get_filenames(args):
    module_root = os.path.dirname(os.path.realpath(__file__))
    logfile = os.path.join(module_root, args.dir, args.exp_name + "_data.csv")
    paramsfile = os.path.join(module_root, args.dir, args.exp_name + "_params.mdl")
    if args.clear:
        if os.path.exists(logfile):
            os.remove(logfile)
        if os.path.exists(paramsfile):
            os.remove(paramsfile)
    return logfile, paramsfile


def scale_hypers(args):
    args.max_ep_len = int(args.max_ep_len * args.scale_hypers)
    args.batch_size = int(args.batch_size * args.scale_hypers)
    args.steps_per_epoch = int(args.steps_per_epoch * args.scale_hypers)
    args.start_steps = int(args.start_steps * args.scale_hypers)
    args.epochs = int(args.epochs * args.scale_hypers)


def base_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=5000)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--scale_hypers", type=float, default=1.0)
    parser.add_argument("--test_iters", type=int, default=10)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--clear", action="store_true", help="clear out previous logs")
    
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--update", action="store_true")

    return parser
