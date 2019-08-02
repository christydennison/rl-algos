import torch
import gym
import pandas as pd
import os
import collections
import random
import numpy as np


Transition = collections.namedtuple("Transition", ["old_obs", "new_obs", "action", "reward", "done", "log_prob"])


dtype = torch.float

class Net(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(400, 300), activation=torch.nn.ReLU, output_activation=None):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
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
        layer = torch.nn.Linear(start_size, self.output_size)
        layers.append(layer)
        if self.output_activation:
            layers.append(self.output_activation())
        self.seq = torch.nn.Sequential(*layers).float()

    def get_params(self):
        return self.state_dict()

    def load_from_model(self, model):
        self.load_state_dict(model.get_params())
        self.eval()

    # def save(self, path):
    #     torch.save(self.state_dict(), path)

    def forward(self, *inp):
        # [i.float() for i in inp]
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
        old_obs, new_obs, acts, rews, dones, log_probs = [], [], [], [], [], []
        for transition in transitions:
            old_obs.append(torch.tensor(transition.old_obs).unsqueeze(0))
            new_obs.append(torch.tensor(transition.new_obs).unsqueeze(0))
            acts.append(transition.action.unsqueeze(0))
            rews.append(torch.tensor(transition.reward))
            dones.append(torch.tensor(transition.done))
            log_probs.append(torch.tensor(transition.log_prob).unsqueeze(0))
        return torch.cat(old_obs).float(), torch.cat(new_obs).float(), torch.cat(acts).float(), torch.tensor(rews).unsqueeze(1).float(), torch.tensor(dones).unsqueeze(1), torch.cat(log_probs).float()

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
            for _ in range(size_diff):
                self.data.pop(0)



class Agent():
    def __init__(self, args, env, act_fn):
        self.args = args
        self.act_fn = act_fn
        self.env = env
        self.max_ep_len = self.args.steps_per_epoch or self.env.spec.max_episode_steps
        print("Max Episode Length", self.max_ep_len)
        self.replay_buffer = ReplayBuffer(maxsize=args.replay_size)
        self.started = False

    def get_env_dims(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]

    def run_trajectory(self):
        done = False
        old_obs = self.env.reset()
        steps = 0
        while not(done or steps >= self.max_ep_len):
            action, log_prob = self.act_fn(old_obs, random=not self.started and self.args.start_steps > steps)

            new_obs, reward, done, _ = self.env.step(action.detach().numpy())

            self.replay_buffer.add(Transition(old_obs, new_obs, action, reward, done, log_prob))
            steps += 1

        self.started = True
        return steps

    def run_trajectories(self):
        steps = 0
        while steps < self.args.steps_per_epoch:
            steps += self.run_trajectory()
        print(f"Batch collected {steps} samples")
        return steps

    def test(self, steps=1000):
        observation = self.env.reset()
        total_reward = 0
        for _ in range(steps):
            self.env.render()

            with torch.no_grad():
                action, _ = self.act_fn(observation, random=False, deterministic=True)

            observation, reward, done, _ = self.env.step(action.detach().numpy())
            total_reward += reward
            if done:
                break

    def done(self):
        self.test()
        self.env.close()



class DataLogger():
    def __init__(self, filename):
        self.filename = filename
        self.unsaved = {}
        self.use_headers = True
        self.index_key = "Epoch"

    def log_tabular(self, key, val):
        self.unsaved[key] = [val]

    def dump_tabular(self):
        df = pd.DataFrame(self.unsaved, index=self.unsaved[self.index_key])
        print(df)
        df.to_csv(self.filename, sep='\t', mode='a', header=self.use_headers)
        self.unsaved = {}
        self.use_headers = False

    def get_data(self):
        return pd.read_table(self.filename)


def train_base(args):
    env = gym.make(args.env_name)
    act_limit = env.action_space.high[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    print("*** IS DISCRETE: ", discrete)

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, act_limit, obs_dim, act_dim


def base_argparser(*moreargs):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--paramsdir", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=5000)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=int, default=0.99)
    parser.add_argument("--start_steps", type=int, default=0)
    parser.add_argument("--test", action="store_true")


    return parser

