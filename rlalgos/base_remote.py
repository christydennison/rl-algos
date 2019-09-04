import torch
import torch.nn.functional as F
import gym
import pandas as pd
import os, subprocess, sys
import collections
import random
import numpy as np
from mpi4py import MPI
from torch.multiprocessing import Process
from rcall import meta, settings
from tensorflow.io import gfile


LOG_STD_MAX = 2
LOG_STD_MIN = -20
LOG_PROB_CONST = np.log(2 * np.pi)

Transition = collections.namedtuple("Transition", ["old_obs", "new_obs", "action", "reward", "cost", "done", "log_prob", "step"])
Trajectory = collections.namedtuple("Trajectory", ["s", "sp", "actions", "rewards", "costs", "dones", "log_probs", "steps"])

dtype = torch.float


class MinVar(torch.nn.Module):
    def __init__(self, var_size, fill_value=0.0, min_val=0.0):
        super(MinVar, self).__init__()
        self.var = torch.nn.Parameter(torch.full(var_size, fill_value, dtype=torch.float32, requires_grad=True))
        self.min_val = min_val

    def forward(self):
        return self.var

    def check(self):
        if self.var < self.min_val:
            with torch.no_grad():
                self.var += torch.clamp((self.min_val - self.var), min=self.min_val)
                # self.var = torch.clamp(self.var, min=self.min_val)
            return True
        return False


class SoftVar(torch.nn.Module):
    def __init__(self, var_size, fill_value=-1.0):
        super(SoftVar, self).__init__()
        self.var = torch.nn.Parameter(torch.full(var_size, fill_value, dtype=torch.float32, requires_grad=True))

    def forward(self):
        return F.softplus(self.var)


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


class NetWithVar(Net):
    def __init__(self, input_size, output_sizes, var_size, hidden_sizes=(400, 300), activation=torch.nn.ReLU, output_activation=None):
        super(NetWithVar, self).__init__(input_size, output_sizes, hidden_sizes, activation, output_activation)
        self.var = torch.nn.Parameter(torch.rand(var_size, dtype=torch.float32, requires_grad=True))

    def forward(self, *inp):
        return self.seq(*inp), self.var


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
        old_obs, new_obs, acts, rews, costs, dones, log_probs, steps = [], [], [], [], [], [], [], []
        for transition in transitions:
            old_obs.append(torch.tensor(transition.old_obs).unsqueeze(0))
            new_obs.append(torch.tensor(transition.new_obs).unsqueeze(0))
            acts.append(torch.tensor(transition.action).unsqueeze(0))
            rews.append(torch.tensor(transition.reward))
            costs.append(torch.tensor(transition.cost))
            dones.append(torch.tensor(transition.done))
            log_probs.append(torch.tensor(transition.log_prob).unsqueeze(0))
            steps.append(transition.step)
        return Trajectory(torch.cat(old_obs).float(), torch.cat(new_obs).float(), torch.cat(acts).float(), torch.tensor(rews).unsqueeze(1).float(), torch.tensor(costs).unsqueeze(1).float(), torch.tensor(dones).unsqueeze(1), torch.tensor(log_probs).unsqueeze(1).float(), np.array(steps))

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
            size_diff = self.size - self.maxsize
            print(f"size_diff {size_diff}")
            self.data = self.data[size_diff:]



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
        self.old_obs = None

    def get_env_dims(self):
        return self.env.observation_space.shape[0], self.env.action_space.shape[0]

    def run_trajectory(self, steps_so_far=0):
        done = False
        if self.old_obs is None:
            self.old_obs = self.env.reset()
        steps = 0
        ep_rew = 0
        ep_cost = 0

        # reset to memorize trajectory
        # set_seeds(self.env, self.args.seed)
        # self.env.seed(self.args.seed)

        while not(done or steps >= self.max_ep_len or steps + steps_so_far >= self.args.steps_per_epoch):
            if not self.started and self.start_steps <= self.total_steps:
                self.started = True
                print(f"Start steps completed with {self.total_steps} steps")

            action, log_prob, _, _ = self.act_fn(self.old_obs, random=not self.started)
            action_detached = action.detach().numpy()

            try:
                new_obs, reward, env_done, info = self.env.step(action_detached)
                cost = info.get('cost', 0)
            except Exception:
                print(f"Exception when trying to step at step {steps} with done signal {done}")

            done = False if steps == self.max_ep_len else env_done

            self.replay_buffer.add(Transition(self.old_obs, new_obs, action_detached, reward, cost, done, log_prob.item(), self.total_steps))
            steps += 1
            self.total_steps += 1
            self.old_obs = new_obs if not env_done else None
            ep_rew += reward
            ep_cost += cost

        return steps, ep_rew, ep_cost

    def run_trajectories(self):
        steps = 0
        lens = []
        trajectories = []
        while steps < self.args.steps_per_epoch:
            traj_steps, traj_rew, traj_cost = self.run_trajectory(steps_so_far=steps)
            steps += traj_steps
            lens.append(traj_steps)
            trajectories.append(self.replay_buffer.get())
            self.replay_buffer.clear()

        obs = torch.cat([traj.s for traj in trajectories])
        obs_sp = torch.cat([traj.sp for traj in trajectories])
        actions = torch.cat([traj.actions for traj in trajectories])
        rewards = torch.cat([traj.rewards for traj in trajectories])
        costs = torch.cat([traj.costs for traj in trajectories])
        log_probs = torch.cat([traj.log_probs for traj in trajectories])

        return trajectories, np.array(lens), obs, obs_sp, actions, rewards, costs, log_probs

    def test(self, render=False, record=False):
        if record:
            self.test_env = gym.wrappers.Monitor(self.test_env, self.args.dir, force=True)
        observation = self.test_env.reset()
        total_reward = 0
        total_cost = 0
        steps = 0
        done = False
        while not(done or steps >= self.max_ep_len):
            if render:
                self.test_env.render()

            with torch.no_grad():
                action, _, _, _ = self.act_fn(observation, random=False, deterministic=True)

            observation, reward, done, info = self.test_env.step(action.detach().numpy())
            total_reward += reward
            total_cost += info.get('cost', 0)
            steps += 1
        if render:
            print(f"Ep Length: {steps}")
            print(f"Ep Reward: {total_reward}")
        return steps, total_reward, total_cost

    def done(self):
        # if get_rank() == 0:
        #     self.test(render=True)
        self.env.close()
        sys.exit()



class DataLogger():
    def __init__(self, filename, args, rank=None):
        self.filename = filename
        self.unsaved = {}
        self.rank = rank if rank is not None else get_rank()
        self.use_headers = (rank == 0)
        self.index_key = 'Epoch'
        self.args = args
        self.print_count = 0

    def log_tabular(self, key, val):
        self.unsaved[key] = [val]

    def dump_tabular(self):
        df = pd.DataFrame(self.unsaved, index=self.unsaved[self.index_key])
        if self.rank == 0:
            if self.print_count % 10 == 0:
                print(df.round(3), flush=True)
            else:
                print(df.round(3).to_string().split("\n")[1], flush=True)
        self.print_count += 1
        if self.args.backend == 'local':
            df.to_csv(self.filename, sep='\t', mode='a', header=self.use_headers)
        else:
            with gfile.GFile(self.filename, 'a') as f:
                df.to_csv(f, sep='\t', mode='a', header=self.use_headers)
        self.unsaved = {}
        self.use_headers = False

    def get_data(self):
        return pd.read_table(self.filename)


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


def normalize(tensor):
    mu = tensor.mean()
    std = tensor.std()
    return (tensor - mu) / (std + 1e-8)


def get_gce_output_path(run_id, i):
    # where to put results in gcs
    conf = settings.load()
    name = '-'.join([*run_id.split('_')])
    return f"gs://{conf.GS_BUCKET}/results/{run_id}/rcall"


def gaussian_kl_divergence(mu_0, mu_1, log_std0, log_std1):
    std0 = log_std0.exp()
    std1 = log_std1.exp()
    trace_ratio = torch.sum(std0/std1, dim=-1)
    mu_std_ratio = torch.sum((mu_1 - mu_0)**2 / (std1 + 1e-8), dim=-1)
    k = mu_0.shape[-1]
    det_ratio = torch.sum(log_std1, dim=-1) - torch.sum(log_std0, dim=-1)
    return torch.mean(0.5 * (trace_ratio + mu_std_ratio - k + det_ratio))


def set_seeds(env, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def get_seed(args, rank):
    return dict(enumerate(args.seeds)).get(rank, rank * 1000)


def train_base(args, index=None):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    act_limit = env.action_space.high[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    rank = index if index is not None else get_rank()
    rank_print(rank, f"*** IS DISCRETE: {discrete}")

    seed = get_seed(args, rank)
    set_seeds(env, seed)
    print(f"*** USING SEED {seed}", flush=True)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    return env, test_env, act_limit, obs_dim, act_dim


def print_grad(var, var_name, logger):
    if logger is None:
        return
    try:
        var.register_hook(lambda grad: logger.add(var_name, grad))
    except RuntimeError as re:
        print(var_name, "not registered because no gradient present!")


def get_filenames(args):
    if args.backend == 'local':
        module_root = os.path.dirname(os.path.realpath(__file__))
        logfile = os.path.join(module_root, args.dir, f"{args.exp_name}_data.csv")
        paramsfile = os.path.join(module_root, args.dir, f"{args.exp_name}_params.mdl")
    else:
        output_path = get_gce_output_path(args.exp_name, get_rank())
        logfile = os.path.join(output_path, "data.csv")
        paramsfile = os.path.join(output_path, "params.mdl")
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


def rank_print(rank, s):
    if rank == 0:
        print(s)


def get_rank():
    return MPI.COMM_WORLD.Get_rank()


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpiexec", "-n", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def mpi_avg(tensor):
    numpy_tensor = tensor.numpy()
    size = float(MPI.COMM_WORLD.Get_size())
    buf = np.zeros_like(numpy_tensor)
    MPI.COMM_WORLD.Allreduce(numpy_tensor, buf, op=MPI.SUM)
    np.divide(buf, size, out=buf)
    return buf


def average_gradients(model):
    size = float(MPI.COMM_WORLD.Get_size())
    for param in model.parameters():
        grad_data = param.grad.data.numpy()
        buf = np.zeros_like(grad_data)
        MPI.COMM_WORLD.Allreduce(grad_data, buf, op=MPI.SUM)
        np.divide(buf, size, out=buf)
        param.grad.data = torch.tensor(buf)


# guess MPI is fast enough within a machine that going param by param isn't much slower (like 10 seconds for a 50 epoch run. srsly.)
# def average_gradients(model):
#     # lol this isn't any faster to flatten first
#     size = float(MPI.COMM_WORLD.Get_size())
#     flat_grads = flat_pack_gradients(model)
#     buf = np.zeros_like(flat_grads)
#     MPI.COMM_WORLD.Allreduce(flat_grads, buf, op=MPI.SUM)
#     np.divide(buf, size, out=buf)
#     index = 0
#     for param in model.parameters():
#         param_size = param.shape.numel()
#         param.grad.data = torch.tensor(np.reshape(buf[index : index + param_size], param.shape))
#         index += param_size

# def flat_pack_gradients(model):
#     return torch.cat([torch.flatten(param.grad.data) for param in model.parameters()]).numpy()


def base_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
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
    
    parser.add_argument("--ncpu", type=int, default=1)

    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--trace", action="store_true")

    return parser
