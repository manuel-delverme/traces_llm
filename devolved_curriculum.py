import collections

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

DEBUG = False  # sys.gettrace() is not None


class Hyperparameters:
    NUM_CLASSES = 3
    TRACE_LEN = 3
    TRACE_DIM = 2
    BATCH_SIZE = 32
    EPOCHS = 40
    SL_LEARNING_RATE = 0.001
    RL_LEARNING_RATE = 3e-4
    NUM_RL_STEPS = 100 if DEBUG else 50_000
    discount = 0.99


TASK_KEY = "task"
HISTORY_KEY = "history"


def pad_traces(trace: np.ndarray) -> np.ndarray:
    trace_with_padding = np.zeros((Hyperparameters.TRACE_LEN, Hyperparameters.TRACE_DIM))
    trace_with_padding[-len(trace):] = trace
    return trace_with_padding


def samples_to_dataloader(traces: np.ndarray, labels: np.ndarray) -> DataLoader:
    padded_traces = np.array([pad_traces(trace) for trace in traces])
    padded_traces, labels = torch.FloatTensor(padded_traces), torch.LongTensor(labels)
    dataset = TensorDataset(padded_traces, labels)
    return DataLoader(dataset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=True)


class TraceGenerator:
    def __init__(self, num_samples=1000, noise_level=0.05, trace_length=5):
        self.num_samples = num_samples
        self.noise_level = noise_level
        self.trace_length = trace_length

    def generate(self):
        directions = np.array([[0, 1], [1, 0], [1, 1]])

        steps = np.arange(0, self.trace_length)[None, :].repeat(2, 0)
        prototypes = np.einsum("dt, sd -> std", steps, directions)

        samples_per_class = self.num_samples // Hyperparameters.NUM_CLASSES

        # AAA...BBBB...CCC
        samples = np.repeat(prototypes, samples_per_class, 0)

        noise = np.random.normal(0, self.noise_level, samples.shape)
        # initial point is always 0
        noise[:, 0, :] = 0.

        noisy_samples = samples + noise
        labels = np.repeat([0, 1, 2], samples_per_class)

        dataloader = samples_to_dataloader(noisy_samples, labels)
        return dataloader


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(Hyperparameters.TRACE_LEN * Hyperparameters.TRACE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, Hyperparameters.NUM_CLASSES)
        )

    def forward(self, x):
        x = self.preprocess(x)
        return self.net(x)

    def preprocess(self, x):
        assert x.shape == (x.shape[0], Hyperparameters.TRACE_LEN, Hyperparameters.TRACE_DIM)
        return x.view(x.size(0), -1)


def train_model(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Hyperparameters.SL_LEARNING_RATE)
    accuracies = collections.deque([], maxlen=5)

    for epoch in range(Hyperparameters.EPOCHS):
        corrects = 0
        total = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            corrects += (output.argmax(1) == target).sum()
            total += len(target)
            loss.backward()
            optimizer.step()
        accuracy = corrects / total
        accuracies.append(accuracy)
        print(f"Epoch {epoch}/{Hyperparameters.EPOCHS}, Loss: {loss.item():.4f}, accuracy: {accuracy}")
        if len(accuracies) == accuracies.maxlen and all(a > 0.98 for a in accuracies):
            break
    return accuracies[-1]


class CustomHandwritingEnv(gym.Env):
    def __init__(self, decoder: nn.Module):
        self.action_space_shape = (Hyperparameters.TRACE_DIM + 1,)
        self.observation_space = gym.spaces.Dict({
            HISTORY_KEY: gym.spaces.Box(
                low=-1, high=1,
                shape=(Hyperparameters.TRACE_LEN, Hyperparameters.TRACE_DIM)
            ),
            TASK_KEY: gym.spaces.Discrete(Hyperparameters.NUM_CLASSES)
        })
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.action_space_shape)
        self.decoder = decoder
        self.target_label = None
        self.points = None
        self.episode_length = None
        self.reset()

    def reset(self, **kwargs):
        self.target_label = np.random.randint(0, Hyperparameters.NUM_CLASSES)
        self.points = collections.deque([np.array([0.0, 0.0]), ] * Hyperparameters.TRACE_LEN,
                                        maxlen=Hyperparameters.TRACE_LEN)  # 1. Ensure float type
        self.episode_length = 0  # Reset episode length
        return self.get_state(), {TASK_KEY: self.target_label}

    def get_state(self):
        return {HISTORY_KEY: np.array(self.points), TASK_KEY: self.target_label}

    def step(self, action):
        assert action.shape == self.action_space_shape
        action, lift = action[:2], action[2]
        new_point = self.points[-1] + action
        self.points.append(new_point)
        self.episode_length += 1

        # Decode the trace using the trained decoder
        with torch.no_grad():
            trace_tensor = torch.tensor(self.points, dtype=torch.float32).unsqueeze(0)
            output = self.decoder(trace_tensor)
            pred_label = torch.argmax(output, dim=1).item()

        done = lift > 0
        distance_moved = np.linalg.norm(action)

        reward = 0
        if lift > 0:
            if pred_label == self.target_label:
                reward = 1
            else:
                reward = -10
        elif self.episode_length >= Hyperparameters.TRACE_LEN - 1:
            done = True

        reward -= distance_moved

        return self.get_state(), reward, done, done, {}


def run_rl_loop(env, agent) -> (DataLoader, PPO):
    agent.learn(total_timesteps=Hyperparameters.NUM_RL_STEPS, log_interval=20)
    agent.save("agent")

    traces, labels, reward = deterministic_rollout(env, agent)
    dataloader = samples_to_dataloader(traces, labels)
    return dataloader, agent, reward


def plot_trajectories(dataloader, ts):
    traces, labels = dataloader.dataset.tensors
    traces_by_label = [[] for d in set(labels)]
    for t, l in zip(traces, labels):
        traces_by_label[l].append(t)
    labeld = set()
    for l, t in enumerate(traces_by_label):
        for ti in t:
            c = ["r", "g", "b"][l]
            style = ["dashed", "dashdot", "dotted"][l]
            kwargs = dict(c=c, linestyle=style)
            if l not in labeld:
                kwargs["label"] = l
                labeld.add(l)
            lines = np.array(ti)[:, :2]
            # lines += np.random.normal(0, 0.001, lines.shape)

            plt.plot(*lines.T, **kwargs, alpha=0.1)
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.legend()
    plt.title(f"step {ts}")
    plt.show()


def deterministic_rollout(env, agent):
    traces, labels = [], []
    total_reward = 0.
    for _ in range(100):
        state, info = env.reset()
        target_label = info[TASK_KEY]
        done = False
        trace = []

        while not done:
            state_tensor = {
                HISTORY_KEY: torch.tensor(state[HISTORY_KEY], dtype=torch.float32).unsqueeze(0),
                TASK_KEY: torch.tensor(state[TASK_KEY], dtype=torch.long).unsqueeze(0)

            }
            action, _ = agent.predict(state_tensor)
            xyz = state[HISTORY_KEY][-1].tolist()  # + [action[0, -1], ]
            trace.append(xyz)

            new_state, reward, done, _, info = env.step(action.squeeze(0))
            total_reward += reward
            state = new_state

        xyz = state[HISTORY_KEY][-1].tolist()  # + [action[0, -1], ]
        trace.append(xyz)

        traces.append(trace)
        labels.append(target_label)
    return traces, labels, total_reward


def plot_reward_history(episode, all_reward_history, rewards, labels):
    print(f"Episode {episode}, Total Reward: {sum(rewards)}")
    for label in set(labels):
        reward_history = np.array(all_reward_history)[np.array(labels) == label]
        window_size = len(reward_history) // 10
        smoothed_reward_history = np.convolve(reward_history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_reward_history)
    plt.legend(set(labels))
    plt.show()


def main():
    trace_gen = TraceGenerator(trace_length=Hyperparameters.TRACE_LEN)
    dataloader = trace_gen.generate()
    discriminator = Decoder()

    env = CustomHandwritingEnv(discriminator)
    agent = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="/tmp/fetch_reach_tensorboard/",
                learning_rate=Hyperparameters.RL_LEARNING_RATE, gamma=Hyperparameters.discount)
    if DEBUG:
        # Load a good model to speed up for now
        agent2 = agent.load("agent")
        agent.set_parameters(agent2.get_parameters())

    plot_trajectories(dataloader, 0)

    rewards = []
    accuracies = []

    for t in range(100):
        accuracy = train_model(discriminator, dataloader)
        accuracies.append(accuracy)

        env.decoder = discriminator

        dataloader, agent, reward = run_rl_loop(env, agent)
        rewards.append(reward)
        Hyperparameters.NUM_RL_STEPS = 5_000
        plot_trajectories(dataloader, t + 1)
        if len(rewards) % 5 == 0 and rewards:
            plt.plot(rewards)
            plt.show()


if __name__ == "__main__":
    main()
