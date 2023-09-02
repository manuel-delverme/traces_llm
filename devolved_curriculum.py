import collections

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Hyperparameters:
    NUM_CLASSES = 3
    TRACE_LEN = 2
    TRACE_DIM = 2
    BATCH_SIZE = 32
    EPOCHS = 20
    SL_LEARNING_RATE = 0.001
    RL_LEARNING_RATE = 0.001
    NUM_RL_EPISODES = 5000


def samples_to_dataloader(traces: np.ndarray, labels: np.ndarray) -> DataLoader:
    traces, labels = torch.FloatTensor(traces), torch.LongTensor(labels)
    dataset = TensorDataset(traces, labels)
    return DataLoader(dataset, batch_size=Hyperparameters.BATCH_SIZE, shuffle=True)


class TraceGenerator:
    def __init__(self, num_samples=1000, noise_level=0.05):
        self.num_samples = num_samples
        self.noise_level = noise_level

    def generate(self):
        traces, labels = [], []
        for _ in range(self.num_samples // Hyperparameters.NUM_CLASSES):
            noise = np.random.normal(0, self.noise_level, (2, 2))
            traces.extend([
                np.array([[0, 0], [0, 1]]) + noise,
                np.array([[0, 0], [1, 0]]) + noise,
                np.array([[0, 0], [1, 1]]) + noise
            ])
            labels.extend([0, 1, 2])
        dataloader = samples_to_dataloader(traces, labels)
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

    for epoch in range(Hyperparameters.EPOCHS):
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{Hyperparameters.EPOCHS}, Loss: {loss.item():.4f}")


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.policy = nn.ModuleList([nn.Sequential(
            nn.Linear(Hyperparameters.TRACE_DIM * Hyperparameters.TRACE_LEN, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )] * Hyperparameters.NUM_CLASSES)

    def forward(self, goal_idx, state):
        assert goal_idx.shape == (1,)
        assert state.shape == (1, Hyperparameters.TRACE_LEN, Hyperparameters.TRACE_DIM)
        return self.policy[goal_idx](state.view(state.size(0), -1))


class CustomHandwritingEnv:
    def __init__(self, decoder: nn.Module):
        self.action_space_shape = (2,)
        self.decoder = decoder
        self.target_label = None
        self.points = None
        self.episode_length = None
        self.reset()

    def reset(self):
        self.target_label = np.random.randint(0, Hyperparameters.NUM_CLASSES)
        self.points = collections.deque([np.array([0.0, 0.0])],
                                        maxlen=Hyperparameters.TRACE_LEN)  # 1. Ensure float type
        self.episode_length = 0  # Reset episode length
        return self.get_state()

    def get_state(self):
        # Last 5 points, zero-padded if fewer
        state = np.zeros((Hyperparameters.TRACE_LEN, Hyperparameters.TRACE_DIM))
        if len(self.points) > 0:
            state[-len(self.points):] = np.array(self.points)
        return state

    def step(self, action):
        assert action.shape == self.action_space_shape
        new_point = self.points[-1] + action
        self.points.append(new_point)
        self.episode_length += 1

        # Decode the trace using the trained decoder
        with torch.no_grad():
            trace_tensor = torch.tensor(self.points, dtype=torch.float32).unsqueeze(0)
            output = self.decoder(trace_tensor)
            pred_label = torch.argmax(output, dim=1).item()

        reward = -1  # Penalty for each step
        done = False
        if pred_label == self.target_label:
            reward = 10  # Reward for correct decoding
            done = True  # End episode if correct decoding

        if self.episode_length >= Hyperparameters.TRACE_LEN:
            done = True

        return self.get_state(), reward, done


def run_rl_loop(discriminator: nn.Module) -> DataLoader:
    env = CustomHandwritingEnv(discriminator)
    agent = Controller()
    optimizer = optim.Adam(agent.parameters(), lr=Hyperparameters.RL_LEARNING_RATE)
    reward_history = []
    for episode in range(Hyperparameters.NUM_RL_EPISODES):
        state = env.reset()
        done = False
        rewards = []
        taken_action_logprobs = []
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            goal_idx = torch.tensor(env.target_label, dtype=torch.long).unsqueeze(0)

            action_logprobs = agent(goal_idx, state_tensor).squeeze(0)
            action_distr_param = torch.softmax(action_logprobs.detach(), dim=1)
            action = torch.multinomial(action_distr_param, num_samples=1).squeeze(1)
            action_logprob = action_logprobs.gather(1, action.unsqueeze(1)).squeeze(1)

            new_state, reward, done = env.step(action.numpy())

            rewards.append(reward)
            taken_action_logprobs.append(action_logprob)

            state = new_state

        rewards = torch.tensor(rewards, dtype=torch.float32)

        returns = []
        return_ = 0
        for r in reversed(rewards):  # Calculate returns for each step
            return_ = r + 0.99 * return_
            returns.insert(0, return_)  # Insert at the beginning of the list

        returns = torch.tensor(returns, dtype=torch.float32)

        optimizer.zero_grad()

        loss = -torch.sum(returns * torch.stack(taken_action_logprobs))
        loss.backward()
        optimizer.step()

        reward_history.append(sum(rewards))

        if episode % Hyperparameters.NUM_RL_EPISODES // 10 == 0:
            plot_reward_history(episode, reward_history, rewards)

    traces, labels = deterministic_rollout(env, agent)
    return samples_to_dataloader(traces, labels)


def deterministic_rollout(env, agent):
    traces, labels = [], []
    for _ in range(100):
        state = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_logprobs = agent(state_tensor).squeeze(0)
            action = torch.argmax(action_logprobs, dim=1)
            new_state, reward, done = env.step(action.numpy())
            traces.append(state)
            labels.append(env.target_label)
            state = new_state
    return traces, labels


def plot_reward_history(episode, reward_history, rewards):
    print(f"Episode {episode}, Total Reward: {sum(rewards)}")
    window_size = len(reward_history) // 10
    smoothed_reward_history = np.convolve(reward_history, np.ones(window_size) / window_size, mode='valid')
    plt.plot(smoothed_reward_history)
    plt.ylabel('Total rewards')
    plt.xlabel('Episodes')
    plt.show()


def main():
    trace_gen = TraceGenerator()
    dataloader = trace_gen.generate()
    discriminator = Decoder()

    for _ in range(100):
        train_model(discriminator, dataloader)
        dataloader = run_rl_loop(discriminator)


if __name__ == "__main__":
    main()
