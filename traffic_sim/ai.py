"""Deep-Q learning controller used by the simulator."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


class TrafficAIAgent:
    def __init__(self, state_size: int = Config.STATE_SIZE, action_size: int = Config.ACTION_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.gamma = Config.GAMMA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.training_steps = 0
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, emergency_mode=False):
        if emergency_mode or np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.model.eval()
            q_values = self.model(state_tensor)
            self.model.train()
        return int(np.argmax(q_values.cpu().numpy()[0]))

    def replay(self, batch_size=Config.BATCH_SIZE):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor(np.array([e[0] for e in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in minibatch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in minibatch])).to(self.device)
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (self.gamma * next_q * ~dones)
        loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.training_steps += 1
        if self.training_steps % 100 == 0:
            self.update_target_model()


