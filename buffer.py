import numpy as np
from collections import deque
import torch

class Buffer:
    def __init__(self, params: dict, args: dict) -> None:
        self.buffer = deque()
        self.size = params["BUFFER_SIZE"]
        self.batch_size = params["BATCH_SIZE"]
        self.params = params
        self.args = args

    def append(self, obs: np.ndarray, reward: float) -> None:
        self.buffer.append((obs, reward))
        if len(self.buffer) >= self.size:
            self.buffer.popleft()
    
    def get_random_batch(self) -> None:
        indices = np.random.randint(0, np.floor(len(self.buffer) / 2 - 1), size=self.batch_size)
        obs, next_obs, reward = [], [], [] 
        for i in indices:
            obs.append(self.buffer[2 * i][0])
            next_obs.append(self.buffer[2 * i + 1][0])
            reward.append(self.buffer[2 * i][1])
        obs, next_obs, reward = np.array(obs), np.array(next_obs), np.array(reward)
        return torch.Tensor(obs).to(self.args.device), \
            torch.Tensor(next_obs).to(self.args.device), \
            torch.Tensor(reward).to(self.args.device)