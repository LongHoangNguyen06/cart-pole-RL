import numpy as np
from collections import deque
import torch

class Buffer:
    def __init__(self, params: dict) -> None:
        self.buffer = deque()
        self.size = params["BUFFER_SIZE"]
        self.batch_size = params["BATCH_SIZE"]
        self.params = params

    def append(self, current_state: np.ndarray, 
                    action: int, 
                    next_state: np.ndarray, 
                    current_reward: float) -> None:
        """Append new sample to buffer.

        Args:
            next_state (np.ndarray): State obtained after submitting action.
            current_reward (float): Reward obtained after submitting action.
        """
        self.buffer.append((current_state, action, next_state, current_reward))
        if len(self.buffer) >= self.size:
            self.buffer.popleft()
    
    def get_random_batch(self) -> None:
        """Generate random data batch

        Returns:
            torch.tensor: 4 tensors correspond to: current observations, action used, next observation and yielded reward.
        """
        obs, action, next_obs, reward = [], [], [], []
        # Collect random samples
        for i in  np.random.randint(0, len(self.buffer), size=self.batch_size):
            obs.append(self.buffer[i][0])
            action.append(self.buffer[i][1])
            next_obs.append(self.buffer[i][2])
            reward.append(self.buffer[i][3])
        
        # Convert to tensor and return
        obs, next_obs, reward = np.array(obs), np.array(next_obs), np.array(reward)
        return torch.Tensor(obs).to(self.params["DEVICE"]), \
            torch.Tensor(action).to(self.params["DEVICE"]).long(), \
            torch.Tensor(next_obs).to(self.params["DEVICE"]), \
            torch.Tensor(reward).to(self.params["DEVICE"])