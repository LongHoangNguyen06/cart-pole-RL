from typing import Tuple
import numpy as np
import torch
from torchrl.data import ReplayBuffer, ListStorage

class Buffer:
    def __init__(self, params: dict) -> None:
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=params["BUFFER_SIZE"]),
                                   batch_size=params["BATCH_SIZE"])
        self.params = params

    def append(self, current_observation: np.ndarray, 
                    action: int, 
                    next_observation: np.ndarray, 
                    reward: float) -> None:
        """Append new sample to buffer.

        Args:
            current_observation (np.ndarray): Input state s_t.
            action (float): Input action a_t.
            next_observation (np.ndarray): State s_{t + 1} obtained after submitting action.
            reward (float): Reward r_t obtained after submitting action.
        """
        self.buffer.append({
            "current_observation": current_observation, 
            "action": action, 
            "next_observation": next_observation, 
            "reward": reward
        })
    
    def get_random_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random data batch

        Returns:
            torch.tensor: 4 tensors correspond to: 
                current_observation
                action
                next_observation
                reward
        """
        obs, action, next_obs, reward = [], [], [], []
        for i in  self.buffer.sample():
            obs.append(i["current_observation"])
            action.append(i["action"])
            next_obs.append(i["next_observation"])
            reward.append(i["reward"])
        return torch.cat(obs, 0).to(self.params["DEVICE"]), \
            torch.cat(action, 0).to(self.params["DEVICE"]).long(), \
            torch.cat(next_obs, 0).to(self.params["DEVICE"]), \
            torch.cat(reward, 0).to(self.params["DEVICE"])
