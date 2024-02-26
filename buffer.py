from typing import Tuple
import numpy as np
import torch
from torchrl.data import ReplayBuffer, ListStorage

class Buffer:
    def __init__(self, params: dict) -> None:
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=params["BUFFER_SIZE"]),
                                   batch_size=params["BATCH_SIZE"])

    def append(self, current_state: np.ndarray, 
                    action: int, 
                    next_state: np.ndarray, 
                    current_reward: float) -> None:
        """Append new sample to buffer.

        Args:
            next_state (np.ndarray): State obtained after submitting action.
            current_reward (float): Reward obtained after submitting action.
        """
        self.buffer.append({
            "current_state": current_state, 
            "action": action, 
            "next_state": next_state, 
            "current_reward": current_reward
        })
    
    def get_random_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random data batch

        Returns:
            torch.tensor: 4 tensors correspond to: 
                current state
                action used
                next state
                current reward
        """
        obs, action, next_obs, reward = [], [], [], []
        for i in  self.buffer.sample():
            obs.append(i["current_state"])
            action.append(i["action"])
            next_obs.append(i["next_state"])
            reward.append(i["current_reward"])
        return torch.cat(obs, 0).to(self.params["DEVICE"]), \
            torch.cat(action, 0).to(self.params["DEVICE"]).long(), \
            torch.cat(next_obs, 0).to(self.params["DEVICE"]), \
            torch.cat(reward, 0).to(self.params["DEVICE"])
