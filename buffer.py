from typing import Tuple
import numpy as np
import torch
from torchrl.data import ReplayBuffer, ListStorage

class Buffer:
    def __init__(self, params: dict) -> None:
        max_size = int(params["BUFFER_SIZE"] * params["TRAINING_EPOCHS"])
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=max_size),
                                   batch_size=params["BATCH_SIZE"])
        self.params = params

    def append(self, observation: np.ndarray, 
                    action: int, 
                    next_observation: np.ndarray, 
                    reward: float,
                    terminated: bool) -> None:
        """Append new sample to buffer.

        Args:
            observation (np.ndarray): Input state s_t.
            action (float): Input action a_t.
            next_observation (np.ndarray): State s_{t + 1} obtained after submitting action.
            reward (float): Reward r_t obtained after submitting action.
            terminated (bool): Determine if after observation + action the game end
        """
        self.buffer.extend([{
            "observation": observation, 
            "action": action, 
            "next_observation": next_observation, 
            "reward": reward, 
            "terminated": terminated
        }])
    
    def get_random_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate random data batch

        Returns:
            torch.tensor: 4 tensors correspond to: 
                observation
                action
                next_observation
                reward
        """
        observation, action, next_observation, reward, terminated = [], [], [], [], []
        for i in  self.buffer.sample():
            observation.append(i["observation"])
            action.append(i["action"])
            next_observation.append(i["next_observation"])
            reward.append(i["reward"])
            terminated.append(i["terminated"])
        return torch.cat(observation, 0).to(self.params["DEVICE"]), \
            torch.cat(action, 0).to(self.params["DEVICE"]).long(), \
            torch.cat(next_observation, 0).to(self.params["DEVICE"]), \
            torch.cat(reward, 0).to(self.params["DEVICE"]), \
            torch.cat(terminated, 0).to(self.params["DEVICE"]).int()
