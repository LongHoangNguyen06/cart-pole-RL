import gymnasium as gym
import numpy as np
from gymnasium.core import ActType

def min_max_scaling(observation: np.ndarray, a_min: float, a_max: float):
    assert a_min < a_max
    observation = np.clip(observation, a_min=a_min, a_max=a_max)
    return (observation - a_min) / (a_max - a_min) - 0.5

def preprocess_data(observation: np.ndarray, params: dict):
    """Preprocess observation. 
    Min and max values found at 
    https://gymnasium.farama.org/environments/classic_control/cart_pole/.

    Args:
        observation (torch.Tensor): Observation
        params (dict): Parameters for preprocessing

    Returns:
        np.ndarray: Preprocessed data
    """
    # Position
    observation[0] = min_max_scaling(observation[0], 
                                a_min=params["MIN_CART_POSITION"],
                                a_max=params["MAX_CART_POSITION"])
    
    # Velocity
    observation[1] = min_max_scaling(observation[1], 
                                a_min=params["MIN_CART_VELOCITY"],
                                a_max=params["MAX_CART_VELOCITY"])

    # Angle
    observation[2] = min_max_scaling(observation[2], 
                                a_min=params["MIN_CART_POLE_ANGLE"],
                                a_max=params["MAX_CART_POLE_ANGLE"])

    # Angular velocity
    observation[3] = min_max_scaling(observation[3], 
                                a_min=params["MIN_CART_POLE_ANGULAR_VELOCITY"],
                                a_max=params["MAX_CART_POLE_ANGULAR_VELOCITY"])
    return observation

def preprocess_reward(observation: np.ndarray, reward: float, terminated: bool, 
                      truncated: bool, info: dict, params: dict) -> float:
    """Preprocess reward since penalty is not included.

    Args:
        observation (np.ndarray): Observation.
        reward (float): Output reward.
        terminated (bool): Whether a terminal observation (as defined under the MDP of the task) is reached. 
        truncated (bool):  whether a truncation condition outside the scope of the MDP is satisfied. 
            Typically a timelimit, but could also be used to indicate agent physically going out of bounds. 
            Can be used to end the episode prematurely before a terminal observation is reached.
        info (dict): contains auxiliary diagnostic information (helpful for debugging, learning, and logging)
        params (dict): Training parameters

    Returns:
        float: reward or penalty
    """
    if terminated:
        if truncated:
            return params["REWARD_REWARD_SUCCESS"]
        else:
            return params["REWARD_PENALTY_FAIL"]
    return reward

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, params: dict):
        super().__init__(env)
        self.params = params

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        #observation = preprocess_data(observation=observation, params=self.params)
        return observation, info

    def step(self, action: ActType):
        observation, reward, terminated, truncated, info = super().step(action)
        #observation = preprocess_data(observation=observation, params=self.params)
        #reward = preprocess_reward(observation=observation, reward=reward, terminated=terminated, truncated=truncated, info=info, params=self.params)

        return observation, reward, terminated, truncated, info