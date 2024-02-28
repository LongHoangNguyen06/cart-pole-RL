import traceback
import gymnasium as gym
import numpy as np
import torch
import wandb
from tqdm import tqdm
import json
from typing import Tuple
import numpy as np
import torch
from torchrl.data import ReplayBuffer, ListStorage
    
class Buffer:
    def __init__(self, params: dict) -> None:
        max_size = int(params["BUFFER_SIZE"] * params["TRAINING_EPISODES"] * 250)
        self.buffer = ReplayBuffer(storage=ListStorage(max_size=max_size),
                                   batch_size=params["BATCH_SIZE"])
        self.params = params
    
    def __len__(self):
        return len(self.buffer)

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

class Network(torch.nn.Module):
    """Policy network's MLP architecture"""
    def __init__(self, params):        
        super().__init__()
        self.layers = []
        for i in range(len(params["ARCHITECTURE"]) - 2):
            self.layers.append(torch.nn.Sequential(
                    torch.nn.Linear(in_features=params["ARCHITECTURE"][i], out_features=params["ARCHITECTURE"][i+1], bias=True, device=params["DEVICE"]),
                    torch.nn.LeakyReLU(),
                    torch.nn.BatchNorm1d(num_features=params["ARCHITECTURE"][i+1], device=params["DEVICE"]),
                    torch.nn.Dropout(p=params["DROPOUT_P"])
                )
            )
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Linear(in_features=params["ARCHITECTURE"][-2], out_features=params["ARCHITECTURE"][-1], bias=True, device=params["DEVICE"])
            )
        ]
        self.activations = [torch.zeros(1, s) for s in params["ARCHITECTURE"]]
        self.params = params
        self.net = torch.nn.Sequential(*self.layers)
    
    def forward(self, x):
        self.activations[0] = x
        for i, layer in enumerate(self.layers):
            self.activations[i + 1] = layer(self.activations[i])
        return self.activations[-1]
    
def mean_weight(layer):
    """Mean of weight.

    Args:
        layer (torch.nn.Sequential): A single layer.

    Returns:
        float: output.
    """
    weights = []
    for param in layer.parameters():
        weights.append(param.data.view(-1))
    all_weights = torch.cat(weights)
    return torch.mean(all_weights)

def std_weight(layer):
    """Std of weight.

    Args:
        layer (torch.nn.Sequential): A single layer.

    Returns:
        float: output.
    """
    weights = []
    for param in layer.parameters():
        weights.append(param.data.view(-1))
    all_weights = torch.cat(weights)
    return torch.std(all_weights)

def mean_grad(layer):
    """Mean of grad.

    Args:
        layer (torch.nn.Sequential): A single layer.

    Returns:
        float: output.
    """
    grads = []
    for param in layer.parameters():
        if param.grad is not None:
            grads.append(param.grad.data.view(-1))
    if len(grads) > 0:
        all_grads = torch.cat(grads)
        return torch.mean(all_grads)
    else:
        return 0

def std_grad(layer):
    """Std of grad.

    Args:
        layer (torch.nn.Sequential): A single layer.

    Returns:
        float: output.
    """
    grads = []
    for param in layer.parameters():
        if param.grad is not None:
            grads.append(param.grad.data.view(-1))
    if len(grads) > 0:
        all_grads = torch.cat(grads)
        return torch.std(all_grads)
    else:
        return 0

def train_iteration(net: Network, 
                    dup_net: Network, 
                    opt: torch.optim.Optimizer,
                    buff: Buffer, 
                    params: dict) -> float:
    """One training loop of Deep-Q-Learning

    Args:
        net (Network): 
        dup_net (Network): Target 
        opt (torch.optim.Optimizer): Optimizer.
        buff (Buffer): Replay buffer.
        params (dict): Training parameters

    Returns:
        float: loss of the batch
    """
    net.train()
    # Sampling random states
    observation, action, next_observation, reward, terminated = buff.get_random_batch()
    
    # Compute expected
    next_state_best_rewards = dup_net(next_observation).max(dim=-1)[0] * (1 - terminated)
    state_action_expected_reward = reward + params["GAMMA"] * next_state_best_rewards # Bellman-Equation
    
    # Compute actual
    all_state_pred = net(observation)
    state_action_reward_pred = all_state_pred[torch.arange(len(all_state_pred)), action]
    
    # Loss and backprop
    loss = ((state_action_expected_reward - state_action_reward_pred)**2).mean()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    opt.step()
    return loss

def duplicate(net: Network) -> Network:
    """
    Duplicate target network.
    """
    copied_net = Network(params=net.params)
    copied_net.load_state_dict(net.state_dict())
    return copied_net

class ActionInferrer:
    """Class to help output decision of network.

    In training time, the network randomly output either the its believed-to-be best decision
    or a random action.

    At inference time, it shall output its faithfully believed-to-be best decision.
    """
    def __init__(self, net: Network, params: dict) -> None:
        self.net = net
        self.params = params
        self.epoch = 0

    def get_epsilon(self):
        last_greedy_epoch = self.params["FINAL_GREEDY_EPSILON_EPOCH"] * self.params["TRAINING_EPOCHS"]
        return np.clip((1 - self.epoch / last_greedy_epoch), self.params["FINAL_GREEDY_EPSILON"], 1.0)

    def get_train_action(self, x: np.ndarray):
        if np.random.rand() < self.get_epsilon():
            action = np.random.choice(self.params["ACTIONS"])
        else:
            action = self.get_best_action(x)
        self.epoch += 1
        return action
    
    def get_best_action(self, x: np.ndarray):
        x = torch.Tensor(x).to(self.params["DEVICE"])
        return self.params["ACTIONS"][self.net(x).argmax()]

def train(params: dict):
    """Train model

    Args:
        params (dict): parameters to train model
    """
    # Initialize network and RL loop
    net = Network(params=params)
    action_inferrer = ActionInferrer(net=net, params=params)
    dup_net = duplicate(net=net)
    buff = Buffer(params=params)
    opt = torch.optim.RMSprop(params=net.parameters(), lr=params["LR"])
    env = gym.make('CartPole-v1', render_mode=params["MODE"])
    params["MODEL_PARAMETERS"] = str(sum(p.numel() for p in net.parameters() if p.requires_grad))
    
    # Debug
    print("#"*100)
    for key, value in params.items():
        key = key.ljust(max(len(k) for k in params)) 
        print(key, value)
    print("#"*100)

    # Initialize variables
    hyperopt_performance = 0
    epoch = 0
    episode_rewards = []
    # Training loop
    for episode in tqdm(range(params["TRAINING_EPISODES"])):
        env.render()
        
        # Training part
        observation, _ = env.reset(seed=params["RANDOM_SEED"])
        episode_reward = 0
        terminated = False
        while not terminated:
            epoch += 1
            if len(buff) > params["BATCH_SIZE"]:
                train_iteration(net=net, dup_net=dup_net, opt=opt, buff=buff, params=params)
        
            # Get action to fill buffer
            net.eval()
            action = action_inferrer.get_train_action(observation.reshape(1, -1))

            # Simulate environment once and insert next observation to buffer
            reward = 0
            for _ in range(params["FRAME_SKIP"]):
                next_observation, re, terminated, truncated, _ = env.step(action)
                reward += re
                episode_reward += re
                if terminated or truncated: break
            
            # Episode reward
            buff.append(observation=observation, 
                        action=action, 
                        next_observation=next_observation, 
                        reward=reward,
                        terminated=terminated)

            # Duplicate the network once for a while and fix it
            if epoch % params["DUP_FREQ"] == 0: dup_net = duplicate(net=net)
            
            # Reset to new map if terminated
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_reward = 0

                # Debugging
                if len(episode_rewards) > params["SCORING_WINDOW_SIZE"]:
                    hyperopt_performance = max(hyperopt_performance, np.mean(episode_rewards[-params["SCORING_WINDOW_SIZE"]:]))
                    wandb.log({"metric/performance": hyperopt_performance}, step=episode)
                wandb.log({"metric/greedy_epsilon": action_inferrer.get_epsilon()},step=episode)
            observation = next_observation

        observation, _ = env.reset()
    wandb.finish()
    env.close()

def normal_train(params: dict):
    """Manual training loop

    Args:
        params (dict): Training hyper parameters.
    """
    wandb.init(project="Cart Pole RL", name=params["EXPERIMENT_NAME"])
    train(params=params)

def hyperopt(device: str, mode: str, sweep_id = None):
    """Bayesian hyperparameter-optimization loop.

    Args:
        device (str): device to train on.
        mode (str): rgb_array or human mode.
        sweep_id (_type_, optional): wandb's sweep id to do optimization in batch.. Defaults to None.

    Raises:
        e: _description_
    """
    session_counter = 0
    def hyperopt_training_loop(config=None):
        """Inner loop."""
        nonlocal session_counter
        session_counter += 1
        try:    
            import time
            with wandb.init(config = config, name=f"session_{session_counter}_{round(time.time())}"):
                params = wandb.config
                params["DEVICE"] = device
                params["MODE"] = mode
                if params["NUMER_HIDDEN_LAYERS"] == 1:
                    params["ARCHITECTURE"] = [4, params["HIDDEN_LAYER_1"], params["HIDDEN_LAYER_2"], 2]
                elif params["NUMER_HIDDEN_LAYERS"] == 2 and params["HIDDEN_LAYER_2"] == 0:
                    params["ARCHITECTURE"] = [4, params["HIDDEN_LAYER_1"], 2]
                elif params["NUMER_HIDDEN_LAYERS"] == 2 and params["HIDDEN_LAYER_2"] != 0:
                    wandb.log({"metric/performance": 0})
                    wandb.finish()
                    return
                params = dict(params)
                train(params)
        except Exception as e:
            print(traceback.format_exc())
            raise e
    
    if sweep_id is None:
        with open("config/hyperopt_search_space.json", "r") as f:
            config = json.load(f)
            sweep_id = wandb.sweep(config, project='Cart Pole RL')
    wandb.agent(sweep_id, hyperopt_training_loop, count=10000, project='Cart Pole RL')