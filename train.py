import gymnasium as gym
import random
import numpy as np
import torch
import network
from buffer import Buffer
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use('TKAgg')

Path("train").mkdir(parents=True, exist_ok=True)

def apply_random_seed(random_seed: int) -> None:
    """Sets seed to ``random_seed`` in random, numpy and torch."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_data(obs: torch.Tensor, params: dict):
    """Preprocess observation. 
    Min and max values found at https://gymnasium.farama.org/environments/classic_control/cart_pole/.

    Args:
        obs (torch.Tensor): _description_
        params (dict): _description_

    Returns:
        _type_: _description_
    """
    obs[:, 0] /=  params["MAX_CART_POSITION"]

    obs[:, 1] = torch.clip(obs[:, 1], min=-params["CLIP_CART_VELOCITY"], 
                           max=params["CLIP_CART_VELOCITY"])
    obs[:, 1] /= params["CLIP_CART_VELOCITY"]

    obs[:, 2] /= params["MAX_CART_POLE_ANGLE"]

    obs[:, 3] = torch.clip(obs[:, 3],
                                   min=-params["MAX_CART_POLE_VELOCITY"],
                                   max=params["MAX_CART_POLE_VELOCITY"])
    obs[:, 3] /= params["MAX_CART_POLE_VELOCITY"]
    return obs
    

def train_iteration(net: network.Network, 
                    dup_net: network.Network, 
                    opt: torch.optim.Optimizer,
                    buff: Buffer, 
                    params: dict) -> float:
    """One training loop of Deep-Q-Learning

    Args:
        net (network.Network): _description_
        dup_net (network.Network): _description_
        opt (torch.optim.Optimizer): _description_
        buff (Buffer): _description_
        params (dict): _description_

    Returns:
        float: _description_
    """
    obs, action, next_obs, rewards = buff.get_random_batch()
    obs = preprocess_data(obs=obs, params=params)
    next_obs = preprocess_data(obs=next_obs, params=params)

    next_state_best_rewards = dup_net(next_obs).max(dim=-1, keepdims=True)[0]
    current_state_actual_rewards = rewards.view(-1, 1) + params["GAMMA"] * next_state_best_rewards # Bellman-Equation
    current_state_pred = net(obs)[:, action]
    
    loss = ((current_state_actual_rewards - current_state_pred)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

def train(params: dict):
    # Initialize network and RL loop
    net = network.Network(params=params)
    action_inferrer = network.ActionInferrer(net=net, params=params)
    dup_net = network.duplicate(net=net)
    buff = Buffer(params=params)
    opt = torch.optim.Adam(params=net.parameters(), lr=params["LR"])
    env = gym.make('CartPole-v1', render_mode=params["MODE"])
    apply_random_seed(params["RANDOM_SEED"])
    
    # Initialize variables for debugging
    seed = 0
    obs, _ = env.reset(seed=seed)
    losses = []
    rewards = []
    reward = 0

    # Training loop
    for epoch in range(params["TRAINING_EPOCHS"]):
        env.render()
        
        # Training part
        if epoch >= params["BATCH_SIZE"]:
            loss = train_iteration(net=net, dup_net=dup_net, opt=opt, buff=buff, params=params)
            losses.append(loss.item())
        
        # Get action to fill buffer
        action = action_inferrer.get_train_action(x=torch.Tensor(obs).to(params["DEVICE"]))

        # Simulate environment once and insert next state to buffer
        next_obs, re, done, _, _ = env.step(action)
        reward += re
        buff.append(current_state=obs, action=action, next_state=next_obs, current_reward=re)

        # Duplicate the network once for a while and fix it
        if epoch % params["DUP_FREQ"] == 0:
            dup_net = network.duplicate(net=net)

        # Reset to new map if done
        if done:
            rewards.append(reward)
            reward = 0
            seed += 1
            next_obs, _ = env.reset(seed=seed)  # Reset the environment if the episode is over
        
        # Debugging part
        if done:
            plt.figure()
            plt.plot(losses)
            plt.savefig("train/losses.png")

            plt.figure()
            plt.plot(rewards)
            plt.savefig("train/rewards.png")

        # Set next observation as current one
        obs = next_obs