import gymnasium as gym
import random
import numpy as np
import torch
import network
from buffer import Buffer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')

def apply_random_seed(random_seed: int) -> None:
    """Sets seed to ``random_seed`` in random, numpy and torch."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_data(obs: torch.Tensor, 
                    params: dict):
    obs[:, 0] /=  params["MAX_CART_POSITION"]

    obs[:, 1] = torch.clip(obs[:, 1],
                                   min=-params["CLIP_CART_VELOCITY"],
                                   max=params["CLIP_CART_VELOCITY"])
    obs[:, 1] /= params["CLIP_CART_VELOCITY"]

    obs[:, 2] /= params["MAX_CART_POLE_ANGLE"]

    obs[:, 3] = torch.clip(obs[:, 3],
                                   min=-params["MAX_CART_POLE_VELOCITY"],
                                   max=params["MAX_CART_POLE_VELOCITY"])
    obs[:, 3] /= params["MAX_CART_POLE_VELOCITY"]
    return obs
    

def train_iteration(net: network.Network, 
                    dupnet: network.Network, 
                    opt: torch.optim.Optimizer,
                    buff: Buffer, 
                    params: dict) -> float:
    """One training loop of Deep-Q-Learning

    Args:
        net (network.Network): _description_
        dupnet (network.Network): _description_
        opt (torch.optim.Optimizer): _description_
        buff (Buffer): _description_
        params (dict): _description_

    Returns:
        float: _description_
    """
    obs, next_obs, rewards = buff.get_random_batch()
    obs = preprocess_data(obs=obs, params=params)
    next_obs = preprocess_data(obs=next_obs, params=params)

    next_state_best_rewards = dupnet(next_obs).max(dim=-1, keepdims=True)[0]
    current_state_actual_rewards = rewards.view(-1, 1) + params["GAMMA"] * next_state_best_rewards # Bellman-Equation
    current_state_pred = net(obs)
    
    loss = ((current_state_actual_rewards - current_state_pred)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

def train(params: dict, architecture: dict, args: dict):
    net = network.Network(architecture=architecture, params=params, device=args.device)
    dupnet = network.duplicate(net=net)
    buff = Buffer(params=params, args=args)
    opt = torch.optim.Adam(params=net.parameters(), lr=params["LR"])
    env = gym.make('CartPole-v1', render_mode=args.mode)
    env.action_space.seed(params["ACTION_SPACE_SEED"])
    apply_random_seed(params["RANDOM_SEED"])
    
    map_id = 0
    obs, _ = env.reset(seed=map_id)
    losses = []

    for epoch in range(params["TRAINING_EPOCHS"]):
        env.render()
        if epoch >= params["BATCH_SIZE"]:
            loss = train_iteration(net=net, dupnet=dupnet, opt=opt, buff=buff, params=params)
            losses.append(loss.item())
        
        action = net.get_action(x=torch.Tensor(obs).to(args.device), train=True)
        obs, reward, done, _, _ = env.step(action)
        buff.append(obs=obs, reward=reward)
        if epoch % params["DUP_FREQ"] == 0:
            dupnet = network.duplicate(net=net)
        if done:
            obs, _ = env.reset()  # Reset the environment if the episode is over
    plt.plot(losses)
    plt.show()