import gymnasium as gym
import random
import numpy as np
import torch
import network
from buffer import Buffer
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import wandb

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
    wandb.init(project="Cart Pole RL")

    # Initialize network and RL loop
    net = network.Network(params=params)
    action_inferrer = network.ActionInferrer(net=net, params=params)
    dup_net = network.duplicate(net=net)
    buff = Buffer(params=params)
    opt = torch.optim.Adam(params=net.parameters(), lr=params["LR"])
    env = gym.make('CartPole-v1', render_mode=params["MODE"])
    apply_random_seed(params["RANDOM_SEED"])
    
    # Initialize variables
    seed = 0
    obs, _ = env.reset(seed=seed)
    episode_reward = 0

    # Initialize variables for debugging
    debug_losses = []
    debug_rewards = []
    debug_cart_position = []
    debug_cart_velocity = []
    debug_pole_angle = []
    debug_pole_angular_velocity = []
    debug_single_awards = []

    # Training loop
    for epoch in range(params["TRAINING_EPOCHS"]):
        env.render()
        
        # Training part
        if epoch >= params["BATCH_SIZE"]:
            loss = train_iteration(net=net, dup_net=dup_net, opt=opt, buff=buff, params=params)
            debug_losses.append(loss.item())
        
        # Get action to fill buffer
        action = action_inferrer.get_train_action(x=torch.Tensor(obs).to(params["DEVICE"]))

        # Simulate environment once and insert next state to buffer
        for _ in range(params["FRAME_SKIP"]):
            next_obs, re, done, _, _ = env.step(action)

        if done:
            if episode_reward + re < params["MAX_REWARD"]:
                re = params["REWARD_PENALTY_FAIL"]
        episode_reward += re
        buff.append(current_state=obs, action=action, next_state=next_obs, current_reward=re)

        # Duplicate the network once for a while and fix it
        if epoch % params["DUP_FREQ"] == 0:
            dup_net = network.duplicate(net=net)
        
        # Debugging part
        debug_single_awards.append(re)
        debug_cart_position.append(obs[0])
        debug_cart_velocity.append(obs[1])
        debug_pole_angle.append(obs[2])
        debug_pole_angular_velocity.append(obs[3])
        if done:
            debug_rewards.append(episode_reward)
            wandb.log({"episode_reward": episode_reward})

            fig = plt.figure()
            plt.plot(debug_losses)
            wandb.log({"losses": wandb.Image(fig)})
            plt.savefig("train/losses.png")

            fig = plt.figure()
            plt.plot(debug_rewards)
            wandb.log({"rewards": wandb.Image(fig)})
            plt.savefig("train/rewards.png")

            fig = plt.figure()
            plt.plot(debug_single_awards)
            wandb.log({"single_rewards": wandb.Image(fig)})
            plt.savefig("train/single_rewards.png")

            fig = plt.figure()
            plt.plot(debug_cart_position)
            wandb.log({"positions": wandb.Image(fig)})
            plt.savefig("train/positions.png")

            fig = plt.figure()
            plt.plot(debug_cart_velocity)
            wandb.log({"velocity": wandb.Image(fig)})
            plt.savefig("train/velocity.png")

            fig = plt.figure()
            plt.plot(debug_pole_angle)
            wandb.log({"angle": wandb.Image(fig)})
            plt.savefig("train/angle.png")

            fig = plt.figure()
            plt.plot(debug_pole_angular_velocity)
            wandb.log({"angular_velocity": wandb.Image(fig)})
            plt.savefig("train/angular_velocity.png")

        # Reset to new map if done
        if done:
            seed += 1
            next_obs, _ = env.reset(seed=seed)  # Reset the environment if the episode is over
            episode_reward = 0

        # Set next observation as current one
        obs = next_obs
    wandb.finish()
