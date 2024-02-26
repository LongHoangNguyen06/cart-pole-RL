import gymnasium as gym
import torch
from env_wrapper import EnvWrapper
import network
from buffer import Buffer
from pathlib import Path
import wandb
    
Path("train").mkdir(parents=True, exist_ok=True)

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
    observation, action, next_observation, rewards = buff.get_random_batch()
    next_state_best_rewards = dup_net(next_observation).max(dim=-1, keepdims=True)[0]
    current_state_actual_rewards = rewards.view(-1, 1) + params["GAMMA"] * next_state_best_rewards # Bellman-Equation
    current_state_pred = net(observation)[:, action]
    
    loss = ((current_state_actual_rewards - current_state_pred)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss

def train(params: dict):
    """Train model

    Args:
        params (dict): parameters to train model
    """
    wandb.init(project="Cart Pole RL")

    # Initialize network and RL loop
    net = network.Network(params=params)
    action_inferrer = network.ActionInferrer(net=net, params=params)
    dup_net = network.duplicate(net=net)
    buff = Buffer(params=params)
    opt = torch.optim.Adam(params=net.parameters(), lr=params["LR"])
    env = EnvWrapper(env=gym.make('CartPole-v1', render_mode=params["MODE"]), params=params)
    
    # Initialize variables
    seed = 0
    observation, _ = env.reset(seed=seed)
    episode_reward = 0

    # Training loop
    for epoch in range(params["TRAINING_EPOCHS"]):
        env.render()
        
        # Training part
        loss = None
        if epoch > params["BATCH_SIZE"]:
            loss = train_iteration(net=net, dup_net=dup_net, opt=opt, buff=buff, params=params)
        
        # Get action to fill buffer
        action = action_inferrer.get_train_action(observation)

        # Simulate environment once and insert next observation to buffer
        for _ in range(params["FRAME_SKIP"]):
            next_observation, reward, terminated, _, _ = env.step(action)
            if terminated: break
        
        # Episode reward
        episode_reward += reward
        buff.append(current_state=observation, 
                    action=action, 
                    next_observation=next_observation, 
                    reward=reward)

        # Duplicate the network once for a while and fix it
        if epoch % params["DUP_FREQ"] == 0:
            dup_net = network.duplicate(net=net)
        
        # Debugging part
        wandb.log({"episode_reward": episode_reward})
        wandb.log({"positions": next_observation[0]})
        wandb.log({"reward": reward})
        wandb.log({"velocity": next_observation[1]})
        wandb.log({"angle": next_observation[2]})
        wandb.log({"angular_velocity": next_observation[3]})
        if loss: wandb.log({"loss": loss})

        # Reset to new map if terminated
        if terminated:
            seed += 1
            next_observation, _ = env.reset(seed=seed)  # Reset the environment if the episode is over
            episode_reward = 0

        # Set next observation as current one
        observation = next_observation
    wandb.finish()
