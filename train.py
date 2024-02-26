import gymnasium as gym
import torch
from env_wrapper import EnvWrapper
import network
from buffer import Buffer
from pathlib import Path
import wandb
from tqdm import tqdm

Path("train").mkdir(parents=True, exist_ok=True)

def train_iteration(net: network.Network, 
                    dup_net: network.Network, 
                    opt: torch.optim.Optimizer,
                    buff: Buffer, 
                    params: dict) -> float:
    """One training loop of Deep-Q-Learning

    Args:
        net (network.Network): Network.
        dup_net (network.Network): Target network.
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
    current_state_actual_rewards = reward + params["GAMMA"] * next_state_best_rewards # Bellman-Equation
    
    # Compute actual
    current_state_pred = net(observation)[:, action]
    
    # Loss and backprop
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
    wandb.init(project="Cart Pole RL", name=params["EXPERIMENT_NAME"])

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
    for epoch in tqdm(range(params["TRAINING_EPOCHS"])):
        env.render()
        
        # Training part
        loss = None
        if epoch > params["BATCH_SIZE"]:
            loss = train_iteration(net=net, dup_net=dup_net, opt=opt, buff=buff, params=params)
        
        # Get action to fill buffer
        net.eval()
        action = action_inferrer.get_train_action(observation.reshape(1, -1))

        # Simulate environment once and insert next observation to buffer
        for _ in range(params["FRAME_SKIP"]):
            next_observation, reward, terminated, truncated, _ = env.step(action)
            if terminated: break
        
        # Episode reward
        episode_reward += reward
        buff.append(observation=observation, 
                    action=action, 
                    next_observation=next_observation, 
                    reward=reward,
                    terminated=terminated)

        # Duplicate the network once for a while and fix it
        if epoch % params["DUP_FREQ"] == 0:
            dup_net = network.duplicate(net=net)
        
        # Debugging part
        wandb.log({"metric/greedy_epsilon": action_inferrer.get_epsilon()})
        wandb.log({"metric/reward": reward})
        if loss: wandb.log({"metric/loss": float(loss)})

        wandb.log({"input/positions": float(next_observation[0])})
        wandb.log({"input/velocity": float(next_observation[1])})
        wandb.log({"input/angle": float(next_observation[2])})
        wandb.log({"input/angular_velocity": float(next_observation[3])})
        
        wandb.log({"weight/mean_weight": net.mean_weight()})
        wandb.log({"weight/std_weight": net.std_weight()})
        wandb.log({"weight/mean_grad": net.mean_grad()})
        wandb.log({"weight/std_grad": net.std_grad()})
        
        wandb.log({"activation/mean_layer1": torch.mean(net.x1)})
        wandb.log({"activation/std_layer1": torch.std(net.x1)})
        wandb.log({"activation/mean_layer2": torch.mean(net.x2)})
        wandb.log({"activation/std_layer2": torch.std(net.x2)})
        wandb.log({"activation/mean_layer3": torch.mean(net.x3)})
        wandb.log({"activation/std_layer3": torch.std(net.x3)})
        wandb.log({"activation/action": int(action)})
        net.x1 = torch.zeros_like(net.x1)
        net.x2 = torch.zeros_like(net.x2)
        net.x3 = torch.zeros_like(net.x3)

        # Reset to new map if terminated
        if terminated:
            seed += 1
            next_observation, _ = env.reset(seed=seed)  # Reset the environment if the episode is over
            wandb.log({"training/episode_reward": episode_reward})
            episode_reward = 0

        # Set next observation as current one
        observation = next_observation
    wandb.finish()
