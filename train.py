import gymnasium as gym
import torch
from env_wrapper import EnvWrapper
import network
from buffer import Buffer
import wandb
from tqdm import tqdm

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
    state_action_expected_reward = reward + params["GAMMA"] * next_state_best_rewards # Bellman-Equation
    
    # Compute actual
    all_state_pred = net(observation)
    state_action_reward_pred = all_state_pred[torch.arange(len(all_state_pred)), action]
    
    # Loss and backprop
    loss = ((state_action_expected_reward - state_action_reward_pred)**2).mean()
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
    observation, _ = env.reset(seed=params["RANDOM_SEED"])
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
            if terminated or truncated: break
        
        # Episode reward
        episode_reward += reward
        buff.append(observation=observation, 
                    action=action, 
                    next_observation=next_observation, 
                    reward=reward,
                    terminated=terminated)

        # Duplicate the network once for a while and fix it
        if epoch % params["DUP_FREQ"] == 0: dup_net = network.duplicate(net=net)
        
        # Debugging part
        if True:
            wandb.log({"metric/greedy_epsilon": action_inferrer.get_epsilon()},step=epoch)
            wandb.log({"metric/reward": reward},step=epoch)
            if loss: wandb.log({"metric/loss": float(loss)},step=epoch)

            wandb.log({"input/positions": float(next_observation[0])},step=epoch)
            wandb.log({"input/velocity": float(next_observation[1])},step=epoch)
            wandb.log({"input/angle": float(next_observation[2])},step=epoch)
            wandb.log({"input/angular_velocity": float(next_observation[3])},step=epoch)
            
            for i, layer in enumerate(net.layers):
                wandb.log({f"layer{i+1}/mean_activation": torch.mean(net.activations[i+1])},step=epoch)
                wandb.log({f"layer{i+1}/std_activation": torch.std(net.activations[i+1])},step=epoch)
                wandb.log({f"layer{i+1}/mean_weight": net.mean_weight(layer)},step=epoch)
                wandb.log({f"layer{i+1}/std_weight": net.std_weight(layer)},step=epoch)
                wandb.log({f"layer{i+1}/mean_grad": net.mean_grad(layer)},step=epoch)
                wandb.log({f"layer{i+1}/std_grad": net.std_grad(layer)},step=epoch)

        # Reset to new map if terminated
        if terminated or truncated:
            next_observation, _ = env.reset()  # Reset the environment if the episode is over
            wandb.log({"metric/episode_reward": episode_reward})
            episode_reward = 0

        # Set next observation as current one
        observation = next_observation
    wandb.finish()
