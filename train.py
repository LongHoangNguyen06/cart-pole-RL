import traceback
import gymnasium as gym
import numpy as np
import torch
import network
from buffer import Buffer
import wandb
from tqdm import tqdm
import json

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
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
    opt.step()
    return loss

def train(params: dict):
    """Train model

    Args:
        params (dict): parameters to train model
    """
    # Initialize network and RL loop
    net = network.Network(params=params)
    action_inferrer = network.ActionInferrer(net=net, params=params)
    dup_net = network.duplicate(net=net)
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
    observation, _ = env.reset(seed=params["RANDOM_SEED"])
    episode_reward = 0
    episode_rewards = []
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

            wandb.log({'output/action': float(action)},step=epoch)
            
            # Reset to new map if terminated
            if terminated or truncated:
                next_observation, _ = env.reset()  # Reset the environment if the episode is over
                wandb.log({"metric/episode_reward": episode_reward},step=epoch)
                wandb.log({'episode_start/positions': float(next_observation[0])},step=epoch)
                wandb.log({'episode_start/velocity': float(next_observation[1])},step=epoch)
                wandb.log({'episode_start/angle': float(next_observation[2])},step=epoch)
                wandb.log({'episode_start/angular_velocity': float(next_observation[3])},step=epoch)
                episode_rewards.append(episode_reward)
                episode_reward = 0

        # Set next observation as current one
        observation = next_observation

    # Evaluating how well the model work
    windows, window_size = [], params["SCORING_WINDOW_SIZE"]

    for i in range(len(episode_rewards) - window_size + 1):
        windows.append(np.mean(episode_rewards[i:i + window_size]))
    performance = np.max(windows)
    wandb.log({"metric/performance": performance})
    wandb.finish()
    env.close()
    return performance

def normal_train(params):
    wandb.init(project="Cart Pole RL", name=params["EXPERIMENT_NAME"])
    train(params=params)

def hyperopt(device: str, mode: str, agent_id = None):
    session_counter = 0
    def hyperopt_training_loop(config=None):
        nonlocal session_counter
        session_counter += 1
        try:    
            import time
            with wandb.init(config = config, name=f"session_{session_counter}_{round(time.time())}"):
                params = wandb.config
                params["DEVICE"] = device
                params["MODE"] = mode
                params["ARCHITECTURE"] = [4, params["HIDDEN_LAYER_1"], params["HIDDEN_LAYER_2"], 2]
                params = dict(params)
                train(params)
        except Exception as e:
            print(traceback.format_exc())
            raise e
    
    if agent_id is None:
        with open("config/hyperopt_search_space.json", "r") as f:
            config = json.load(f)
            agent_id = wandb.sweep(config, project='Cart Pole RL')
    print(f"Using agent_id = {agent_id}")
    wandb.agent(agent_id, hyperopt_training_loop, count=10000, project='Cart Pole RL')