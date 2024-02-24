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

def train_iteration(net: network.Network, buff: Buffer, params: dict) -> float:
    pass

def train(params: dict, architecture: dict, device: str):
    net = network.Network(architecture=architecture, params=params, device=device)
    buff = Buffer(params=params)
    env = gym.make('CartPole-v1', render_mode="human")
    env.action_space.seed(params["ACTION_SPACE_SEED"])
    apply_random_seed(params["RANDOM_SEED"])
    
    map_id = 0
    observation, _ = env.reset(seed=map_id)
    losses = []

    for epoch in range(params["TRAINING_EPOCHS"]):
        env.render()
        if epoch >= params["BATCH_SIZE"]:
            loss = train_iteration(net=net, buff=buff, params=params)
            losses.append(loss)
        
        action = net.get_action(x=torch.Tensor(observation).to(device), train=True)
        observation, reward, done, _, _ = env.step(action)
        buff.append(obs=observation, reward=reward)
        if done:
            observation, _ = env.reset()  # Reset the environment if the episode is over
    plt.plot(losses)