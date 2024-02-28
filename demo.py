import torch
import gymnasium as gym
import time

class Network(torch.nn.Module):
    """Policy network's MLP architecture"""
    def __init__(self):        
        super().__init__()
        self.layers = [
                torch.nn.Sequential(
                torch.nn.Linear(in_features=4, out_features=64, bias=True),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(num_features=64)
            ),
            torch.nn.Sequential(
                torch.nn.Linear(in_features=64, out_features=32, bias=True),
                torch.nn.LeakyReLU(),
                torch.nn.BatchNorm1d(num_features=32),
            ),
            torch.nn.Sequential(
                torch.nn.Linear(in_features=32, out_features=2, bias=True),
            )
        ]
        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(torch.Tensor(x))

def demo():
    """Demo mode

    Args:
        params (dict): parameters
    """
    net = Network()
    net.load_state_dict(torch.load("static/model.pth", map_location=torch.device("cpu")))
    env = gym.make('CartPole-v1', render_mode="human")
    net.eval()
    for _ in range(10):
        terminated = False
        reward = 0
        observation, _ = env.reset()
        while not terminated:
            action = int(net(observation.reshape(1, -1)).argmax())
            for _ in range(3):
                observation, re, terminated, truncated, _ = env.step(action)
                env.render()
                reward += re
                time.sleep(0.01)
                if terminated or truncated: break
        print(reward)

demo()