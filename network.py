import torch

import torch.nn
import numpy as np

class Network(torch.nn.Module):
    def __init__(self, params):        
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=16, bias=True, device=params["DEVICE"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=16, out_features=8, bias=True, device=params["DEVICE"]),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=8, out_features=2, bias=True, device=params["DEVICE"])
        )
        self.params = params
    
    def forward(self, x):
        return self.net(x)
    
    def mean_weight(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.view(-1))
        all_weights = torch.cat(weights)
        return torch.mean(all_weights)
    
    def std_weight(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.view(-1))
        all_weights = torch.cat(weights)
        return torch.std(all_weights)
    
    def mean_grad(self):
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
        if len(grads) > 0:
            all_grads = torch.cat(grads)
            return torch.mean(all_grads)
        else:
            return 0
    
    def std_grad(self):
        grads = []
        for param in self.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
        if len(grads) > 0:
            all_grads = torch.cat(grads)
            return torch.std(all_grads)
        else:
            return 0


def duplicate(net: Network) -> Network:
    copied_net = Network(params=net.params)
    copied_net.load_state_dict(net.state_dict())
    return copied_net

class ActionInferrer:
    def __init__(self, net: Network, params: dict) -> None:
        self.net = net
        self.params = params
        self.epoch = 0

    def get_epsilon(self):
        return np.clip((1 - self.epoch / self.params["FINAL_GREEDY_EPSILON_EPOCH"]), self.params["FINAL_GREEDY_EPSILON"], 1.0)

    def get_train_action(self, x: np.ndarray):
        if np.random.rand() < self.get_epsilon():
            action = np.random.choice(self.params["ACTIONS"])
        else:
            action = self.get_best_action(x)
        self.epoch += 1
        return action
    
    def get_best_action(self, x: torch.Tensor):
        x = torch.Tensor(x).to(self.params["DEVICE"])
        return self.params["ACTIONS"][self.net(x).argmax()]