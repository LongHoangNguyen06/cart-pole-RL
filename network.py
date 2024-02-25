import torch

import torch.nn
import collections
import numpy as np

class Builder(object):
    def __init__(self, *namespaces):
        self._namespace = collections.ChainMap(*namespaces)
 
    def __call__(self, name, *args, **kwargs):
        try:
            return self._namespace[name](*args, **kwargs)
        except Exception as e:
            raise e.__class__(str(e), name, args, kwargs) from e
 
    def add_namespace(self, namespace, index=-1):
        if index >= 0:
            namespaces = self._namespace.maps
            namespaces.insert(index, namespace)
            self._namespace = collections.ChainMap(*namespaces)
        else:
            self._namespace = self._namespace.new_child(namespace)
 
 
def build_network(architecture, builder=Builder(torch.nn.__dict__)):
    """
    Configuration for feedforward networks is list by nature. We can write 
    this in simple data structures. In yaml format it can look like:
    .. code-block:: yaml
        architecture:
            - Conv2d:
                args: [3, 16, 25]
                stride: 1
                padding: 2
            - ReLU:
                inplace: true
            - Conv2d:
                args: [16, 25, 5]
                stride: 1
                padding: 2
    Note, that each layer is a list with a single dict, this is for readability.
    For example, `builder` for the first block is called like this:
    .. code-block:: python
        first_layer = builder("Conv2d", *[3, 16, 25], **{"stride": 1, "padding": 2})
    the simpliest ever builder is just the following function:
    .. code-block:: python
         def build_layer(name, *args, **kwargs):
            return layers_dictionary[name](*args, **kwargs)
    
    Some more advanced builders catch exceptions and format them in debuggable way or merge 
    namespaces for name lookup
    
    .. code-block:: python
    
        extended_builder = Builder(torch.nn.__dict__, mynnlib.__dict__)
        net = build_network(architecture, builder=extended_builder)
        
    """
    layers = []
    for block in architecture:
        assert len(block) == 1
        name, kwargs = list(block.items())[0]
        if kwargs is None:
            kwargs = {}
        args = kwargs.pop("args", [])
        layers.append(builder(name, *args, **kwargs))
    return torch.nn.Sequential(*layers)

class Network(torch.nn.Module):
    def __init__(self, params):        
        super().__init__()
        # self.net = build_network(architecture=params["ARCHITECTURE"])
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

    def get_train_action(self, x: torch.Tensor):
        if np.random.rand() < self.get_epsilon():
            print("Random action")
            action = np.random.choice(self.params["ACTIONS"])
        else:
            print("Best action")
            action = self.get_best_action(x)
        self.epoch += 1
        return action
    
    def get_best_action(self, x: torch.Tensor):
        return self.params["ACTIONS"][self.net(x).argmax()]