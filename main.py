import argparse
import yaml
from pathlib import Path
import train
import numpy as np
import torch
import random
from pygit2 import Repository

def apply_random_seed(random_seed: int) -> None:
    """Sets seed to ``random_seed`` in random, numpy and torch."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Cart-Pole trainer', description='Train RL model for cart pole')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--mode', default="human")
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    params = yaml.safe_load(Path(f"config/hyperparameters{Repository('.').head.shorthand}.yaml"))
    params["DEVICE"] = args.device
    params["MODE"] = args.mode
    params["TRAIN"] = args.train
    params["EXPERIMENT_NAME"] = f"experiment_{Repository('.').head.shorthand.zfill(3)}"

    print("#"*100)
    for key, value in params.items():
        key = key.ljust(max(len(k) for k in params)) 
        print(key, value)
    print("#"*100)

    if args.train:
        apply_random_seed(params["RANDOM_SEED"])
        train.train(params=params)
