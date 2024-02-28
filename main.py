import argparse
import yaml
from pathlib import Path
import DQL
from pygit2 import Repository

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Cart-Pole trainer', description='Train RL model for cart pole')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--mode', default="human")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--hyper_opt', action='store_true')
    parser.add_argument('--sweep_id', default=None)
    args = parser.parse_args()

    if args.train:
        params = yaml.safe_load(Path(f"config/hyperparameters{Repository('.').head.shorthand}.yaml").read_text())
        params["DEVICE"] = args.device
        params["MODE"] = args.mode
        params["TRAIN"] = args.train
        params["EXPERIMENT_NAME"] = f"experiment_{Repository('.').head.shorthand.zfill(3)}"
        DQL.normal_train(params=params)
    elif args.hyper_opt:
        DQL.hyperopt(device=args.device, mode=args.mode, sweep_id=args.sweep_id)