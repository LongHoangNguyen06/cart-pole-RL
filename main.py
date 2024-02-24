import argparse
import yaml
from pathlib import Path
import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Cart-Pole trainer', description='Train RL model for cart pole')
    parser.add_argument('--architecture')
    parser.add_argument('--hyperparameter')
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--train', action='store_true')  # on/off flag
    args = parser.parse_args()

    architecture = yaml.safe_load(Path(args.architecture).read_text())
    parameters = yaml.safe_load(Path(args.hyperparameter).read_text())
    if args.train:
        train.train(params=parameters, architecture=architecture, device=args.device)
