"""Command-line interface for reproducible training runs."""

import argparse
import random


def build_parser():
    parser = argparse.ArgumentParser(description='Train a Q-learning Snake agent.')
    parser.add_argument('--episodes', type=int, help='Stop after this many games.')
    parser.add_argument('--headless', action='store_true', help='Disable rendering.')
    parser.add_argument('--plot', action='store_true', help='Show live score plots.')
    parser.add_argument('--seed', type=int, help='Seed Python, NumPy, and PyTorch.')
    parser.add_argument('--checkpoint', default='model/model.pth')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes is not None and args.episodes < 1:
        parser.error('--episodes must be positive')
    # Defer scientific imports so --help works without the training stack.
    import numpy as np
    import torch

    from .agent import train

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    try:
        train(
            episodes=args.episodes,
            render=not args.headless,
            live_plot=args.plot,
            checkpoint=args.checkpoint,
        )
    except KeyboardInterrupt:
        print('Training stopped.')
