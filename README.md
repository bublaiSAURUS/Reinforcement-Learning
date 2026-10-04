# Snake RL

A PyTorch reinforcement-learning agent that learns to play Snake through
trial and error. The project packages the game environment, policy, replay
memory, neural network, and training loop as an installable Python library.

## Installation

Requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -e .
```

Install optional live plotting with `python -m pip install -e ".[plot]"`.
The core dependencies are NumPy, Pygame, and PyTorch. A graphical desktop is
required for rendered games; headless training does not initialize a display.

## Training

```bash
# Watch training in the Snake window; Ctrl+C or close the window to stop.
snake-rl

# Run a finite, reproducible initial setup without a display.
snake-rl --headless --episodes 200 --seed 42

# Enable live score and mean-score plots (requires the plot extra).
snake-rl --episodes 100 --plot --checkpoint model/experiment.pth
```

`python -m snake_rl` runs the same command. Use `snake-rl --help` for options.
Without `--episodes`, training continues until interrupted. Rendered training
is capped at 40 frames per second; headless training runs without that cap.
Seeds initialize Python, NumPy, and PyTorch randomness but do not guarantee
identical outcomes across platforms and PyTorch versions.

## Python API

```python
from snake_rl.agent import train

train(episodes=100, render=False, checkpoint="model/run.pth")
```

Use the environment independently:

```python
from snake_rl.game import SnakeGameAI

game = SnakeGameAI(render=False)
try:
    reward, done, score = game.play_step([1, 0, 0])
finally:
    game.close()
```

Actions are one-hot vectors in the order **straight, right turn, left turn**.
After a terminal step, call `game.reset()` before starting another episode.

## Learning behavior

The refactor retains the original network, observation encoding, rewards,
exploration schedule, and update order:

| Setting | Default |
| --- | --- |
| Board | 640 × 480 pixels; 20-pixel cells |
| State | 11 integer features |
| Network | 11 inputs → 256 ReLU units → 3 Q-values |
| Optimizer / loss | Adam / mean squared error |
| Learning rate | 0.001 |
| Discount factor | 0.9 |
| Replay capacity | 100,000 transitions |
| Replay batch | Up to 1,000 transitions per completed game |
| Rewards | +10 for food, −10 for collision or timeout, 0 otherwise |

The state contains danger straight/right/left, the four direction flags, and
four flags describing food position relative to the head. The agent updates
on each transition, remembers it, and trains on a replay batch after each game.
Exploration uses `randint(0, 200) < 80 - n_games`; after 80 games it stops
selecting random actions. The timeout remains the original total episode
frame count exceeding `100 * len(snake)` after inserting the new head.

The original trainer allows gradients through its bootstrapped target. That
behavior is retained to preserve the working implementation; it differs from
the detached target typically used in DQN. There is no separate target network.

## Checkpoints

The best model is saved when an episode's score exceeds the previous record,
starting from a record of zero. Runs that never score above zero do not produce
a checkpoint. Parent directories are created automatically.

Checkpoints contain network weights only. Replay memory, optimizer state,
episode counters, and random-generator state are not saved, so they do not
provide an exact training resume. Layer names and tensor shapes remain
compatible with the original `Linear_QNet` checkpoints.

```python
import torch
from snake_rl.model import Linear_QNet

model = Linear_QNet(11, 256, 3)
model.load_state_dict(torch.load("model/model.pth", weights_only=True))
model.eval()
```

## Repository layout

```text
src/snake_rl/
    agent.py       State encoding, exploration, replay, training orchestration
    game.py        Snake rules and optional renderer
    model.py       Q-network, trainer, weight saving
    plotting.py    Optional score visualization
    cli.py         Training command and seed setup
    __main__.py    python -m snake_rl entry point
test_game.py       Movement, growth, collision, and timeout regression tests
pyproject.toml    Dependencies, build configuration, and command entry point
```

Pygame's built-in font is used so rendering works outside the repository root.
The original `arial.ttf` asset is retained but is no longer required at runtime.

## Development

```bash
python -m pip install -e ".[dev,plot]"
pytest
ruff check .
ruff format --check .
python -m build
```

The game tests run without a display and verify straight movement, turning,
food rewards, growth, wall collisions, and timeout termination. Model weights
and generated build files are excluded from version control.
