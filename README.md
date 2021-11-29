# Balloon Learning Environment

The Balloon Learning Environment (BLE) is a simulator for training RL agents
to control stratospheric balloons. It is a followup to the Nature paper
["Autonomous navigation of stratospheric balloons using reinforcement learning"](https://www.nature.com/articles/s41586-020-2939-8).

## Installation

For now, the BLE can be used by cloning the source:

```
git clone https://github.com/google/balloon-learning-environment
```

```
cd balloon-learning-environment
```

We recommend using a virtual environment:

```
python -m venv .venv && source .venv/bin/activate
```

Make sure pip is the latest version:

```
pip install --upgrade pip
```

Install all the prerequisites:

```
pip install -r requirements.txt
```

The BLE internally uses a neural network as part of the environment, so we
recommend installing jax with GPU support.
See the [jax codebase](https://github.com/google/jax#pip-installation-gpu-cuda)
for instructions.

## Training an Agent
The set of agents available to train are listed in the [Agent
Registry](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/agents/agent_registry.py).
You can train one of these with the following command:

```
python -m balloon_learning_environment.train \
  --base_dir=/tmp/ble/train \
  --agent=quantile
```

## Evaluating an Agent

See the [evaluation readme](https://github.com/google/balloon-learning-environment/blob/master/balloon_learning_environment/eval/README.md) for instructions on evaluating an agent.

## Giving credit

If you use the Balloon Learning Environment in your work, we ask that you use
the following BibTeX entry:

```
@software{Greaves_Balloon_Learning_Environment_2021,
  author = {Greaves, Joshua and Candido, Salvatore and Dumoulin, Vincent and Goroshin, Ross and Ponda, Sameera S. and Bellemare, Marc G. and Castro, Pablo Samuel},
  month = {12},
  title = {{Balloon Learning Environment}},
  url = {https://github.com/google/balloon-learning-environment},
  version = {1.0.0},
  year = {2021}
}
```

