"""
Simple script to compare the weights of two agents.
"""
import numpy as np
from pathlib import Path

from definitions import ROOT_DIR
from balloon_learning_environment.utils import run_helpers


if __name__ == "__main__":
  agent_type = 'quantile'
  ble_action_space = 3
  ble_obs_space = (1099,)

  run_helpers.bind_gin_variables(agent_type,
                                 None,
                                 [])

  # agent1_checkpoint_dir = Path(ROOT_DIR, 'ble_data', 'features_all', agent_type, str(1), 'checkpoints')
  agent1_checkpoint_dir = Path(ROOT_DIR, 'data', 'features_all', agent_type, str(3), 'checkpoints')
  agent1_checkpoint_idx = 0
  agent2_checkpoint_dir = Path(ROOT_DIR, 'data', 'features_all', agent_type, str(3), 'checkpoints')
  agent2_checkpoint_idx = 2

  agent1 = run_helpers.create_agent(
    agent_type,
    ble_action_space,
    observation_shape=ble_obs_space)

  agent1.load_checkpoint(agent1_checkpoint_dir, agent1_checkpoint_idx)
  print("Loaded agent 1")

  agent2 = run_helpers.create_agent(
    agent_type,
    ble_action_space,
    observation_shape=ble_obs_space)

  agent2.load_checkpoint(agent2_checkpoint_dir, agent2_checkpoint_idx)
  print("Loaded agent 2")

  for layer_name, v1 in agent1.online_params['params'].items():
    v2 = agent2.online_params['params'][layer_name]
    for k, w1 in v1.items():
      w2 = v2[k]
      if np.all(w1 == w2):
        print(f"Two network parameters are the same! Layer {layer_name}, weight {k}")
      else:
        print(f"Differing parameters: Layer {layer_name}, weight {k}")

