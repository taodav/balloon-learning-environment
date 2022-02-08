# coding=utf-8
# Copyright 2021 The Balloon Learning Environment Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions used by the main train binary."""

import os
import os.path as osp
import json
from time import time, ctime
from typing import Iterable, List, Optional, Sequence

from absl import logging
from balloon_learning_environment.agents import agent as base_agent
from balloon_learning_environment.env import balloon_env
from balloon_learning_environment.metrics import collector_dispatcher
from balloon_learning_environment.metrics import statistics_instance
from balloon_learning_environment.eval import suites, eval_lib

def get_collector_data(
    collectors: Optional[Iterable[str]] = None
) -> List[collector_dispatcher.CollectorConstructorType]:
  """Returns a list of gin files and constructors for each passed collector."""
  collector_constructors = []
  for c in collectors:
    if c not in collector_dispatcher.AVAILABLE_COLLECTORS:
      continue
    collector_constructors.append(
        collector_dispatcher.AVAILABLE_COLLECTORS[c])
  return collector_constructors


def write_eval_result(result: Sequence[eval_lib.EvaluationResult],
                      eval_dir: str, iteration_number: int) -> None:
  file_name = f'eval_{iteration_number}.json'
  file_path = osp.join(eval_dir, file_name)
  indent = 2

  os.makedirs(eval_dir, exist_ok=True)
  with open(file_path, 'w') as f:
    json.dump(result, f, cls=eval_lib.EvalResultEncoder, indent=indent)


def run_training_loop(
    base_dir: str,
    env: balloon_env.BalloonEnv,
    agent: base_agent.Agent,
    num_episodes: int,
    max_episode_length: int,
    collector_constructors: Sequence[
        collector_dispatcher.CollectorConstructorType],
    *,
    render_period: int = 10,
    checkpoint_period: int = 1,
    eval_period: int = 1,
    eval_size: str = 'tiny_eval') -> None:
  """Runs a training loop for a specified number of steps.

  Args:
    base_dir: The directory to use as the experiment root. This is where
      checkpoints and collector outputs will be written.
    env: The environment to train on.
    agent: The agent to train.
    num_episodes: The number of episodes to train for.
    max_episode_length: The number of episodes at which to end an episode.
    collector_constructors: A sequence of collector constructors for
      collecting and reporting training statistics.
    render_period: The period with which to render the environment. This only
      has an effect if the environments renderer is not None.
  """
  checkpoint_dir = osp.join(base_dir, 'checkpoints')
  # Possibly reload the latest checkpoint, and start from the next episode
  # number.
  start_episode = agent.reload_latest_checkpoint(checkpoint_dir) + 1
  dispatcher = collector_dispatcher.CollectorDispatcher(
      base_dir, env.action_space.n, collector_constructors, start_episode)
  # Maybe pass on a summary writer to the environment.
  env.set_summary_writer(dispatcher.get_summary_writer())
  # Maybe pass on a sumary writer to the agent.
  agent.set_summary_writer(dispatcher.get_summary_writer())

  dispatcher.pre_training()

  time_start = time()
  prev_time = time_start
  avg_time_per_ep = 0

  # # initialize eval functions
  # eval_dir = osp.join(base_dir, 'evals')
  # eval_suite = suites.get_eval_suite(eval_size)
  # eval_result = eval_lib.eval_agent(agent, env, eval_suite,
  #                                   render_period=render_period)
  # write_eval_result(eval_result, eval_dir, 0)

  agent.set_mode(base_agent.AgentMode.TRAIN)

  for episode in range(start_episode, num_episodes):
    dispatcher.begin_episode()
    obs = env.reset()
    # Request first action from agent.
    a = agent.begin_episode(obs)
    terminal = False
    final_episode_step = max_episode_length
    r = 0.0

    for i in range(max_episode_length):
      # Pass action to environment.
      obs, r, terminal, _ = env.step(a)

      if i % render_period == 0:
        env.render()  # No-op if renderer is None.

      # Record the current transition.
      dispatcher.step(statistics_instance.StatisticsInstance(
          step=i,
          action=a,
          reward=r,
          terminal=terminal))

      if terminal:
        final_episode_step = i + 1
        break

      # Pass observation to agent, request new action.
      a = agent.step(r, obs)

    # The environment has no timeout, so terminal really is a terminal state.
    agent.end_episode(r, terminal)

    # Possibly checkpoint the agent.
    if (episode + 1) % checkpoint_period == 0:
      agent.save_checkpoint(checkpoint_dir, episode)

    # TODO(joshgreaves): Fix dispatcher logging the same data twice on terminal.
    dispatcher.end_episode(statistics_instance.StatisticsInstance(
        step=final_episode_step,
        action=a,
        reward=r,
        terminal=terminal))

    curr_time = time()
    time_per_fix_freq = curr_time - prev_time
    avg_time_per_ep += (1 / (episode + 1)) * (time_per_fix_freq - avg_time_per_ep)
    time_remaining = (num_episodes - episode) * avg_time_per_ep
    logging.info(f"Remaining time: {time_remaining / 60:.2f} minutes")
    prev_time = curr_time

    # if episode % eval_period == 0:
    #   eval_result = eval_lib.eval_agent(agent, env, eval_suite,
    #                                     render_period=render_period)
    #   write_eval_result(eval_result, eval_dir, 0)

  dispatcher.end_training()
