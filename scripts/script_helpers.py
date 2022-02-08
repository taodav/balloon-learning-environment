from pathlib import Path
from itertools import product
from typing import List

from definitions import ROOT_DIR


def generate_runs(run_dict: dict, runs_dir: Path, runs_fname: str = 'runs.txt',
                  run_command: str = '-m balloon_learning_environment.train',
                  data_dir: Path = Path(ROOT_DIR) / 'data',
                  sub_dir_keys: List[str] = []) -> List[str]:
  """
  :param runs_dir: Directory to put the runs
  :param runs_fname: What do we call our run file?
  :param main_fname: what is our python entry script?
  :return:
  """

  runs_path = runs_dir / runs_fname

  if runs_path.is_file():
    runs_path.unlink()

  f = open(runs_path, 'a+')

  keys, values = [], []
  for k, v in run_dict.items():
    keys.append(k)
    values.append(v)
  num_runs = 0
  for i, args in enumerate(product(*values)):

    arg = {k: v for k, v in zip(keys, args)}

    base_dir = data_dir
    for dir_key in sub_dir_keys:
      base_dir /= f"{dir_key}_{arg[dir_key]}"

    run_string = f"python {run_command}"

    for k, v in arg.items():

      if v is True:
        run_string += f" --{k}"
      elif v is False or v is None:
        continue
      else:
        run_string += f" --{k}={v}"

    run_string += f" --base_dir={base_dir}"
    run_string += "\n"
    f.write(run_string)
    num_runs += 1

    print(num_runs, run_string)
  f.close()


def generate_eval_runs(run_dict: dict, runs_dir: Path, runs_fname: str = 'runs.txt',
                       run_command: str = '-m balloon_learning_environment.eval.eval_checkpoints',
                       data_dir: Path = Path(ROOT_DIR) / 'data',
                       sub_dir_keys: List[str] = []) -> List[str]:
  """
  :param runs_dir: Directory to put the runs
  :param runs_fname: What do we call our run file?
  :param main_fname: what is our python entry script?
  :return:
  """

  runs_path = runs_dir / runs_fname

  if runs_path.is_file():
    runs_path.unlink()

  f = open(runs_path, 'a+')

  keys, values = [], []
  for k, v in run_dict.items():
    keys.append(k)
    values.append(v)
  num_runs = 0
  for i, args in enumerate(product(*values)):

    arg = {k: v for k, v in zip(keys, args)}

    base_dir = data_dir
    for dir_key in sub_dir_keys:
      base_dir /= f"{dir_key}_{arg[dir_key]}"

    base_dir /= arg['agent']
    # base_dir /= str(arg['run_number'])

    run_string = f"python {run_command}"

    for k, v in arg.items():

      if v is True:
        run_string += f" --{k}"
      elif v is False or v is None:
        continue
      else:
        run_string += f" --{k}={v}"

    # TODO: refactor this
    run_string += f" --output_dir={base_dir}"
    run_string += f" --checkpoint_dir={base_dir}\n"
    f.write(run_string)
    num_runs += 1

    print(num_runs, run_string)
  f.close()
