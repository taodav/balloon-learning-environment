from pathlib import Path

from scripts.script_helpers import generate_eval_runs
from definitions import ROOT_DIR, HOME_DIR


def write_mean_vs_all_eval_jobs() -> None:
  runs_dir = Path(ROOT_DIR) / 'scripts' / 'mean_vs_all'
  runs_fname = 'mean_vs_all_eval.txt'

  (runs_dir / runs_fname).unlink(missing_ok=True)

  data_dir = Path(HOME_DIR) / 'scratch' / 'ble_data'

  run_dict = {
    'agent': ['quantile'],
    'env_name': ['BalloonLearningEnvironment-v0'],
    'features': ['mean', 'all'],
    'run_number': list(range(1, 11)),
  }
  generate_eval_runs(run_dict, runs_dir, runs_fname, data_dir=data_dir,
                      sub_dir_keys=['features'])

  # TODO: refactor this for if you want to eval a features=all agent on features=mean


if __name__ == "__main__":
  write_mean_vs_all_eval_jobs()
